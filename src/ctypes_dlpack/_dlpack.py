import ctypes
import sys
from typing import Any

from ._c_api import (
    DLDevice,
    DLManagedTensor,
    DLManagedTensorVersioned,
    DLPackVersion,
    DLTensor,
)


PYTHON_API = ctypes.pythonapi

# Configure Python C-API entry points once, then reuse them from hot paths.
PYTHON_API.PyCapsule_GetPointer.restype = ctypes.c_void_p
PYTHON_API.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PYTHON_API.PyCapsule_SetName.restype = ctypes.c_int
PYTHON_API.PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]

PYTHON_API.PyCapsule_GetName.restype = ctypes.c_char_p
PYTHON_API.PyCapsule_GetName.argtypes = [ctypes.py_object]

PYTHON_API.PyMem_RawMalloc.restype = ctypes.c_void_p
PYTHON_API.PyMem_RawMalloc.argtypes = [ctypes.c_size_t]

PYTHON_API.PyMem_RawFree.restype = None
PYTHON_API.PyMem_RawFree.argtypes = [ctypes.c_void_p]

PYTHON_API.PyCapsule_New.restype = ctypes.py_object
PYTHON_API.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

PYTHON_API.PyErr_Fetch.restype = None
PYTHON_API.PyErr_Fetch.argtypes = [
    ctypes.POINTER(ctypes.py_object),
    ctypes.POINTER(ctypes.py_object),
    ctypes.POINTER(ctypes.py_object),
]

PYTHON_API.PyErr_Clear.restype = None
PYTHON_API.PyErr_Clear.argtypes = []

PYTHON_API.Py_IncRef.restype = None
PYTHON_API.Py_IncRef.argtypes = [ctypes.py_object]


def _immortal(callback):
    """Leak a reference to ``callback`` so it is never garbage-collected.

    The C function pointer of a ``CFUNCTYPE`` object is only valid while the Python
    object is alive. Deleters and capsule destructors can be called by other libraries
    at any time, including during interpreter shutdown after the module globals
    holding the callbacks have been cleared, so the callbacks must outlive the module.

    The callbacks themselves must also check ``sys.is_finalizing()`` (bound as a
    default argument, since module globals may already be cleared) and return early
    during shutdown, as they can not rely on module globals in this case.
    """
    PYTHON_API.Py_IncRef(callback)
    return callback


def _fetch_pyerr():
    """Detach and return the current C-API exception instance, or ``None``.

    ``PyErr_Fetch`` clears the thread-local exception state, so the caller is free to
    raise a fresh exception without CPython discarding the original one.
    """
    etype = ctypes.py_object()
    evalue = ctypes.py_object()
    etb = ctypes.py_object()
    PYTHON_API.PyErr_Fetch(ctypes.byref(etype), ctypes.byref(evalue), ctypes.byref(etb))
    return evalue.value or None


# ``PyCapsule_GetName``/``PyCapsule_GetPointer`` re-bound with a raw ``c_void_p``
# argument. These are needed from inside ``_versioned_capsule_destructor``:
# CPython invokes the destructor during ``PyCapsule dealloc`` while the capsule
# refcount is already 0, so the argument cannot be passed as ``py_object``
# (which would INCREF then DECREF back to 0, recursively re-triggering
# ``dealloc``). The raw pointer is obtained from the cached symbol address.
_PyCapsule_GetName_c_void_p = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)(
    ctypes.cast(PYTHON_API.PyCapsule_GetName, ctypes.c_void_p).value
)

_PyCapsule_GetPointer_c_void_p = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p
)(ctypes.cast(PYTHON_API.PyCapsule_GetPointer, ctypes.c_void_p).value)


DLPACK_VERSIONED_NAME = b"dltensor_versioned"
USED_DLPACK_VERSIONED_NAME = b"used_dltensor_versioned"
DLPACK_NAME = b"dltensor"
USED_DLPACK_NAME = b"used_dltensor"


def _versioned_capsule_destructor(capsule_ptr, _is_finalizing=sys.is_finalizing):
    """Release the ``DLManagedTensorVersioned`` held by a capsule.

    This is invoked by CPython when the capsule is garbage collected without having
    been consumed by a ``__dlpack__`` importer (i.e. its name is still
    ``dltensor_versioned``). Capsules renamed to ``used_dltensor_versioned`` have
    already been taken over by an importer and are left untouched.

    The argument is a raw ``void*`` (the capsule pointer) rather than
    ``py_object`` because CPython invokes the destructor during ``dealloc`` with
    the capsule already at refcount 0: wrapping it as ``py_object`` would INCREF
    then DECREF it back to 0, recursively re-triggering ``dealloc``.
    """
    if _is_finalizing():
        return

    if not capsule_ptr:
        return

    name = _PyCapsule_GetName_c_void_p(capsule_ptr)
    if name != DLPACK_VERSIONED_NAME:
        # Clear any C-API exception (e.g. set by ``GetName`` on failure)
        PYTHON_API.PyErr_Clear()
        return

    ptr = _PyCapsule_GetPointer_c_void_p(capsule_ptr, DLPACK_VERSIONED_NAME)
    if not ptr:
        PYTHON_API.PyErr_Clear()
        return

    versioned_ptr = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensorVersioned))
    versioned = versioned_ptr.contents
    if versioned.deleter:
        versioned.deleter(versioned_ptr)


# Keep the CFUNCTYPE alive forever, since CPython holds the destructor pointer
# inside the capsule.
_VERSIONED_CAPSULE_DESTRUCTOR_C = _immortal(
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_versioned_capsule_destructor)
)


def _unversioned_capsule_destructor(capsule_ptr, _is_finalizing=sys.is_finalizing):
    """Release the ``DLManagedTensor`` held by a legacy capsule.

    See ``_versioned_capsule_destructor`` for why the argument is a raw
    ``void*`` instead of ``py_object``.
    """
    if _is_finalizing():
        return

    if not capsule_ptr:
        return

    name = _PyCapsule_GetName_c_void_p(capsule_ptr)
    if name != DLPACK_NAME:
        # Clear any C-API exception (see ``_versioned_capsule_destructor``).
        PYTHON_API.PyErr_Clear()
        return

    ptr = _PyCapsule_GetPointer_c_void_p(capsule_ptr, DLPACK_NAME)
    if not ptr:
        PYTHON_API.PyErr_Clear()
        return

    unversioned_ptr = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensor))
    unversioned = unversioned_ptr.contents
    if unversioned.deleter:
        unversioned.deleter(unversioned_ptr)


# Keep the CFUNCTYPE alive forever, since CPython holds the destructor pointer
# inside the capsule.
_UNVERSIONED_CAPSULE_DESTRUCTOR_C = _immortal(
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_unversioned_capsule_destructor)
)


@_immortal
@ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))
def _versioned_wrapper_deleter(versioned_ptr, _is_finalizing=sys.is_finalizing):
    """Forward to the legacy deleter, then free the wrapper allocation."""
    if _is_finalizing():
        return

    if not versioned_ptr:
        return

    versioned = versioned_ptr.contents
    original_ptr = ctypes.cast(versioned.manager_ctx, ctypes.POINTER(DLManagedTensor))

    if original_ptr and original_ptr.contents.deleter:
        original_ptr.contents.deleter(original_ptr)

    PYTHON_API.PyMem_RawFree(ctypes.cast(versioned_ptr, ctypes.c_void_p))


@_immortal
@ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
def _unversioned_wrapper_deleter(unversioned_ptr, _is_finalizing=sys.is_finalizing):
    """Forward to the versioned deleter, then free the wrapper allocation."""
    if _is_finalizing():
        return

    if not unversioned_ptr:
        return

    unversioned = unversioned_ptr.contents
    original_ptr = ctypes.cast(
        unversioned.manager_ctx, ctypes.POINTER(DLManagedTensorVersioned)
    )

    if original_ptr and original_ptr.contents.deleter:
        original_ptr.contents.deleter(original_ptr)

    PYTHON_API.PyMem_RawFree(ctypes.cast(unversioned_ptr, ctypes.c_void_p))


def _wrap_unversioned_as_versioned(unversioned_ptr):
    """Wrap a legacy DLManagedTensor pointer in a versioned allocation."""
    versioned_size = ctypes.sizeof(DLManagedTensorVersioned)
    versioned_mem = PYTHON_API.PyMem_RawMalloc(versioned_size)
    if not versioned_mem:
        raise MemoryError("failed to allocate DLManagedTensorVersioned")

    try:
        versioned_ptr = ctypes.cast(
            versioned_mem, ctypes.POINTER(DLManagedTensorVersioned)
        )
        versioned = versioned_ptr.contents

        versioned.version = DLPackVersion()
        versioned.version.major = 1
        versioned.version.minor = 0
        versioned.manager_ctx = ctypes.cast(unversioned_ptr, ctypes.c_void_p)
        versioned.deleter = _versioned_wrapper_deleter
        versioned.flags = 0

        # DLManagedTensor starts with a DLTensor, so copying from the struct base
        # address into versioned.dl_tensor preserves the payload layout.
        ctypes.memmove(
            ctypes.addressof(versioned.dl_tensor),
            unversioned_ptr,
            ctypes.sizeof(DLTensor),
        )

        return versioned_ptr
    except BaseException:
        PYTHON_API.PyMem_RawFree(versioned_mem)
        raise


def _wrap_versioned_as_unversioned(versioned_ptr):
    """Wrap a ``DLManagedTensorVersioned`` pointer in a legacy allocation.

    This is used to expose a versioned tensor to importers that only support the
    unversioned DLPack protocol (``max_version`` is ``None`` or has a major
    version of ``0``).
    """
    unversioned_size = ctypes.sizeof(DLManagedTensor)
    unversioned_mem = PYTHON_API.PyMem_RawMalloc(unversioned_size)
    if not unversioned_mem:
        raise MemoryError("failed to allocate DLManagedTensor")

    try:
        unversioned_ptr = ctypes.cast(unversioned_mem, ctypes.POINTER(DLManagedTensor))
        unversioned = unversioned_ptr.contents

        unversioned.manager_ctx = ctypes.cast(versioned_ptr, ctypes.c_void_p)
        unversioned.deleter = _unversioned_wrapper_deleter

        # Both structs embed a ``DLTensor`` with identical layout; copy the
        # payload from the versioned struct into the legacy one.
        ctypes.memmove(
            ctypes.addressof(unversioned.dl_tensor),
            ctypes.addressof(versioned_ptr.contents.dl_tensor),
            ctypes.sizeof(DLTensor),
        )

        return unversioned_ptr
    except BaseException:
        PYTHON_API.PyMem_RawFree(unversioned_mem)
        raise


# Function pointer type of ``DLManagedTensorVersioned.deleter``
_VERSIONED_DELETER_TYPE = dict(DLManagedTensorVersioned._fields_)["deleter"]

# State for ``_attach_keepalive``, indexed by the address of the
# ``DLManagedTensorVersioned``. Each entry contains the address of the original
# deleter of the tensor (or ``None`` if it was null) and the list of objects to keep
# alive. Entries are removed by ``_keepalive_deleter`` once the consumer releases
# the tensor.
_KEEPALIVE_REGISTRY: dict[int, tuple[int | None, list[Any]]] = {}


@_immortal
@ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))
def _keepalive_deleter(versioned_ptr, _is_finalizing=sys.is_finalizing):
    """Restore and call the original deleter, then release the keepalive objects."""
    if _is_finalizing():
        return

    if not versioned_ptr:
        return

    address = ctypes.cast(versioned_ptr, ctypes.c_void_p).value
    entry = _KEEPALIVE_REGISTRY.pop(address, None)
    if entry is None:
        return

    original_deleter_address, keepalive = entry
    if original_deleter_address is None:
        versioned_ptr.contents.deleter = _VERSIONED_DELETER_TYPE()
    else:
        original_deleter = _VERSIONED_DELETER_TYPE(original_deleter_address)
        versioned_ptr.contents.deleter = original_deleter
        original_deleter(versioned_ptr)

    # only release the keepalive objects once the original tensor is gone
    del entry, keepalive


def _attach_keepalive(versioned_ptr, keepalive):
    """Keep ``keepalive`` alive until the deleter of ``versioned_ptr`` is called.

    This modifies the ``DLManagedTensorVersioned`` in place, replacing its deleter
    with ``_keepalive_deleter``. The original deleter is restored before being
    called, so the producer sees its own struct unchanged.
    """
    address = ctypes.cast(versioned_ptr, ctypes.c_void_p).value
    entry = _KEEPALIVE_REGISTRY.get(address)
    if entry is not None:
        # the deleter was already replaced, add to the existing objects
        entry[1].append(keepalive)
        return

    versioned = versioned_ptr.contents
    # Store the address of the original deleter rather than ``versioned.deleter``
    # itself: ctypes returns a view into the struct memory for this field, which
    # would point to ``_keepalive_deleter`` once we overwrite it below.
    original_deleter_address = ctypes.cast(versioned.deleter, ctypes.c_void_p).value
    _KEEPALIVE_REGISTRY[address] = (original_deleter_address, [keepalive])
    versioned.deleter = _keepalive_deleter


def _make_dlpack_versioned_capsule(dl_managed_versioned_ptr):
    ptr_value = ctypes.cast(dl_managed_versioned_ptr, ctypes.c_void_p).value
    if not ptr_value:
        raise ValueError("DLManagedTensorVersioned pointer is null")

    destructor = ctypes.cast(_VERSIONED_CAPSULE_DESTRUCTOR_C, ctypes.c_void_p)
    return PYTHON_API.PyCapsule_New(ptr_value, DLPACK_VERSIONED_NAME, destructor)


def _make_dlpack_capsule(dl_managed_ptr):
    ptr_value = ctypes.cast(dl_managed_ptr, ctypes.c_void_p).value
    if not ptr_value:
        raise ValueError("DLManagedTensor pointer is null")

    destructor = ctypes.cast(_UNVERSIONED_CAPSULE_DESTRUCTOR_C, ctypes.c_void_p)
    return PYTHON_API.PyCapsule_New(ptr_value, DLPACK_NAME, destructor)


def _call_dlpack_with_fallbacks(array, stream, max_version, dl_device, copy):
    """Try multiple __dlpack__ signatures for broad compatibility.

    This tries both the ``2023.12`` array API signature (with ``max_version``,
    ``dl_device``, and ``copy`` kwargs), as well as the legacy signature (with only
    ``stream``). The first signature that does not raise a ``TypeError`` (due to unknown
    keyword arguments) is used, and the resulting capsule is returned.
    """
    last_error = None
    try:
        try:
            return array.__dlpack__(
                stream=stream,
                max_version=max_version,
                dl_device=dl_device,
                copy=copy,
            )
        except TypeError as error:
            last_error = error

        try:
            return array.__dlpack__(stream=stream)
        except TypeError as error:
            last_error = error

        try:
            return array.__dlpack__()
        except TypeError as error:
            last_error = error

        raise RuntimeError("__dlpack__ call failed") from last_error
    finally:
        # Break the reference cycle ``last_error -> traceback -> this frame ->
        # last_error``, which would otherwise keep the caller's frame (and all
        # objects it references) alive until the next garbage collection.
        last_error = None


class DLPackArray:
    """
    Take a ctypes pointer to a ``DLManagedTensorVersioned`` and wrap it in a Python
    object that implements the ``__dlpack__`` protocol. This allows the pointer to be
    passed to other libraries that support DLPack, while ensuring that the pointer is
    only used once and that the underlying memory is freed when the object is garbage
    collected.

    >>> import numpy as np
    >>> import torch
    >>> # Get the pointer in some way
    >>> pointer = ...
    >>> array = DLPackArray(pointer)
    >>> # the array can be used with `from_dlpack`
    >>> use_numpy = True
    >>> if use_numpy:
    ...     np_array = np.from_dlpack(array)
    ...     print(np_array.shape)
    ... else:
    ...     torch_tensor = torch.from_dlpack(array)
    ...     print(torch_tensor.shape)
    """

    def __init__(
        self,
        pointer: ctypes.POINTER(DLManagedTensorVersioned),
        *,
        keepalive: Any = None,
    ):
        """
        :param pointer: pointer to a ``DLManagedTensorVersioned`` to be re-exposed as
            Python object following the ``__dlpack__`` protocol.
        :param keepalive: arbitrary Python object to keep alive as long as the data in
            ``pointer`` is alive, i.e. until the ``DLManagedTensorVersioned`` deleter is
            called. Objects stored in ``keepalive`` are kept alive by a global registry
            which is not visible to the garbage collector, meaning that reference cycles
            going through ``keepalive`` will never be collected.
        """
        if not isinstance(pointer, ctypes.POINTER(DLManagedTensorVersioned)):
            raise TypeError(
                "`pointer` must be a ctypes.POINTER(DLManagedTensorVersioned), "
                f"got {type(pointer).__name__}"
            )

        if not pointer:
            raise ValueError("`pointer` is null")

        if keepalive is not None:
            _attach_keepalive(pointer, keepalive)

        self._pointer = pointer

    def __del__(self):
        pointer = getattr(self, "_pointer", None)
        if pointer is None:
            return

        self._pointer = None
        if pointer.contents.deleter:
            pointer.contents.deleter(pointer)

    def __dlpack__(
        self,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ):
        pointer = self._pointer
        if pointer is None:
            raise RuntimeError("can not call __dlpack__ twice on DLPackArray")

        if stream is not None:
            raise RuntimeError("only `stream=None` is currently supported")

        if dl_device is not None and dl_device != self.__dlpack_device__():
            raise BufferError("device conversion is not supported")

        if copy is None:
            copy = False

        if copy:
            raise BufferError("only `copy=False` is supported")

        if max_version is None or max_version[0] == 0:
            # The caller only supports the unversioned DLPack protocol (or did
            # not request a specific version): wrap our versioned tensor into a
            # legacy ``DLManagedTensor`` capsule.
            unversioned_ptr = _wrap_versioned_as_unversioned(pointer)
            result = _make_dlpack_capsule(unversioned_ptr)
        else:
            result = _make_dlpack_versioned_capsule(pointer)

        self._pointer = None
        return result

    def __dlpack_device__(self):
        if self._pointer is None:
            raise RuntimeError("can not query __dlpack_device__ after __dlpack__")

        device = self._pointer[0].dl_tensor.device
        return (device.device_type.value, device.device_id)

    def __repr__(self):
        if self._pointer is None:
            return "<DLPackArray (consumed)>"

        return "<DLPackArray>"


def array_as_dlpack(
    array,
    *,
    stream=None,
    max_version: DLPackVersion | tuple[int, int] | None = None,
    dl_device: DLDevice | tuple[int, int] | None = None,
    copy: bool | None = None,
    keepalive: Any = None,
) -> ctypes.POINTER(DLManagedTensorVersioned):
    """
    Get a ctypes pointer to a ``DLManagedTensorVersioned`` struct from any array
    implementing the ``__dlpack__`` protocol. See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
    for the protocol specification.

    This function will try to work with older version of the ``__dlpack__`` protocol,
    wrapping the unversioned ``DLManagedTensor`` returned into a
    ``DLManagedTensorVersioned``. Some arguments might be ignored in this case, since
    they do not have an equivalent.

    :param array: any array type that implements the ``__dlpack__`` protocol
    :param stream: the CUDA or ROCm synchronization stream, passed as-is to the
        ``__dlpack__`` method of ``array``.
    :param max_version: the maximum DLPack version that the caller of this function
        supports. If this is a ``DLPackVersion``, it is converted to the 2-tuple
        expected by ``__dlpack__``.
    :param dl_device: the DLPack device type. Default is ``None``, meaning the exported
        pointer should be on the same device as ``array``. If this is a ``DLDevice``, it
        is converted to the 2-tuple expected by ``__dlpack__``.
    :param copy: boolean indicating whether or not to copy the input.
    :param keepalive: arbitrary Python object to keep alive as long as the returned
        ``DLManagedTensorVersioned`` is alive, i.e. until its deleter is called. Objects
        stored in ``keepalive`` are kept alive by a global registry which is not visible
        to the garbage collector, meaning that reference cycles going through
        ``keepalive`` will never be collected.
    """

    if max_version is None:
        max_version_tuple = None
    elif isinstance(max_version, DLPackVersion):
        max_version_tuple = (max_version.major, max_version.minor)
    elif (
        isinstance(max_version, tuple)
        and len(max_version) == 2
        and isinstance(max_version[0], int)
        and isinstance(max_version[1], int)
    ):
        max_version_tuple = max_version
    else:
        raise TypeError(
            "`max_version` must be a DLPackVersion, a 2-tuple of ints, or None; "
            f"got {type(max_version).__name__}"
        )

    if dl_device is None:
        dl_device_tuple = None
    elif isinstance(dl_device, DLDevice):
        dl_device_tuple = (int(dl_device.device_type.value), dl_device.device_id)
    elif (
        isinstance(dl_device, tuple)
        and len(dl_device) == 2
        and isinstance(dl_device[0], int)
        and isinstance(dl_device[1], int)
    ):
        dl_device_tuple = dl_device
    else:
        raise TypeError(
            "`dl_device` must be a DLDevice, a 2-tuple of ints, or None; "
            f"got {type(dl_device).__name__}"
        )

    capsule = _call_dlpack_with_fallbacks(
        array,
        stream,
        max_version_tuple,
        dl_device_tuple,
        copy,
    )
    capsule_name = PYTHON_API.PyCapsule_GetName(capsule)

    if capsule_name == DLPACK_VERSIONED_NAME:
        pointer = PYTHON_API.PyCapsule_GetPointer(capsule, DLPACK_VERSIONED_NAME)
        if not pointer:
            err = _fetch_pyerr()
            raise RuntimeError("failed to get pointer from versioned capsule") from err

        result = ctypes.cast(pointer, ctypes.POINTER(DLManagedTensorVersioned))

        status = PYTHON_API.PyCapsule_SetName(capsule, USED_DLPACK_VERSIONED_NAME)
        if status != 0:
            err = _fetch_pyerr()
            raise RuntimeError("failed to mark versioned capsule as used") from err

    elif capsule_name == DLPACK_NAME:
        pointer = PYTHON_API.PyCapsule_GetPointer(capsule, DLPACK_NAME)
        if not pointer:
            err = _fetch_pyerr()
            raise RuntimeError("failed to get pointer from legacy capsule") from err

        unversioned_ptr = ctypes.cast(pointer, ctypes.POINTER(DLManagedTensor))
        result = _wrap_unversioned_as_versioned(unversioned_ptr)

        # Only disown the capsule once ``result`` is fully built; if SetName ran
        # earlier and any subsequent step raised, the capsule destructor would
        # see the ``used_`` name and skip the deleter, leaking the source tensor.
        status = PYTHON_API.PyCapsule_SetName(capsule, USED_DLPACK_NAME)
        if status != 0:
            err = _fetch_pyerr()
            raise RuntimeError("failed to mark legacy capsule as used") from err
    else:
        # Unexpected capsule name. If ``GetName`` itself failed (returned NULL)
        # it set a C-API exception which we clear here; otherwise there is
        # nothing to clear.
        PYTHON_API.PyErr_Clear()
        name = capsule_name.decode() if capsule_name else "NULL"
        raise RuntimeError(
            f"unexpected capsule name: {name}; expected 'dltensor' "
            "or 'dltensor_versioned'"
        )

    actual_device = result.contents.dl_tensor.device
    actual_device = (actual_device.device_type.value, actual_device.device_id)
    if dl_device_tuple is not None and actual_device != dl_device_tuple:
        if result.contents.deleter:
            result.contents.deleter(result)

        raise ValueError(
            "DLPack device mismatch: "
            f"expected type={dl_device_tuple[0]} id={dl_device_tuple[1]}, "
            f"got type={actual_device[0]} id={actual_device[1]}"
        )

    if keepalive is not None:
        try:
            _attach_keepalive(result, keepalive)
        except BaseException:
            if result.contents.deleter:
                result.contents.deleter(result)
            raise

    return result
