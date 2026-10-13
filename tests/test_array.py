import ctypes
import subprocess
import sys
import textwrap

import numpy as np
import torch

from ctypes_dlpack import (
    DLDevice,
    DLDeviceType,
    DLManagedTensorVersioned,
    DLPackArray,
    DLPackVersion,
    array_as_dlpack,
)


def test_numpy_to_ctypes():
    array = np.arange(300).reshape((10, 30))

    ptr = array_as_dlpack(
        array,
        dl_device=DLDevice(DLDeviceType.kDLCPU, 0),
        stream=None,
        max_version=DLPackVersion(1, 0),
    )
    dlpack = DLPackArray(ptr)
    assert np.all(np.from_dlpack(dlpack) == array)

    ptr = array_as_dlpack(
        array,
        dl_device=DLDevice(DLDeviceType.kDLCPU, 0),
        stream=None,
        max_version=DLPackVersion(1, 0),
    )
    dlpack = DLPackArray(ptr)
    assert torch.all(torch.from_dlpack(dlpack) == torch.from_numpy(array))


def test_torch_to_ctypes():
    array = torch.rand(23, 6, 48, dtype=torch.float16)

    # dl_device (1, 0) is CPU:0
    ptr = array_as_dlpack(array, dl_device=(1, 0), stream=None, max_version=(1, 0))
    dlpack = DLPackArray(ptr)
    assert torch.all(torch.from_dlpack(dlpack) == array)

    ptr = array_as_dlpack(array, dl_device=(1, 0), stream=None, max_version=(1, 0))
    dlpack = DLPackArray(ptr)
    assert np.all(np.from_dlpack(dlpack) == array.numpy())


class _KeepAlive:
    def __init__(self, deleted: list):
        # share a reference to the list with the calling code. Since lists are mutable,
        # this allows the calling code to observe when the deleter is called.
        self._deleted = deleted

    def __del__(self):
        self._deleted.append(True)


def test_keepalive_dlpack_array():
    array = np.arange(300).reshape((10, 30))

    # keepalive released when the DLPackArray is garbage collected without being used
    deleted = []
    ptr = array_as_dlpack(array, max_version=(1, 0))
    dlpack = DLPackArray(ptr, keepalive=_KeepAlive(deleted))
    assert deleted == []
    del dlpack
    assert deleted == [True]

    # keepalive released only once the consumer releases the data
    for max_version in [(1, 0), None]:
        deleted = []
        ptr = array_as_dlpack(array, max_version=(1, 0))
        dlpack = DLPackArray(ptr, keepalive=_KeepAlive(deleted))
        if max_version is None:
            tensor = torch.utils.dlpack.from_dlpack(dlpack.__dlpack__())
        else:
            tensor = torch.from_dlpack(dlpack)
        del dlpack
        assert deleted == []

        assert torch.all(tensor == torch.from_numpy(array))
        del tensor
        assert deleted == [True]


def test_keepalive_array_as_dlpack():
    array = torch.rand(23, 6, 48)

    deleted = []
    ptr = array_as_dlpack(array, max_version=(1, 0), keepalive=_KeepAlive(deleted))
    assert deleted == []
    ptr.contents.deleter(ptr)
    assert deleted == [True]

    deleted = []
    ptr = array_as_dlpack(array, max_version=(1, 0), keepalive=_KeepAlive(deleted))
    np_array = np.from_dlpack(DLPackArray(ptr))
    assert deleted == []
    assert np.all(np_array == array.numpy())
    del np_array
    assert deleted == [True]


def test_keepalive_multiple():
    array = np.arange(300).reshape((10, 30))

    deleted_1 = []
    deleted_2 = []
    ptr = array_as_dlpack(array, max_version=(1, 0), keepalive=_KeepAlive(deleted_1))
    dlpack = DLPackArray(ptr, keepalive=_KeepAlive(deleted_2))
    tensor = torch.from_dlpack(dlpack)
    del dlpack
    assert deleted_1 == []
    assert deleted_2 == []

    del tensor
    assert deleted_1 == [True]
    assert deleted_2 == [True]


def test_keepalive_calls_original_deleter():
    events = []

    @ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))
    def deleter(ptr):
        # the original deleter must be restored before being called
        assert ptr.contents.deleter is not None
        events.append("deleter")

    class KeepAlive:
        def __del__(self):
            events.append("keepalive")

    tensor = DLManagedTensorVersioned()
    tensor.deleter = deleter
    dlpack = DLPackArray(ctypes.pointer(tensor), keepalive=KeepAlive())
    assert events == []

    del dlpack
    # the keepalive object is only released after the original deleter ran
    assert events == ["deleter", "keepalive"]
    assert (
        ctypes.cast(tensor.deleter, ctypes.c_void_p).value
        == ctypes.cast(deleter, ctypes.c_void_p).value
    )


def test_shutdown_with_live_arrays():
    # arrays still alive when the interpreter shuts down will call the deleters
    # after the module globals have been cleared.

    code = textwrap.dedent("""
        import sys

        import numpy as np
        import torch

        from ctypes_dlpack import DLPackArray, array_as_dlpack

        class KeepAlive:
            pass

        # store arrays in the sys module. During shutdown, CPython clears the
        # globals of all other modules (including ctypes_dlpack) before clearing
        # sys, so these arrays are guaranteed to outlive the globals of
        # ctypes_dlpack.

        sys.KEEPALIVE = np.from_dlpack(
            DLPackArray(
                array_as_dlpack(np.arange(12), max_version=(1, 0)),
                keepalive=KeepAlive(),
            )
        )

        sys.WRAPPED_UNVERSIONED = np.from_dlpack(
            DLPackArray(array_as_dlpack(torch.arange(12), max_version=(0, 8)))
        )

        sys.WRAPPED_VERSIONED = torch.utils.dlpack.from_dlpack(
            DLPackArray(array_as_dlpack(np.arange(12), max_version=(1, 0))).__dlpack__()
        )

        sys.CAPSULE = DLPackArray(
            array_as_dlpack(np.arange(12), max_version=(1, 0))
        ).__dlpack__(max_version=(1, 0))
        """)

    # run the code in a separate interpreter, and check that it shuts down cleanly
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_shutdown_with_keepalive_cycle():
    # reference cycles through ``keepalive`` are never collected, and the deleter
    # will only be called during interpreter shutdown
    code = textwrap.dedent(
        """
        import numpy as np

        from ctypes_dlpack import DLPackArray, array_as_dlpack

        class Owner:
            pass

        owner = Owner()
        owner.cached = np.from_dlpack(
            DLPackArray(
                array_as_dlpack(np.arange(12), max_version=(1, 0)),
                keepalive=owner,
            )
        )
        del owner
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
