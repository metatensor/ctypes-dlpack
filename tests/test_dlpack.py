import ctypes
import gc

import numpy as np
import pytest
import torch

from ctypes_dlpack import (
    DLDevice,
    DLDeviceType,
    DLPackArray,
    DLPackVersion,
    array_as_dlpack,
)
from ctypes_dlpack._c_api import DLManagedTensorVersioned


_Deleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))


def create_numpy_array():
    return np.random.rand(2, 3, 4)


def create_torch_array():
    return torch.rand(23, 6, 48, dtype=torch.float16)


@pytest.mark.parametrize("array", [create_numpy_array(), create_torch_array()])
def test_unconsumed_capsule_runs_deleter(array):
    """A capsule produced by ``DLPackArray.__dlpack__`` but never consumed by an
    importer must still release the underlying ``DLManagedTensorVersioned`` via
    the capsule destructor when garbage collected."""

    calls = []
    keepalive = []

    def tracking_deleter(original):
        # Extract the raw function-pointer address: ``_Deleter(original)``
        # would treat ``original`` as a Python callable and wrap it in a
        # trampoline that calls back into ``original`` (recursing forever).
        original_addr = ctypes.cast(original, ctypes.c_void_p).value
        original_cb = _Deleter(original_addr)

        def deleter(ptr):
            calls.append(1)
            if original_cb:
                original_cb(ptr)

        # Keep the trampoline alive: only the raw function pointer is stored in
        # the struct, so the Python-side ``_Deleter`` must be retained.
        cb = _Deleter(deleter)
        keepalive.append(cb)
        return cb

    ptr = array_as_dlpack(
        array,
        dl_device=DLDevice(DLDeviceType.kDLCPU, 0),
        stream=None,
        max_version=DLPackVersion(1, 0),
    )
    # Replace the original deleter with a tracking one to observe the call.
    ptr[0].deleter = tracking_deleter(ptr[0].deleter)

    dlpack = DLPackArray(ptr)
    capsule = dlpack.__dlpack__()
    assert len(calls) == 0

    del capsule
    gc.collect()
    assert len(calls) == 1
