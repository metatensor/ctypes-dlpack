import numpy as np
import torch

from ctypes_dlpack import (
    DLDevice,
    DLDeviceType,
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
