import jax.numpy as jnp
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


def test_jax_to_ctypes():
    array = jnp.arange(300).reshape((10, 30))

    ptr = array_as_dlpack(
        array,
        dl_device=DLDevice(DLDeviceType.kDLCPU, 0),
        stream=None,
        max_version=DLPackVersion(1, 0),
    )
    dlpack = DLPackArray(ptr)
    assert jnp.all(jnp.from_dlpack(dlpack) == array)

    ptr = array_as_dlpack(
        array,
        dl_device=DLDevice(DLDeviceType.kDLCPU, 0),
        stream=None,
        max_version=DLPackVersion(1, 0),
    )
    dlpack = DLPackArray(ptr)
    assert np.all(np.from_dlpack(dlpack) == np.asarray(array))


def test_cupy_to_ctypes():
    cp = pytest.importorskip("cupy")

    # Check if CUDA is available
    try:
        _ = cp.cuda.Device(0).compute_capability
        # all good
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA is not available")

    array = cp.arange(300, dtype=cp.float32).reshape((10, 30))

    cuda_device = DLDevice(DLDeviceType.kDLCUDA, 0)
    ptr = array_as_dlpack(
        array,
        dl_device=cuda_device,
        stream=None,
        max_version=DLPackVersion(1, 0),
    )
    dlpack = DLPackArray(ptr)
    assert cp.all(cp.from_dlpack(dlpack) == array)

    ptr = array_as_dlpack(
        array,
        dl_device=cuda_device,
        stream=None,
        max_version=DLPackVersion(1, 0),
    )
    dlpack = DLPackArray(ptr)
    torch_array = torch.from_dlpack(dlpack)
    assert torch_array.device.type == "cuda"
    assert torch.all(torch_array == torch.from_dlpack(array))
