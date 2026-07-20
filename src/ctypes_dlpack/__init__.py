from ._c_api import (
    DLPACK_FLAG_BITMASK_IS_COPIED,
    DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED,
    DLPACK_FLAG_BITMASK_READ_ONLY,
    DLPACK_MAJOR_VERSION,
    DLPACK_MINOR_VERSION,
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    DLManagedTensor,
    DLManagedTensorVersioned,
    DLPackCurrentWorkStream,
    DLPackDLTensorFromPyObjectNoSync,
    DLPackExchangeAPI,
    DLPackExchangeAPIHeader,
    DLPackManagedTensorAllocator,
    DLPackManagedTensorFromPyObjectNoSync,
    DLPackManagedTensorToPyObjectNoSync,
    DLPackVersion,
    DLTensor,
)
from ._dlpack import DLPackArray, array_as_dlpack


__version__ = "0.1.0"


__all__ = [
    "DLPACK_FLAG_BITMASK_IS_COPIED",
    "DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED",
    "DLPACK_FLAG_BITMASK_READ_ONLY",
    "DLPACK_MAJOR_VERSION",
    "DLPACK_MINOR_VERSION",
    "DLPackManagedTensorAllocator",
    "DLPackManagedTensorFromPyObjectNoSync",
    "DLPackDLTensorFromPyObjectNoSync",
    "DLPackCurrentWorkStream",
    "DLPackManagedTensorToPyObjectNoSync",
    "DLDeviceType",
    "DLDataTypeCode",
    "DLPackVersion",
    "DLDevice",
    "DLDataType",
    "DLTensor",
    "DLManagedTensor",
    "DLManagedTensorVersioned",
    "DLPackExchangeAPIHeader",
    "DLPackExchangeAPI",
    "DLPackArray",
    "array_as_dlpack",
]


for _name in __all__:
    _value = globals()[_name]
    if getattr(_value, "__module__", None) == "ctypes_dlpack._c_api":
        _value.__module__ = __name__


del _name, _value
