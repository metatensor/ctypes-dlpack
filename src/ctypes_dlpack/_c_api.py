# ruff: noqa
"""Auto-generated ctypes declarations for DLPack."""

import ctypes

DLPACK_FLAG_BITMASK_IS_COPIED = 1 << 1
DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED = 1 << 2
DLPACK_FLAG_BITMASK_READ_ONLY = 1 << 0
DLPACK_MAJOR_VERSION = 1
DLPACK_MINOR_VERSION = 3


class EnumType(type(ctypes.c_int32)):
    def __new__(metacls, name, bases, namespace):
        if "_members_" not in namespace:
            members = {}
            for key, value in namespace.items():
                if not key.startswith("_"):
                    members[key] = value
            namespace["_members_"] = members
        else:
            members = namespace["_members_"]

        namespace["_reverse_map_"] = {v: k for k, v in members.items()}
        cls = type(ctypes.c_int32).__new__(metacls, name, bases, namespace)
        for key, value in cls._members_.items():
            globals()[key] = value
        return cls

    def __repr__(self):
        return f"<Enum {self.__name__}>"


class Enum(ctypes.c_int32, metaclass=EnumType):
    _members_ = {}

    def __repr__(self):
        value_name = self._reverse_map_.get(self.value, str(self.value))
        return f"{self.__class__.__name__}.{value_name}"

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other

        return type(self) is type(other) and self.value == other.value


class DLDeviceType(Enum):
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16
    kDLMAIA = 17
    kDLTrn = 18


class DLDataTypeCode(Enum):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6
    kDLFloat8_e3m4 = 7
    kDLFloat8_e4m3 = 8
    kDLFloat8_e4m3b11fnuz = 9
    kDLFloat8_e4m3fn = 10
    kDLFloat8_e4m3fnuz = 11
    kDLFloat8_e5m2 = 12
    kDLFloat8_e5m2fnuz = 13
    kDLFloat8_e8m0fnu = 14
    kDLFloat6_e2m3fn = 15
    kDLFloat6_e3m2fn = 16
    kDLFloat4_e2m1fn = 17


class DLPackVersion(ctypes.Structure):
    pass


class DLDevice(ctypes.Structure):
    pass


class DLDataType(ctypes.Structure):
    pass


class DLTensor(ctypes.Structure):
    pass


class DLManagedTensor(ctypes.Structure):
    pass


class DLManagedTensorVersioned(ctypes.Structure):
    pass


class DLPackExchangeAPIHeader(ctypes.Structure):
    pass


class DLPackExchangeAPI(ctypes.Structure):
    pass


DLPackManagedTensorAllocator = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(DLTensor),
    ctypes.POINTER(ctypes.POINTER(DLManagedTensorVersioned)),
    ctypes.c_void_p,
    ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p),
)

DLPackManagedTensorFromPyObjectNoSync = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(DLManagedTensorVersioned)),
)

DLPackDLTensorFromPyObjectNoSync = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(DLTensor)
)

DLPackCurrentWorkStream = ctypes.CFUNCTYPE(
    ctypes.c_int, DLDeviceType, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)
)

DLPackManagedTensorToPyObjectNoSync = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(DLManagedTensorVersioned),
    ctypes.POINTER(ctypes.c_void_p),
)

DLPackVersion._fields_ = [
    ("major", ctypes.c_uint32),
    ("minor", ctypes.c_uint32),
]

DLDevice._fields_ = [
    ("device_type", DLDeviceType),
    ("device_id", ctypes.c_int32),
]

DLDataType._fields_ = [
    ("code", ctypes.c_uint8),
    ("bits", ctypes.c_uint8),
    ("lanes", ctypes.c_uint16),
]

DLTensor._fields_ = [
    ("data", ctypes.c_void_p),
    ("device", DLDevice),
    ("ndim", ctypes.c_int32),
    ("dtype", DLDataType),
    ("shape", ctypes.POINTER(ctypes.c_int64)),
    ("strides", ctypes.POINTER(ctypes.c_int64)),
    ("byte_offset", ctypes.c_uint64),
]

DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))),
]

DLManagedTensorVersioned._fields_ = [
    ("version", DLPackVersion),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensorVersioned))),
    ("flags", ctypes.c_uint64),
    ("dl_tensor", DLTensor),
]

DLPackExchangeAPIHeader._fields_ = [
    ("version", DLPackVersion),
    ("prev_api", ctypes.POINTER(DLPackExchangeAPIHeader)),
]

DLPackExchangeAPI._fields_ = [
    ("header", DLPackExchangeAPIHeader),
    ("managed_tensor_allocator", DLPackManagedTensorAllocator),
    ("managed_tensor_from_py_object_no_sync", DLPackManagedTensorFromPyObjectNoSync),
    ("managed_tensor_to_py_object_no_sync", DLPackManagedTensorToPyObjectNoSync),
    ("dltensor_from_py_object_no_sync", DLPackDLTensorFromPyObjectNoSync),
    ("current_work_stream", DLPackCurrentWorkStream),
]
