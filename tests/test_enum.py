from ctypes_dlpack import DLDataTypeCode, DLDeviceType


def test_enum_eq():
    assert DLDeviceType.kDLCPU == 1
    assert 1 == DLDeviceType.kDLCPU
    assert DLDeviceType.kDLCUDA == 2
    assert DLDeviceType.kDLCPU != 2

    assert DLDeviceType.kDLCPU == DLDeviceType.kDLCPU
    assert DLDeviceType.kDLCPU != DLDeviceType.kDLCUDA

    # Both have value 1, but different enum types — must not compare equal.
    assert DLDeviceType.kDLCPU != DLDataTypeCode.kDLInt
    assert DLDataTypeCode.kDLInt != DLDeviceType.kDLCPU


def test_enum_hashable():
    table = {DLDeviceType.kDLCPU: "cpu", DLDeviceType.kDLCUDA: "cuda"}
    assert table[DLDeviceType.kDLCPU] == "cpu"
    assert table[1] == "cpu"
    assert table[DLDeviceType.kDLCUDA] == "cuda"

    s = {DLDeviceType.kDLCPU, DLDeviceType.kDLCUDA, DLDeviceType.kDLCPU}
    assert len(s) == 2


def test_enum_repr():
    # ``DLDeviceType.kDLCPU`` is a plain ``int`` (class-body assignment); the
    # custom ``__repr__`` only applies to instances, as produced by ctypes
    # when reading a structure field typed as ``DLDeviceType``.
    assert repr(DLDeviceType(1)) == "DLDeviceType.kDLCPU"
    assert repr(DLDataTypeCode(2)) == "DLDataTypeCode.kDLFloat"

    unknown = DLDeviceType(99)
    assert repr(unknown) == "DLDeviceType.99"

    assert repr(DLDeviceType) == "<Enum DLDeviceType>"
    assert repr(DLDataTypeCode) == "<Enum DLDataTypeCode>"
