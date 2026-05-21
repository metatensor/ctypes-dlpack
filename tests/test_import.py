def test_import_version() -> None:
    import ctypes_dlpack

    assert ctypes_dlpack.__version__ == "0.1.0"
