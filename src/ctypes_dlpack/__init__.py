from . import _c_api
from ._dlpack import DLPackArray, array_as_dlpack


__version__ = "0.1.0"


# expose all public members of the C API at the package level
_public_members = {
    name: getattr(_c_api, name) for name in dir(_c_api) if not name.startswith("_")
}

for name, value in _public_members.items():
    globals()[name] = value

    if getattr(value, "__module__", None) == _c_api.__name__:
        value.__module__ = __name__


__all__ = [
    *_public_members.keys(),
    "DLPackArray",
    "array_as_dlpack",
]


del _public_members
