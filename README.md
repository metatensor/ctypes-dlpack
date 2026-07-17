# ctypes-dlpack

<h4 align="center">

[![tests status](https://github.com/metatensor/ctypes-dlpack/actions/workflows/tests.yml/badge.svg)](https://github.com/metatensor/ctypes-dlpack/actions?query=branch%3Amain)
[![documentation](https://img.shields.io/badge/📚_documentation-latest-sucess)](https://docs.metatensor.org/ctypes-dlpack/)
</h4>

The [DLPack](https://dmlc.github.io/dlpack/latest/) standard allows to share
arrays/tensors data between different frameworks. It is defined as a C API, and
the standard way to use it from Python is the `__dlpack__` protocol defined in
the [array
api](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html),
which relies on `PyCapsule` to pass the pointer around through Python code.

However, there might be times where one does not have a `PyCapsule` but just a
raw pointer to the DLPack C struct. In particular, this might be the case when
adding bindings to a C library through `ctypes` instead of going through a
CPython extension module. This library is here to bridge this gap, creating
ctypes pointers from the `__dlpack__` protocol, and making existing pointers
compatible with this protocol.

This library exposes both the C structs/enums/constants from the DLPack C header
as `ctypes` types (containing among other the version of the DLPack library
that was used to generate the ctypes declarations), and a Python API to convert
between DLPack pointers and Python objects. See the [documentation](http://docs.metatensor.org/ctypes-dlpack/) for more details.

You can install the code with `pip install ctypes-dlpack`.

### License and contributions

This project is distributed under the terms of the [BSD license](LICENSE), and
[maintained](https://github.com/lab-cosmo/.github/blob/main/Maintainers.md) by
[@Luthaf](https://github.com/Luthaf), who will reply to issues and pull requests
opened on this repository as soon as possible. You can mention them directly if
you did not receive an answer after a couple of days.
