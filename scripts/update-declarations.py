#!/usr/bin/env python
"""Download a DLPack header and generate ctypes declarations."""

import argparse
import pathlib
import re
import tempfile
import urllib.request

import pycparser
from pycparser import c_ast, c_generator, parse_file


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_VERSION = "v1.3"
HEADER_TEMPLATE = (
    "https://raw.githubusercontent.com/dmlc/dlpack/{version}/include/dlpack/dlpack.h"
)
OUTPUT_PATH = ROOT / "src" / "ctypes_dlpack" / "_c_api.py"


GENERATOR = c_generator.CGenerator()

ENUM_HELPERS = """
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
""".strip()


class EnumDef:
    def __init__(self, name: str) -> None:
        self.name = name
        self.values: list[tuple[str, str]] = []


class StructDef:
    def __init__(self, name: str) -> None:
        self.name = name
        self.members: list[tuple[str, c_ast.Node]] = []


class TypedefDef:
    def __init__(self, name: str, typ: c_ast.Node) -> None:
        self.name = name
        self.typ = typ


class HeaderVisitor(c_ast.NodeVisitor):
    def __init__(self) -> None:
        self.enums: list[EnumDef] = []
        self.structs: list[StructDef] = []
        self.typedefs: list[TypedefDef] = []

    def visit_Typedef(self, node: c_ast.Typedef) -> None:
        if not _keep_symbol(node.name):
            return

        inner = node.type

        if isinstance(inner.type, c_ast.Enum):
            enum = EnumDef(node.name)
            enum_values = inner.type.values
            if enum_values is not None:
                for enumerator in enum_values.enumerators:
                    value = "0"
                    if enumerator.value is not None:
                        value = _clean_int_literals(GENERATOR.visit(enumerator.value))
                    enum.values.append((enumerator.name, value))
            self.enums.append(enum)
            return

        if isinstance(inner.type, c_ast.Struct):
            struct = StructDef(node.name)
            declarations = inner.type.decls or []
            for declaration in declarations:
                struct.members.append((declaration.name, declaration.type))
            self.structs.append(struct)
            return

        self.typedefs.append(TypedefDef(node.name, inner.type))


def _clean_int_literals(expression: str) -> str:
    expression = re.sub(r"([0-9A-Fa-fx]+)[uUlL]+", r"\1", expression)
    return expression


def _keep_symbol(name: str | None) -> bool:
    if name is None:
        return False
    return name.startswith("DL") or name.startswith("DLPack")


def _typedecl_name(type_decl: c_ast.TypeDecl) -> str:
    if isinstance(type_decl.type, c_ast.IdentifierType):
        return " ".join(type_decl.type.names)
    if isinstance(type_decl.type, c_ast.Struct):
        return type_decl.type.name
    if isinstance(type_decl.type, c_ast.Enum):
        return type_decl.type.name
    raise ValueError(f"unsupported type declaration {type(type_decl.type)!r}")


def _ctypes_primitive(name: str) -> str:
    primitives = {
        "void": "None",
        "char": "ctypes.c_char",
        "int": "ctypes.c_int",
        "unsigned int": "ctypes.c_uint",
        "int8_t": "ctypes.c_int8",
        "uint8_t": "ctypes.c_uint8",
        "int16_t": "ctypes.c_int16",
        "uint16_t": "ctypes.c_uint16",
        "int32_t": "ctypes.c_int32",
        "uint32_t": "ctypes.c_uint32",
        "int64_t": "ctypes.c_int64",
        "uint64_t": "ctypes.c_uint64",
        "size_t": "ctypes.c_size_t",
    }
    if name in primitives:
        return primitives[name]
    return name


def to_ctypes(node: c_ast.Node) -> str:
    if isinstance(node, c_ast.PtrDecl):
        if isinstance(node.type, c_ast.FuncDecl):
            return to_ctypes(node.type)

        if isinstance(node.type, c_ast.TypeDecl):
            pointee = _typedecl_name(node.type)
            if pointee == "void":
                return "ctypes.c_void_p"
            if pointee == "char":
                return "ctypes.c_char_p"

        inner = to_ctypes(node.type)
        if inner == "None":
            return "ctypes.c_void_p"
        return f"ctypes.POINTER({inner})"

    if isinstance(node, c_ast.ArrayDecl):
        if not isinstance(node.dim, c_ast.Constant):
            raise ValueError("dynamic arrays are not supported")
        return f"{to_ctypes(node.type)} * {node.dim.value}"

    if isinstance(node, c_ast.TypeDecl):
        return _ctypes_primitive(_typedecl_name(node))

    if isinstance(node, c_ast.IdentifierType):
        return _ctypes_primitive(" ".join(node.names))

    if isinstance(node, c_ast.FuncDecl):
        restype = to_ctypes(node.type)
        params: list[str] = []
        if node.args is not None:
            for parameter in node.args.params:
                if isinstance(parameter, c_ast.Typename):
                    params.append(to_ctypes(parameter.type))
                else:
                    params.append(to_ctypes(parameter.type))
        if params == ["None"]:
            params = []
        return f"ctypes.CFUNCTYPE({restype}{', ' if params else ''}{', '.join(params)})"

    raise ValueError(f"unsupported AST node {type(node)!r}")


def parse_defines(header_text: str) -> dict[str, str]:
    defines: dict[str, str] = {}
    for line in header_text.splitlines():
        if not line.startswith("#define "):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        name = parts[1]
        value = parts[2].strip()
        if "(" in name:
            continue
        if not name.startswith("DLPACK_"):
            continue
        if not re.search(r"[0-9]", value):
            continue
        defines[name] = _clean_int_literals(value)
    return defines


def normalize_header(header_text: str) -> str:
    return header_text.replace("typedef enum : int32_t {", "typedef enum {")


def download_header(version: str) -> str:
    url = HEADER_TEMPLATE.format(version=version)
    with urllib.request.urlopen(url) as response:  # noqa: S310
        return response.read().decode("utf8")


def parse_header(header_text: str) -> HeaderVisitor:
    fake_includes = (
        pathlib.Path(pycparser.__file__).resolve().parent
        / "utils"
        / "fake_libc_include"
    )

    with tempfile.TemporaryDirectory() as directory:
        header_file = pathlib.Path(directory) / "dlpack.h"
        header_file.write_text(header_text, encoding="utf8")
        ast = parse_file(
            str(header_file),
            use_cpp=True,
            cpp_path="gcc",
            cpp_args=[
                "-E",
                "-nostdinc",
                f"-I{fake_includes}",
                "-D__attribute__(x)=",
                "-D__extension__=",
            ],
        )

    visitor = HeaderVisitor()
    visitor.visit(ast)
    return visitor


def generate_python_module(defines: dict[str, str], parsed: HeaderVisitor) -> str:
    lines: list[str] = []
    lines.append("# ruff: noqa")
    lines.append('"""Auto-generated ctypes declarations for DLPack."""')
    lines.append("")
    lines.append("import ctypes")
    lines.append("")

    for name in sorted(defines):
        lines.append(f"{name} = {defines[name]}")

    lines.append("")
    lines.extend(ENUM_HELPERS.splitlines())

    for enum in parsed.enums:
        lines.append("")
        lines.append(f"class {enum.name}(Enum):")
        for name, value in enum.values:
            lines.append(f"    {name} = {value}")

    for struct in parsed.structs:
        lines.append("")
        lines.append(f"class {struct.name}(ctypes.Structure):")
        lines.append("    pass")

    for typedef in parsed.typedefs:
        lines.append("")
        lines.append(f"{typedef.name} = {to_ctypes(typedef.typ)}")

    for struct in parsed.structs:
        lines.append("")
        lines.append(f"{struct.name}._fields_ = [")
        for member_name, member_type in struct.members:
            lines.append(f'    ("{member_name}", {to_ctypes(member_type)}),')
        lines.append("]")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=DEFAULT_VERSION, help="DLPack release tag")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="output python file")
    args = parser.parse_args()

    header = download_header(args.version)
    defines = parse_defines(header)
    parsed = parse_header(normalize_header(header))
    generated = generate_python_module(defines, parsed)

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf8")
