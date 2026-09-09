#!/usr/bin/env python3
"""Check HelloDimensionsProof's retained native artifacts and execute its binary.

After `composer compile <HelloDimensionsProof.fidproj> -k`, run:
  python3 tests/StaticStorageNativeRegression.py <sample-directory>/targets

The middle-end correspondence gate checks the PSG -> MLIR boundary. This test
checks that the exact pool bytes/extent/alignment survive LLVM and ELF emission.
Absolute addresses are inspected in this artifact, not asserted as source facts.
Requires cvc5 and mlir-translate; the ELF check currently targets ELF64 little endian.
"""
import json
from pathlib import Path
import re
import struct
import subprocess
import sys


def run(tool, *args, input_text):
    return subprocess.run(
        [tool, *args], input=input_text, text=True, capture_output=True, check=True
    ).stdout


def check(targets):
    intermediates = targets / "intermediates"
    mlir = (intermediates / "10_output.mlir").read_text()
    llvm = (intermediates / "08_output.ll").read_text()
    symbol = "__clef_static_strings"
    declarations = [
        line for line in mlir.splitlines()
        if "memref.global" in line and f"@{symbol} :" in line
    ]
    assert len(declarations) == 1, "expected one settled pool allocation"
    declaration = declarations[0]
    expected = bytes(int(x) for x in re.search(r"dense<\[([^]]+)\]>", declaration)[1].split(","))
    alignment = int(re.search(r"alignment = (\d+) : i64", declaration)[1])
    assert f"memref<{len(expected)}xi8>" in declaration
    assert '"layout_user_strings"' in declaration, "layout anchor lost in MLIR"
    definitions = [line for line in llvm.splitlines() if line.startswith(f"@{symbol} =")]
    assert len(definitions) == 1
    assert f"[{len(expected)} x i8]" in definitions[0] and f"align {alignment}" in definitions[0]

    executable = targets / "hello-dimensions-proof"
    blob = executable.read_bytes()
    assert blob[:6] == b"\x7fELF\x02\x01", "test requires ELF64 little endian"
    section_offset = struct.unpack_from("<Q", blob, 40)[0]
    section_size, section_count = struct.unpack_from("<HH", blob, 58)
    matches = []
    for index in range(section_count):
        _, kind, flags, address, offset, size, _, _, _, _ = struct.unpack_from(
            "<IIQQQQIIQQ", blob, section_offset + section_size * index
        )
        if kind != 1 or not flags & 2:  # Only allocated PROGBITS sections.
            continue
        data = blob[offset:offset + size]
        start = 0
        while (position := data.find(expected, start)) >= 0:
            matches.append((address + position, flags))
            start = position + 1
    assert len(matches) == 1, "exact pool bytes must occur once in allocated ELF data"
    address, flags = matches[0]
    assert address % alignment == 0, "linker failed to preserve pool alignment"
    assert flags & 1 == 0, "pool is in a writable section"
    # The loader enforces program-header permissions, not section flags.
    program_offset = struct.unpack_from("<Q", blob, 32)[0]
    program_size, program_count = struct.unpack_from("<HH", blob, 54)
    readable_backing = False
    ending = address + len(expected)
    for index in range(program_count):
        kind, permissions, file_offset, virtual_address, _, file_size, memory_size, _ = struct.unpack_from(
            "<IIQQQQQQ", blob, program_offset + program_size * index
        )
        if kind != 1:  # PT_LOAD
            continue
        if address < virtual_address + memory_size and virtual_address < ending:
            assert not permissions & 2, "pool overlaps a writable load segment"
        if virtual_address <= address and ending <= virtual_address + file_size:
            assert permissions & 4, "pool load segment is not readable"
            backing_offset = file_offset + address - virtual_address
            assert blob[backing_offset:backing_offset + len(expected)] == expected
            readable_backing = True
    assert readable_backing, "pool is not wholly backed by a readable file-backed load segment"

    source = run("cvc5", "--lang=smt2", input_text=(intermediates / "06b_obligations.smt2").read_text()).split()
    native_query = run("mlir-translate", "--export-smtlib", input_text=(intermediates / "09_obligations.mlir").read_text())
    native = run("cvc5", "--lang=smt2", input_text=native_query).split()
    assert set(source) == {"unsat"} and source == native, "source/native proof dispatch differs"
    result = subprocess.run([executable], input=b"Ada\n", capture_output=True, check=True)
    assert result.stdout == b"Enter your name: Hello, Ada!\n", result.stdout
    return {"poolBytes": len(expected), "alignment": alignment, "ELFAddress": hex(address),
            "exactBytesInReadonlyStorage": True, "readonlyLoadSegment": True, "sourceAndNativeUnsat": len(source),
            "programOutput": result.stdout.decode()}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(check(Path(sys.argv[1]).resolve()), indent=2))
