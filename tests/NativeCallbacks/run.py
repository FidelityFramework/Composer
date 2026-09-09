#!/usr/bin/env python3
"""Compile typed callbacks through MLIR/LLVM/LLD and execute a fresh artifact.

Usage: python3 tests/NativeCallbacks/run.py src/bin/Debug/net10.0/Composer
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

composer = Path(sys.argv[1]).resolve()
source = Path(__file__).resolve().parent
platform = source.parents[2] / "Fidelity.Platform/CPU/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj"
for name, source_file, executable, operations in [
    ("NativeCallbacks", "Main.clef", "native-callbacks", [
        "func.constant @NativeCallbacks.add", "func.constant @NativeCallbacks.subtract", "func.call_indirect"]),
    ("CapturedBuffers", "CapturedBuffers.clef", "captured-buffers", ["func.call_indirect", "memref.dim"]),
    ("CapturedRecords", "CapturedRecords.clef", "captured-records", ["func.call_indirect", "memref.extract_aligned_pointer_as_index"]),
    ("FunctionSnapshots", "FunctionSnapshots.clef", "function-snapshots", ["func.call_indirect"]),
    ("FunctionFields", "FunctionFields.clef", "function-fields", ["func.call_indirect"]),
    ("ListenerEntry", "ListenerEntry.clef", "listener-entry", [
        "func.constant @__clef_callback_", " : (index, i32) -> ()",
        "func.call @ListenerEntry.onDone", "func.call_indirect"]),
    ("IgnoreValues", "IgnoreValues.clef", "ignore-values", [
        "func.call @IgnoreValues.numeric", "func.call @IgnoreValues.optional", "func.call @IgnoreValues.consumeUnit"]),
]:
    with tempfile.TemporaryDirectory(prefix="clef-native-callbacks-") as directory:
        work = Path(directory)
        (work / source_file).write_text((source / source_file).read_text())
        project = (source / f"{name}.fidproj").read_text().replace(
            '"../../../Fidelity.Platform/CPU/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj"', json.dumps(str(platform)))
        project = project.replace('"../../../Fidelity.Platform/CPU/Linux/x86_64/Fidelity.Pthread.fidproj"',
                                  json.dumps(str(platform.with_name("Fidelity.Pthread.fidproj"))))
        (work / f"{name}.fidproj").write_text(project)
        compiled = subprocess.run(
            [str(composer), "compile", str(work / f"{name}.fidproj"), "-k", "--no-color"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        if compiled.returncode:
            print(compiled.stdout, file=sys.stderr)
            sys.exit(compiled.returncode)
        mlir = (work / "targets/intermediates/10_output.mlir").read_text()
        for operation in operations:
            assert operation in mlir, f"Missing compiler-owned callback operation: {operation}"
        subprocess.run([str(work / f"targets/{executable}")], check=True, timeout=10)
    print(f"{name}: passed", flush=True)
