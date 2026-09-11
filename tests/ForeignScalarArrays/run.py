#!/usr/bin/env python3
"""Fresh native scalar-reference copy-in/copyback and range rejection gates."""
import json
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import tempfile

source = Path(__file__).resolve().parent
repos = source.parents[2]
composer = Path(sys.argv[1]).resolve()
work = Path(tempfile.mkdtemp(prefix="clef-scalar-arrays-"))
print(work, flush=True)
project = (source / "ForeignScalarArrays.fidproj").read_text()
for dependency in ("Fidelity.Platform.CompilerSurface", "Fidelity.Pthread"):
    relative = f"Fidelity.Platform/Environments/Linux/x86_64/{dependency}.fidproj"
    project = project.replace(json.dumps("../../../" + relative), json.dumps(str(repos / relative)))

def compile_case(name, source_file):
    target = work / name
    target.mkdir()
    (target / "Bindings.clef").write_text((source / "Bindings.clef").read_text())
    (target / "Main.clef").write_text((source / source_file).read_text())
    (target / "ForeignScalarArrays.fidproj").write_text(project)
    with (target / "build.log").open("w") as log:
        result = subprocess.run([str(composer), "compile", str(target / "ForeignScalarArrays.fidproj"),
                                 "-k", "--no-color"], stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        raise RuntimeError(f"{name} compile failed: {target / 'build.log'}")
    return target

def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

positive = compile_case("positive", "Main.clef")
for perturb in ("0", "165"):
    result = subprocess.run([str(positive / "targets/scalar-arrays")], capture_output=True, text=True,
                            timeout=10, preexec_fn=no_core, env={**os.environ, "MALLOC_PERTURB_": perturb})
    (positive / f"run-{perturb}.log").write_text(f"returncode={result.returncode}\n{result.stdout}{result.stderr}")
    assert result.returncode == 0, result
ir = (positive / "targets/intermediates/10_output.mlir").read_text()
for required in ("memref<?xi64>", "memref<?xi8>", "memref<?xi32>", "arith.extui", "arith.extsi", "memref.dealloc"):
    assert required in ir, f"Missing native scalar projection evidence: {required}"
assert re.search(r"func.call @ffi.write\([^\n]*\n\s+memref.dealloc", ir), "Readonly input unexpectedly has copyback operations"
print("positive: full byte copy, readonly input/aliases, unsigned32 and signed32 copyback passed", flush=True)

negative = compile_case("invalid-byte", "InvalidByte.clef")
result = subprocess.run([str(negative / "targets/scalar-arrays")], capture_output=True, text=True,
                        timeout=10, preexec_fn=no_core)
(negative / "run.log").write_text(f"returncode={result.returncode}\n{result.stdout}{result.stderr}")
assert result.returncode in (-signal.SIGABRT, -signal.SIGILL), result
assert not result.stdout, "Invalid input reached write before rejection"
ir = (negative / "targets/intermediates/10_output.mlir").read_text()
assert "Foreign reference bytes element is outside its declared range" in ir
print(f"invalid-byte: out-of-range last element rejected before native write; {work}", flush=True)
