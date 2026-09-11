"""Native unsigned-index and scalar-zero tests, with poisoned fresh malloc storage."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

source = Path(__file__).resolve().parent
composer = Path(sys.argv[1]).resolve()
platform = source.parents[2] / "Fidelity.Platform/Environments/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj"
work = Path(tempfile.mkdtemp(prefix="clef-memory-arrays-"))
(work / "IndexAndZero.clef").write_text((source / "IndexAndZero.clef").read_text())
project = (source / "IndexAndZero.fidproj").read_text().replace(
    '"../../../Fidelity.Platform/Environments/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj"', json.dumps(str(platform)))
(work / "IndexAndZero.fidproj").write_text(project)
with (work / "compile.log").open("w") as output:
    result = subprocess.run([str(composer), "compile", str(work / "IndexAndZero.fidproj"), "-k", "--no-color"],
                            stdout=output, stderr=subprocess.STDOUT, timeout=240)
if result.returncode:
    raise SystemExit(f"Compile failed: {work / 'compile.log'}")
mlir = (work / "targets/intermediates/10_output.mlir").read_text()
assert "index.castu" in mlir
assert "scf.while" in mlir
def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
for poison in ("0", "165"):
    # glibc's test facility makes missing initialization observable even on a
    # fresh process's first allocations; this is local to the child executable.
    environment = dict(os.environ, MALLOC_PERTURB_=poison)
    result = subprocess.run([str(work / "targets/index-and-zero")], env=environment,
                            preexec_fn=no_core, timeout=10)
    if result.returncode:
        raise SystemExit(f"Native memory gate failed {result.returncode}, poison={poison}: {work}")
print(f"IndexAndZero: PASS (fresh and poisoned allocator storage); artifacts {work}")
