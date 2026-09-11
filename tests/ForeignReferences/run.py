#!/usr/bin/env python3
"""Native option/pointer copy-back and packed-layout regression (LLVM/LLD)."""
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import tempfile

source = Path(__file__).resolve().parent
repos = source.parents[2]
composer = Path(sys.argv[1]).resolve()
work = Path(tempfile.mkdtemp(prefix="clef-pointer-cells-"))
project = (source / "PointerCells.fidproj").read_text()
for relative in ["Fidelity.Platform/Environments/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj",
                 "BAREWire/src/BAREWire.BindingMetadata.fidproj"]:
    project = project.replace(json.dumps("../../../" + relative), json.dumps(str(repos / relative)))
(work / "PointerCells.clef").write_text((source / "PointerCells.clef").read_text())
(work / "PointerCells.fidproj").write_text(project)
with (work / "build.log").open("w") as log:
    subprocess.run([str(composer), "compile", str(work / "PointerCells.fidproj"), "-k", "--no-color"],
                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
for perturb in ["0", "165"]:
    subprocess.run([str(work / "targets/pointer-cells")], check=True, timeout=10,
                   env={**os.environ, "MALLOC_PERTURB_": perturb})
llvm = (work / "targets/intermediates/08_output.ll").read_text()
# A packed Option payload begins one byte after its allocated tag. Loads/stores
# through that address must not promise the natural alignment of an integer.
packed = re.findall(r"(%[\w.]+) = getelementptr i8, ptr %[^,]+, i\d+ 1\b", llvm)
assert packed, "No packed option payload accesses found in retained LLVM"
checked = 0
for pointer in packed:
    for access in re.findall(r"[^\n]*(?:load|store)[^\n]*ptr " + re.escape(pointer) + r", align (\d+)\b", llvm):
        assert access == "1", f"Packed payload {pointer} promises alignment {access}"
        checked += 1
assert checked, "No packed payload loads/stores checked"

# A zero-sized output cell must be rejected before the native function can write.
empty = work / "empty"
empty.mkdir()
text = (source / "PointerCells.clef").read_text().split("[<EntryPoint>]")[0]
text += """[<EntryPoint>]
let main _ =
    let cell: option<CHandle<unit>> array = Array.zeroCreate 0
    allocate cell 16 128
"""
(empty / "PointerCells.clef").write_text(text)
(empty / "PointerCells.fidproj").write_text(project)
with (empty / "build.log").open("w") as log:
    subprocess.run([str(composer), "compile", str(empty / "PointerCells.fidproj"), "-k", "--no-color"],
                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
result = subprocess.run([str(empty / "targets/pointer-cells")], capture_output=True, text=True,
                        timeout=10, preexec_fn=no_core)
empty_llvm = (empty / "targets/intermediates/08_output.ll").read_text()
assert result.returncode in [-signal.SIGABRT, -signal.SIGILL], result
# Unoptimized LLVM retains both arms; the known-empty extent takes abort.
guard = re.search(r"br i1 false, label %[^,]+, label %([\w.]+)", empty_llvm)
assert guard, "Missing constant-false extent guard"
failure = re.search(r"^" + re.escape(guard[1]) + r":[^\n]*\n(.*?)(?=^\w+:|^})", empty_llvm, re.M | re.S)
assert failure and "@abort()" in failure[1] and "unreachable" in failure[1], "Empty reference does not take the rejection path"
print(f"PointerCells: native None/Some copy-back, immutable aliases, {checked} packed accesses, and empty reference rejection passed; {work}")
