#!/bin/bash
# Patches Serena's solidlsp language servers to remove the obsolete
# `self.completions_available.set()` call that was removed from the
# SolidLanguageServer base class.
#
# Run after any Serena upgrade: bash apply_patches.sh

SITE_PACKAGES="/home/hhh/.local/share/uv/tools/serena-agent/lib/python3.14/site-packages"
LS_DIR="$SITE_PACKAGES/solidlsp/language_servers"

# Copy the OCaml language server (full custom file)
cp "$(dirname "$0")/ocaml_language_server.py" "$LS_DIR/ocaml_language_server.py"
echo "Installed: ocaml_language_server.py"

# Patch completions_available out of other custom LS files
for f in mlir_language_server.py llvmir_language_server.py pdll_language_server.py tablegen_language_server.py; do
    if [ -f "$LS_DIR/$f" ] && grep -q "self.completions_available.set()" "$LS_DIR/$f"; then
        python3 -c "
with open('$LS_DIR/$f') as fh:
    content = fh.read()
content = content.replace('        self.completions_available.set()\n', '')
with open('$LS_DIR/$f', 'w') as fh:
    fh.write(content)
"
        echo "Patched: $f"
    else
        echo "Skipped (already patched or missing): $f"
    fi
done

# Clear bytecode cache
rm -f "$LS_DIR/__pycache__"/{ocaml,mlir,llvmir,pdll,tablegen}_language_server.cpython-*.pyc
echo "Cleared bytecode cache"
echo "Done. Restart Serena for changes to take effect."
