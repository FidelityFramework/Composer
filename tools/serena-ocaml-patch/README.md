# Serena OCaml LSP Patch

This patch adds OCaml language support to Serena's `solidlsp` package via `ocamllsp`.

## Prerequisites

```bash
opam install ocaml-lsp-server ocamlformat
eval $(opam env)
which ocamllsp  # should return a path
```

## Applying the Patch

### Dry Run (see what would change)
```bash
python apply_patch.py --dry-run
```

### Apply the Patch
```bash
python apply_patch.py
```

### Verify Installation
```bash
python apply_patch.py --verify-only
```

## What Gets Patched

1. **`ocaml_language_server.py`** is copied to `solidlsp/language_servers/`
2. **`ls_config.py`** is patched to add:
   - `Language.OCAML` enum entry
   - File extension matcher (`*.ml`, `*.mli`)
   - Language server class mapping

## Using with Serena

After applying the patch, update your project's `.serena/project.yml`:

```yaml
languages:
- ocaml
```

## Supported Features

- Code completion
- Go to definition
- Find references
- Hover information
- Diagnostics
- Document symbols
- Rename
- Code actions

## Reverting the Patch

A backup of `ls_config.py` is created at `ls_config.py.pre-ocaml.bak`. To revert:

```bash
cd /path/to/solidlsp
mv ls_config.py.pre-ocaml.bak ls_config.py
rm language_servers/ocaml_language_server.py
```
