#!/usr/bin/env python3
"""
Patch script to add OCaml language support to Serena's solidlsp package.

This script:
1. Copies ocaml_language_server.py to solidlsp/language_servers/
2. Patches ls_config.py to add the Language.OCAML enum entry, file matcher, and LS class mapping

Usage:
    python apply_patch.py [--dry-run] [--serena-path PATH] [--verify-only]

Options:
    --dry-run       Show what would be done without making changes
    --serena-path   Path to solidlsp installation (auto-detected if not specified)
    --verify-only   Only verify the installation, don't apply patches
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def find_serena_path() -> Path | None:
    """Find the solidlsp package path within the serena installation."""
    # Try to find via uv tools
    uv_tools_path = Path.home() / ".local/share/uv/tools/serena-agent"
    if uv_tools_path.exists():
        lib_path = uv_tools_path / "lib"
        if lib_path.exists():
            for item in lib_path.iterdir():
                if item.name.startswith("python"):
                    solidlsp_path = item / "site-packages" / "solidlsp"
                    if solidlsp_path.exists():
                        return solidlsp_path

    # Try to find via python import
    try:
        result = subprocess.run(
            ["python3", "-c", "import solidlsp; print(solidlsp.__file__)"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            solidlsp_init = Path(result.stdout.strip())
            return solidlsp_init.parent
    except Exception:
        pass

    return None


def get_patch_dir() -> Path:
    """Get the directory containing this script and the patch files."""
    return Path(__file__).parent


def copy_language_server(solidlsp_path: Path, dry_run: bool = False) -> bool:
    """Copy the OCaml language server file to solidlsp/language_servers/."""
    patch_dir = get_patch_dir()
    ls_dir = solidlsp_path / "language_servers"

    src = patch_dir / "ocaml_language_server.py"
    dst = ls_dir / "ocaml_language_server.py"

    if not src.exists():
        print(f"  ERROR: Source file not found: {src}")
        return False

    if dry_run:
        print(f"  Would copy: {src} -> {dst}")
    else:
        shutil.copy2(src, dst)
        print(f"  Copied: ocaml_language_server.py")

    return True


def patch_ls_config(solidlsp_path: Path, dry_run: bool = False) -> bool:
    """Patch ls_config.py to add OCaml language entry."""
    ls_config_path = solidlsp_path / "ls_config.py"

    if not ls_config_path.exists():
        print(f"  ERROR: ls_config.py not found at {ls_config_path}")
        return False

    content = ls_config_path.read_text()
    original_content = content

    # Check if already patched
    if 'OCAML = "ocaml"' in content:
        print("  ls_config.py already contains OCAML entry, nothing to do")
        return True

    # 1. Add Language enum entry after TOML (last current entry before @classmethod)
    toml_pattern = r'(TOML = "toml"\s*\n\s*"""[^"]*""")'
    toml_match = re.search(toml_pattern, content)

    if toml_match:
        new_entry = '''
    OCAML = "ocaml"
    """OCaml language server using ocamllsp.
    Provides code completion, go-to-definition, find references, hover, diagnostics.
    Requires: opam install ocaml-lsp-server
    """'''
        insert_pos = toml_match.end()
        content = content[:insert_pos] + new_entry + content[insert_pos:]
        print("  Added OCAML to Language enum")
    else:
        print("  WARNING: Could not find TOML entry in Language enum to insert after")
        return False

    # 2. Add file matcher in get_source_fn_matcher()
    # Insert after LLVM_IR matcher (which is near the other patched entries)
    llvmir_matcher_pattern = r'(case self\.LLVM_IR:\s*\n\s*return FilenameMatcher\("\*\.ll"\))'
    llvmir_matcher_match = re.search(llvmir_matcher_pattern, content)

    if llvmir_matcher_match:
        new_matcher = '''
            case self.OCAML:
                return FilenameMatcher("*.ml", "*.mli")'''
        insert_pos = llvmir_matcher_match.end()
        content = content[:insert_pos] + new_matcher + content[insert_pos:]
        print("  Added file matcher for OCAML (*.ml, *.mli)")
    else:
        # Fallback: insert after HASKELL matcher
        haskell_pattern = r'(case self\.HASKELL:\s*\n\s*return FilenameMatcher\("[^"]*"[^)]*\))'
        haskell_match = re.search(haskell_pattern, content)
        if haskell_match:
            new_matcher = '''
            case self.OCAML:
                return FilenameMatcher("*.ml", "*.mli")'''
            insert_pos = haskell_match.end()
            content = content[:insert_pos] + new_matcher + content[insert_pos:]
            print("  Added file matcher for OCAML after HASKELL")
        else:
            print("  WARNING: Could not find insertion point for file matcher")
            return False

    # 3. Add to get_ls_class() method
    # Insert after LLVM_IR class mapping (or after FSHARP if LLVM_IR not found)
    llvmir_class_pattern = r'(case self\.LLVM_IR:\s*\n\s*from solidlsp\.language_servers\.llvmir_language_server import LLVMIRLanguageServer\s*\n\s*return LLVMIRLanguageServer)'
    llvmir_class_match = re.search(llvmir_class_pattern, content)

    if llvmir_class_match:
        new_class = '''
            case self.OCAML:
                from solidlsp.language_servers.ocaml_language_server import OCamlLanguageServer

                return OCamlLanguageServer'''
        insert_pos = llvmir_class_match.end()
        content = content[:insert_pos] + new_class + content[insert_pos:]
        print("  Added OCaml language server class mapping")
    else:
        # Fallback: insert after FSHARP
        fsharp_class_pattern = r'(case self\.FSHARP:\s*\n\s*from solidlsp\.language_servers\.fsharp_language_server import FSharpLanguageServer\s*\n\s*return FSharpLanguageServer)'
        fsharp_match = re.search(fsharp_class_pattern, content)
        if fsharp_match:
            new_class = '''
            case self.OCAML:
                from solidlsp.language_servers.ocaml_language_server import OCamlLanguageServer

                return OCamlLanguageServer'''
            insert_pos = fsharp_match.end()
            content = content[:insert_pos] + new_class + content[insert_pos:]
            print("  Added OCaml language server class mapping after FSHARP")
        else:
            print("  WARNING: Could not find insertion point for LS class mapping")
            return False

    # Write the patched content
    if content != original_content:
        if dry_run:
            print(f"\n  Would patch: {ls_config_path}")
            print("\n  --- Changes preview ---")
            import difflib
            diff = difflib.unified_diff(
                original_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile="ls_config.py.orig",
                tofile="ls_config.py",
            )
            for i, line in enumerate(diff):
                if i >= 80:
                    print("  ... (truncated)")
                    break
                print(f"  {line}", end="")
        else:
            # Backup original
            backup_path = ls_config_path.with_suffix(".py.pre-ocaml.bak")
            shutil.copy2(ls_config_path, backup_path)
            print(f"  Backed up original to: {backup_path}")

            ls_config_path.write_text(content)
            print(f"  Patched: {ls_config_path}")
        return True
    else:
        print("  No changes needed to ls_config.py")
        return True


def verify_installation(solidlsp_path: Path) -> bool:
    """Verify that the patch was applied correctly."""
    print("\nVerifying installation...")

    # Check language server file exists
    ls_file = solidlsp_path / "language_servers" / "ocaml_language_server.py"
    if not ls_file.exists():
        print(f"  ✗ ocaml_language_server.py not found")
        return False
    else:
        print(f"  ✓ ocaml_language_server.py exists")

    # Try to import and verify
    try:
        sys.path.insert(0, str(solidlsp_path.parent))

        import importlib
        if "solidlsp.ls_config" in sys.modules:
            importlib.reload(sys.modules["solidlsp.ls_config"])

        from solidlsp.ls_config import Language

        if hasattr(Language, "OCAML"):
            print(f"  ✓ Language.OCAML present")
        else:
            print(f"  ✗ Language.OCAML missing")
            return False

        # Check file matcher
        ocaml_matcher = Language.OCAML.get_source_fn_matcher()
        assert ocaml_matcher.is_relevant_filename("test.ml"), "*.ml matcher failed"
        assert ocaml_matcher.is_relevant_filename("test.mli"), "*.mli matcher failed"
        assert not ocaml_matcher.is_relevant_filename("test.py"), "false positive"
        print("  ✓ File matchers work (*.ml, *.mli)")

        # Check LS class import
        ocaml_class = Language.OCAML.get_ls_class()
        assert ocaml_class.__name__ == "OCamlLanguageServer", "OCaml LS class import failed"
        print("  ✓ OCamlLanguageServer class importable")

        # Check ocamllsp binary
        ocamllsp = shutil.which("ocamllsp")
        if ocamllsp:
            print(f"  ✓ ocamllsp found at {ocamllsp}")
        else:
            # Try opam path
            try:
                import subprocess
                result = subprocess.run(["opam", "var", "bin"], capture_output=True, text=True, check=True)
                opam_bin = result.stdout.strip()
                candidate = Path(opam_bin) / "ocamllsp"
                if candidate.exists():
                    print(f"  ✓ ocamllsp found at {candidate} (via opam)")
                else:
                    print(f"  ⚠ ocamllsp not in PATH (run: eval $(opam env))")
            except Exception:
                print(f"  ⚠ ocamllsp not in PATH (run: eval $(opam env))")

        print("\n✓ All verifications passed!")
        return True

    except Exception as e:
        print(f"  ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Patch Serena's solidlsp to add OCaml language support"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--serena-path",
        type=Path,
        help="Path to solidlsp package (auto-detected if not specified)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the installation, don't apply patches",
    )

    args = parser.parse_args()

    if args.serena_path:
        solidlsp_path = args.serena_path
    else:
        solidlsp_path = find_serena_path()

    if not solidlsp_path or not solidlsp_path.exists():
        print("ERROR: Could not find solidlsp package.")
        print("Please specify the path with --serena-path")
        sys.exit(1)

    print(f"Found solidlsp at: {solidlsp_path}")

    if args.verify_only:
        success = verify_installation(solidlsp_path)
        sys.exit(0 if success else 1)

    if args.dry_run:
        print("\n=== DRY RUN MODE ===\n")

    # Step 1: Copy language server file
    print("\nStep 1: Copying OCaml language server file...")
    copied = copy_language_server(solidlsp_path, dry_run=args.dry_run)
    if not copied:
        sys.exit(1)

    # Step 2: Patch ls_config.py
    print("\nStep 2: Patching ls_config.py...")
    patched = patch_ls_config(solidlsp_path, dry_run=args.dry_run)
    if not patched:
        sys.exit(1)

    if not args.dry_run:
        # Step 3: Verify
        verify_installation(solidlsp_path)

    print("\nDone!")
    if args.dry_run:
        print("\nTo apply changes, run without --dry-run")


if __name__ == "__main__":
    main()
