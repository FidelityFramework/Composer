# Composer Regression Test Harness

A standalone F# script-based regression test infrastructure for validating Composer compiler correctness across the FidelityHelloWorld sample suite.

## Quick Start

```bash
# Build once so the runner can load Composer's TOML parser dependencies.
cd /home/hhh/repos/Composer
dotnet build src/Composer.fsproj -c Debug

cd /home/hhh/repos/Composer/tests/regression

# Run full test suite
dotnet fsi Runner.fsx

# Run specific sample(s)
dotnet fsi Runner.fsx -- --sample 01_HelloWorldDirect
dotnet fsi Runner.fsx -- --sample 07_BitsTest --sample 14_Lazy

# Run with verbose output
dotnet fsi Runner.fsx -- --verbose

# Run with custom timeout (seconds)
dotnet fsi Runner.fsx -- --timeout 60
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--sample NAME` | Run specific sample(s). Can be repeated for multiple samples. |
| `--verbose` | Show detailed output including compile errors and diff details. |
| `--timeout SEC` | Override the default timeout for all samples. |
| `--help` | Show help message. |

## Subsetting Runs

Use `--sample` to run a subset of `Manifest.toml` entries.

```bash
# Single exact sample name
dotnet fsi Runner.fsx -- --sample 07_BitsTest

# Multiple explicit samples
dotnet fsi Runner.fsx -- --sample 07_BitsTest --sample 14_Lazy --sample 16_SeqOperations

# Family/substring match (Runner uses name-contains matching)
dotnet fsi Runner.fsx -- --sample HelloWorld
dotnet fsi Runner.fsx -- --sample Recursion
```

Notes:
- `--sample` can be repeated.
- Matching is against the manifest `name` field.
- Matching is substring-based, not strict exact-match only.
- Every supplied filter must match a manifest entry. An unmatched filter fails
  before building the compiler, including when another filter does match.
- Samples run sequentially. `--parallel` is not a supported option.

## How It Works

1. **Single Build**: The harness builds the Composer compiler once at startup (including its selected CCS dependency).
2. **Sample Discovery**: Reads `Manifest.toml` for sample definitions and expected outputs.
3. **Compilation Phase**: Compiles each sample using the built compiler **with `-k` flag**.
4. **Execution Phase**: Runs successfully compiled binaries and compares output.
   - If `stdin_file` is set in `Manifest.toml`, Runner pipes that input to the binary (manifest-driven interactive coverage).
   - A native nonzero exit always fails, even if stdout matches the expectation.
   - Output comparison normalizes CRLF and trims trailing newlines and spaces.
5. **Reporting**: Generates a summary report with pass/fail status.

Process arguments are passed directly. Stdout and stderr drain concurrently;
input delivery, output drainage and process exit share the configured timeout.
On timeout the runner terminates the process tree. The runner's process exit
code reflects the final result.

**Note**: The `-k` flag is always passed to the compiler, so intermediates are always generated. After a test run, you can immediately inspect:
```
samples/console/FidelityHelloWorld/<sample>/targets/intermediates/
```

## Manifest Format

The `Manifest.toml` file defines all samples:

```toml
[config]
samples_root = "../../samples/console/FidelityHelloWorld"
compiler = "../../src/bin/Debug/net10.0/Composer"
default_timeout_seconds = 30

[[samples]]
name = "01_HelloWorldDirect"
project = "HelloWorld.fidproj"
binary = "targets/helloworld"
expected_output = """
Hello, World!
"""

[[samples]]
name = "02_HelloWorldSaturated"
project = "HelloWorld.fidproj"
binary = "targets/helloworld"
stdin_file = "HelloWorld.stdin"    # Optional: provide input
expected_output = """
Enter your name: Hello, Houston!
"""

[[samples]]
name = "16_SeqOperations"
project = "SeqOperations.fidproj"
binary = "targets/SeqOperations"
skip = true                        # Skip this sample
skip_reason = "PRD-16 not yet implemented"
expected_output = ""
```

## Output Format

```
=== Composer Regression Test ===
Run ID: 2026-01-18T15:47:30
Manifest: /home/hhh/repos/Composer/tests/regression/Manifest.toml
Compiler: /home/hhh/repos/Composer/src/bin/Debug/net10.0/Composer

=== Compilation Phase ===
[PASS] 01_HelloWorldDirect (922ms)
[PASS] 02_HelloWorldSaturated (942ms)
[FAIL] 07_BitsTest (725ms)
[SKIP] 16_SeqOperations (-) (PRD-16 not yet implemented)

=== Execution Phase ===
[PASS] 01_HelloWorldDirect (29ms)
[MISMATCH] 05_AddNumbers (28ms)
  First diff at line 3:
    Expected: FloatVal 3.14 -> 3.14
    Actual:   FloatVal 3.14 -> 3.140000
[SKIP] 07_BitsTest (compile failed)

=== Summary ===
Started: 2026-01-18T15:47:30
Completed: 2026-01-18T15:47:44
Duration: 14.3s
Compilation: 13/16 passed, 2 failed, 1 skipped
Execution: 13/14 passed, 0 failed, 1 skipped
Status: FAILED
```

## Exit Codes

- `0`: All tests passed
- `1`: One or more tests failed

## Files

| File | Purpose |
|------|---------|
| `Runner.fsx` | Main test runner script |
| `RunnerCore.fsx` | Process handling, selection, compilation and output comparison |
| `RunnerTests.fsx` | Focused harness tests without compiler builds or sample compilation |
| `Manifest.toml` | Sample definitions and expected outputs |
| `README.md` | This documentation |

## Adding New Samples

1. Add the sample to `Manifest.toml` with:
   - `name`: Directory name under samples_root
   - `project`: The .fidproj file name
   - `binary`: Path to output binary (usually `targets/<name>`)
   - `expected_output`: Expected stdout (use `"""` for multiline)
   - Optional: `stdin_file` for samples needing input
   - Optional: `timeout_seconds` for samples needing more time

2. If the sample needs stdin input, create a `.stdin` file in the sample directory.

## Testing the Harness

After the initial Composer build, run:

```bash
dotnet fsi tests/regression/RunnerTests.fsx
```

These Linux-hosted tests check matching stdout with a failing native exit, launch
failures, concurrent stdout/stderr drainage, timeout and descendant termination,
blocked input, literal argument boundaries, unmatched sample selection, and the
real runner's process exit codes. They create temporary process fixtures and do
not compile the FidelityHelloWorld samples.

## Troubleshooting

**"Manifest not found"**: Run from the `tests/regression/` directory.

**Build failures**: Check that Composer and its selected CCS dependency build and
the compiler path in Manifest.toml is correct. `RunnerCore.fsx` loads the TOML
parser assemblies from the Composer Debug output, so the initial build must
precede loading the runner.

**Output mismatches**: Use `--verbose` to see the full expected vs actual diff.
