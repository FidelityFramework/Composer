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

# Run at most four independent compiler/native jobs per phase
dotnet fsi Runner.fsx -- --jobs 4

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
| `--jobs N` | Maximum independent compiler/native jobs per phase; positive integer, default `1`. |
| `--prune-intermediates` | Retain reachable PSG nodes and the complete participant closure of joint evidence. The default retains full dumps. Recorded in `run.json`. |
| `--results DIR` | Parent directory for a unique run directory and retained artifacts. Defaults to `/tmp/composer-checks` on Linux. |
| `--manifest FILE` | Read a different sample manifest. |
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
- `--jobs` defaults to `1`; increase it for independent sample checks within the
  machine's CPU and memory budget.

## Choosing Check Scope

Use a targeted sample selection for each implementation step, alongside the
source/graph, proof, editor or Alex checks appropriate to the changed semantics.
Add neighboring controls when shared compiler contracts can affect them. Run the
full relevant suites at major integration gates and before whole C-xx closure;
do not rerun every sample after every edit. The
[regression check policy](../../docs/Regression_Check_Policy.md) gives concrete
commands, expansion triggers and evidence requirements. A narrowed passing run
establishes only its selected scope; skipped or failing required behavior remains
open. Keep the selected expectations and retain the runner's artifact directory.

## How It Works

1. **Selection**: Reads the manifest and validates every requested filter before building.
2. **Private Host, Compiler Build and Snapshot**: Builds the .NET process host into this run's private directory, using `--artifacts-path` for isolated build intermediates. Then builds Composer once, including its selected CCS dependency, copies the compiler output into this run's private directory and records its hashes.
3. **Compilation Phase**: Compiles samples in independent CLI jobs, at most `--jobs N` at a time, with full `-k` artifacts retained by default. `--prune-intermediates` selects the [pruned diagnostic view](../../docs/Regression_Check_Policy.md#pruned-diagnostic-artifacts) while preserving compilation semantics and complete joint evidence.
4. **Execution Phase**: After every compilation finishes, runs successfully compiled binaries with the same worker bound and compares output.
   - If `stdin_file` is set in `Manifest.toml`, Runner pipes that input to the binary (manifest-driven interactive coverage).
   - A native nonzero exit always fails, even if stdout matches the expectation.
   - Output comparison normalizes CRLF and trims trailing newlines and spaces.
5. **Reporting**: Generates a summary report with pass/fail status.

Each job uses the .NET `ProcessHost`; arguments are passed directly without a
shell. The bound counts jobs, not operating-system processes: each host launches
its command, and compiler tools can launch children. Stdout and stderr drain
concurrently; input delivery, output drainage and process exit share the
configured timeout. Jobs without supplied input receive EOF. The host preserves
the command's exit code.

On Linux, the host creates a session with `setsid`, and the runner terminates its
process group through .NET P/Invoke. This allows timeout cleanup even after a
parent exits while a descendant still holds the output pipes open. Other
platforms use .NET process-tree termination. The runner's process exit code
reflects the final result. Results retain manifest selection order even when
jobs finish out of order. This is process-level check concurrency; it does not
parallelize graph construction, Baker settlement or Alex witnessing inside one
compiler process.

Runner invocations coordinate their shared build/copy step with a cooperative
lease. Continue coordinating manual builds and other suites that rebuild the
same Composer or CCS output. Compilation workers use the private snapshot after
that step completes.

The runner prints its unique artifact root. `--results` changes the parent, not
the unique run identity. A location inside the compiler output is rejected,
including through directory links, to prevent recursive snapshot copying. Each
selected ordinal receives its own directory, even when two entries compile the
same project or use the same binary name:

```text
/tmp/composer-checks/<timestamp-guid>/
  Manifest.toml
  selection.txt
  run.json                  # resolved inputs, effective limits and expectations
  results.json              # ordered outcomes
  process-host/             # private .NET process host
  host-build/               # isolated host build intermediates
  host-build.stdout.log
  host-build.stderr.log
  host-build.status
  compiler/                 # private compiler output snapshot
  compiler.sha256
  build.stdout.log
  build.stderr.log
  build.status
  0001/
    sample.txt
    compile.stdout.log
    compile.stderr.log
    compile.status
    run.stdout.log           # when native execution is attempted
    run.stderr.log
    run.status
    <binary>
    intermediates/          # selected PSG view, recipe and lowering artifacts
```

Native execution retains the sample directory as its working directory for
manifest-driven input and existing relative resources. Isolated binaries and
compiler artifacts do not isolate arbitrary application writes to shared files.

For direct CLI checks of the same project, give each invocation a distinct
output path and `--artifacts-dir`:

```bash
src/bin/Debug/net10.0/Composer compile \
  samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fidproj \
  -o /tmp/composer-direct-check-1/helloworld -k \
  --artifacts-dir /tmp/composer-direct-check-1
```

The CLI isolation option does not change the default output layout for direct
commands that omit it. Serialized graph views and future in-process incremental
or parallel scheduling remain separate compiler work.

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
| `ParallelRunnerTests.fsx` | .NET-only worker-bound, phase-barrier, process and artifact-isolation tests |
| `../Infrastructure/ProcessHost/ProcessHost.fsproj` | .NET command host; establishes a Linux session for job cleanup |
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

After the initial Composer build, build the host before loading the standalone
harness tests. The runner itself builds a private host automatically:

```bash
dotnet build tests/Infrastructure/ProcessHost/ProcessHost.fsproj
dotnet fsi tests/regression/RunnerTests.fsx
dotnet fsi tests/regression/ParallelRunnerTests.fsx
```

These Linux-hosted tests check matching stdout with a failing native exit, launch
failures, concurrent stdout/stderr drainage, timeout and descendant termination,
blocked input, literal argument boundaries, unmatched sample selection, and the
real runner's process exit codes. They create temporary process fixtures and do
not compile the FidelityHelloWorld samples.

`ParallelRunnerTests.fsx` uses F#/.NET process fixtures to exercise worker limits,
overlap, ordered results, the compile-to-native barrier, independent artifact
paths, stream drainage and timeout isolation. These harness checks do not claim
that any language sample or concurrent native cohort has passed.

## Troubleshooting

**"Manifest not found"**: Run from the `tests/regression/` directory.

**Build failures**: Check that Composer and its selected CCS dependency build and
the compiler path in Manifest.toml is correct. `RunnerCore.fsx` loads the TOML
parser assemblies from the Composer Debug output, so the initial build must
precede loading the runner.

**Output mismatches**: Use `--verbose` to see the full expected vs actual diff.
