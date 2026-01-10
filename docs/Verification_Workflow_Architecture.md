# Verification Workflow Architecture

Fidelity's value proposition is **compile-time guarantees** about memory safety, type fidelity, and cache behavior. This document specifies the verification workflow that substantiates these guarantees through runtime validation.

## Architectural Position

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Fidelity Compilation Pipeline                        │
│                                                                         │
│  F# Source → FNCS → PSG → Alex → MLIR → LLVM → Native Binary           │
│       │         │      │      │      │      │      │                    │
│       └─────────┴──────┴──────┴──────┴──────┴──────┘                    │
│                         │                                               │
│                    Compile-Time                                         │
│                    Guarantees                                           │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Verification Workflow                                │
│                                                                         │
│  Hardware Counter Data → Analysis → Source Mapping → Editor Display    │
│                                                                         │
│  Tools: perf, VTune, hardware PMU                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

The verification workflow is **post-compilation validation**. It does not replace compile-time guarantees; it confirms them against actual hardware behavior.

## Why Verification Matters

Fidelity makes claims about cache behavior based on:

1. **BAREWire layouts**: Deterministic, platform-aware memory layout
2. **Arena isolation**: Per-actor arenas eliminate cross-actor false sharing
3. **Cache line alignment**: `[<CacheLineAligned>]` attribute padding

These claims are derived from hardware specifications (cache line sizes, coherence protocols). The verification workflow confirms that runtime behavior matches specifications.

### The Trust Hierarchy

```
Hardware PMU (ground truth)
    │
    ▼
perf/VTune (collection)
    │
    ▼
Fidelity Analysis (interpretation)
    │
    ▼
Editor Integration (presentation)
```

Hardware performance counters are the source of truth. Everything else is interpretation and presentation.

## Core Components

### 1. Debug Information Emission

Firefly emits DWARF debug information with cache-relevant annotations:

```
DIE: DW_TAG_variable
  DW_AT_name: "workerState"
  DW_AT_type: <ref to WorkerState>
  DW_AT_location: DW_OP_fbreg -64

  // Fidelity-specific extensions (vendor namespace)
  DW_AT_FIDELITY_cache_line_offset: 0      // Offset within cache line
  DW_AT_FIDELITY_cache_line_crossing: no   // Does not span lines
  DW_AT_FIDELITY_arena_id: "actor_7"       // Arena membership
  DW_AT_FIDELITY_shared: no                // Not shared across arenas
```

This enables mapping hardware events back to source locations with semantic context.

### 2. Hardware Event Collection

The primary data source is hardware performance monitoring:

| Event | Meaning | Tool |
|-------|---------|------|
| `MEM_LOAD_L3_MISS_RETIRED.REMOTE_HITM` | Remote cache hit modified (false sharing) | `perf c2c` |
| `OFFCORE_RESPONSE.DEMAND_DATA_RD.L3_MISS.REMOTE_HIT_FORWARD` | Cross-socket cache traffic | `perf stat` |
| `L1D.REPLACEMENT` | L1 data cache evictions | `perf stat` |
| `LLC_MISSES` | Last-level cache misses | `perf stat` |

### 3. Analysis Engine

The analysis engine correlates hardware events with debug information:

```
Input:
  - Hardware event samples (address, count, event type)
  - DWARF debug information (variables, types, locations)
  - Fidelity annotations (arena membership, alignment)

Process:
  1. Map event addresses to debug symbols
  2. Identify cache line boundaries
  3. Correlate concurrent access patterns
  4. Match against compile-time predictions

Output:
  - Confirmed predictions (cache behavior as expected)
  - Violations (runtime differs from compile-time model)
  - Opportunities (potential optimizations)
```

### 4. Editor Integration Protocol

The analysis results flow to editors via a standardized protocol:

```json
{
  "version": "1.0",
  "session": {
    "binary": "/path/to/executable",
    "profile_data": "/path/to/perf.data",
    "timestamp": "2026-01-10T12:00:00Z"
  },
  "diagnostics": [
    {
      "kind": "false_sharing_confirmed",
      "severity": "info",
      "location": {
        "file": "Worker.fs",
        "line": 42,
        "column": 5
      },
      "message": "Arena isolation preventing false sharing",
      "details": {
        "arena_a": "actor_3",
        "arena_b": "actor_7",
        "cache_line_distance": 4096,
        "hitm_events": 0
      }
    },
    {
      "kind": "false_sharing_detected",
      "severity": "warning",
      "location": {
        "file": "SharedState.fs",
        "line": 18,
        "column": 3
      },
      "message": "Adjacent mutable fields on same cache line",
      "details": {
        "field_a": "counter",
        "field_b": "flags",
        "cache_line_offset_a": 0,
        "cache_line_offset_b": 8,
        "hitm_events": 1247,
        "suggested_fix": "Add [<CacheLineAligned>] or use separate arenas"
      }
    }
  ],
  "summary": {
    "total_hitm_events": 1247,
    "confirmed_isolations": 23,
    "detected_issues": 1
  }
}
```

## Editor Integration Models

### Model 1: Terminal Integration (Baseline)

Every editor supports spawning a terminal. This is the baseline:

```bash
# Collect performance data
perf c2c record -o perf.data ./my_program

# Analyze with Fidelity tooling
fidelity-verify perf.data --format=json > results.json

# View in terminal
fidelity-verify perf.data --format=pretty
```

This works in nvim, vscode, WRENEdit, or any editor with a terminal.

### Model 2: LSP Extensions

LSP supports custom notifications and requests. Fidelity can extend LSP:

```typescript
// Custom notification: verification results available
interface VerificationResultsNotification {
  method: 'fidelity/verificationResults',
  params: {
    uri: DocumentUri,
    diagnostics: FidelityDiagnostic[]
  }
}

// Custom request: run verification
interface RunVerificationRequest {
  method: 'fidelity/runVerification',
  params: {
    binary: string,
    workload?: string
  }
}
```

FSNAC (F# Native Autocomplete) implements these extensions. Editors that support custom LSP methods can use them; others fall back to terminal integration.

### Model 3: DAP Extensions

Debug Adapter Protocol extensions for live profiling during debug sessions:

```typescript
// Custom event: cache contention detected
interface CacheContentionEvent {
  event: 'fidelity/cacheContention',
  body: {
    address: number,
    hitm_count: number,
    source_location: Source
  }
}
```

This enables real-time cache behavior visualization during debugging.

### Model 4: Native Integration (WRENEdit)

WRENEdit's multi-WebView architecture enables richer integration:

```
┌─────────────────────────────────────────────────────────────┐
│                     WRENEdit                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Editor WebView │  │  Profile WebView │                  │
│  │                 │  │                   │                  │
│  │  Source code    │  │  Cache line heat  │                  │
│  │  with inline    │  │  map, timeline,   │                  │
│  │  diagnostics    │  │  structure view   │                  │
│  └────────┬────────┘  └────────┬─────────┘                  │
│           │                    │                             │
│           └────────────────────┘                             │
│                    │                                         │
│              BAREWire IPC                                    │
├──────────────────────────────────────────────────────────────┤
│  F# Native Core                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ Verification Engine                                      ││
│  │ - perf integration                                       ││
│  │ - DWARF parsing                                          ││
│  │ - Analysis algorithms                                    ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

The native backend runs verification; WebViews display results with rich visualization.

## Implementation Priority

### Phase 1: CLI Tool (Foundation)

```bash
fidelity-verify <perf.data> [options]
  --binary=<path>       Path to executable (for symbols)
  --format=<json|pretty|sarif>  Output format
  --threshold=<n>       HITM event threshold for warnings
```

This establishes the analysis engine and output format. Works with any editor via terminal.

### Phase 2: LSP Integration

Add custom methods to FSNAC:
- `fidelity/runVerification`
- `fidelity/verificationResults`
- `fidelity/clearVerification`

Editors supporting custom LSP methods get richer integration.

### Phase 3: Editor-Specific Plugins

- **nvim**: Lua plugin wrapping CLI and displaying diagnostics
- **vscode**: Extension using LSP extensions + custom views
- **WRENEdit**: Native multi-WebView integration

Each editor gets the richest integration its architecture supports.

## Verification Scenarios

### Scenario 1: False Sharing Detection

```
Developer writes:
  type WorkerState = { mutable counter: int64; mutable flags: byte }
  let workers = Array.init 8 (fun _ -> { counter = 0L; flags = 0uy })

Compile-time warning (BAREWire analysis):
  "Adjacent mutable fields may cause false sharing if accessed
   from different threads. Consider [<CacheLineAligned>]."

Runtime verification:
  1. Developer runs workload with perf c2c
  2. Analysis correlates HITM events to WorkerState access
  3. Results confirm or refute the compile-time warning
```

### Scenario 2: Arena Isolation Confirmation

```
Developer writes:
  Actor.spawn (fun () ->
    let localState = Arena.alloc<WorkerState> myArena
    // ... work with localState
  )

Compile-time guarantee:
  "Arena 'myArena' is isolated to this actor.
   No cross-actor false sharing possible."

Runtime verification:
  1. Developer runs workload with perf c2c
  2. Analysis confirms zero HITM events for myArena allocations
  3. Results substantiate the compile-time guarantee
```

### Scenario 3: Cache Line Crossing Detection

```
Developer writes:
  [<Struct>]
  type LargeValue = { a: int64; b: int64; c: int64; d: int64; e: int64 }
  // 40 bytes, may span cache lines depending on alignment

Compile-time analysis:
  "LargeValue may span cache line boundary at runtime.
   Consider [<CacheLineAligned>] for hot paths."

Runtime verification:
  1. Developer profiles hot code path
  2. Analysis measures L1D.REPLACEMENT events
  3. Results quantify actual cache line crossing impact
```

## Integration with Existing Tools

### perf (Linux)

Primary integration point for Linux targets:

```bash
# Record cache-to-cache events
perf c2c record -o perf.data ./program

# Record with call graphs for attribution
perf record -g -e cache-misses,LLC-load-misses ./program

# Record specific PMU events
perf stat -e L1-dcache-load-misses,LLC-load-misses ./program
```

Fidelity tooling wraps perf and interprets results.

### VTune (Intel)

For deeper Intel-specific analysis:

```bash
# Memory access analysis
vtune -collect memory-access ./program

# Microarchitecture exploration
vtune -collect uarch-exploration ./program
```

Fidelity can import VTune results when available.

### Platform Independence

The verification workflow abstracts over platform-specific tools:

```
┌─────────────────────────────────────────────────────────────┐
│                 Fidelity Verification API                    │
├─────────────────────────────────────────────────────────────┤
│  Platform Adapters                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Linux/perf  │  │ Intel/VTune │  │ macOS/Instr │  ...    │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

The same analysis concepts apply; data collection differs by platform.

## Output Formats

### JSON (Machine-Readable)

For editor integration and automation:

```json
{
  "diagnostics": [...],
  "summary": {...}
}
```

### SARIF (Static Analysis Results)

For CI/CD integration and standardized tooling:

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "fidelity-verify",
          "version": "1.0.0"
        }
      },
      "results": [...]
    }
  ]
}
```

### Pretty (Human-Readable)

For terminal use:

```
Fidelity Verification Report
============================

Binary: ./my_program
Profile: perf.data (2.3 MB, 1.2M samples)

False Sharing Analysis
----------------------
[OK] Arena 'actor_main' isolated (0 HITM events)
[OK] Arena 'worker_pool' isolated (0 HITM events)
[!!] SharedState.fs:18 - HIGH contention (1,247 HITM events)
     Fields 'counter' and 'flags' on same cache line
     Suggested: Add [<CacheLineAligned>] attribute

Cache Behavior Summary
----------------------
L1D hit rate:  98.2%
LLC hit rate:  94.1%
HITM events:   1,247 (concentrated at SharedState.fs:18)

Compile-time predictions: 23 confirmed, 1 violation
```

## Security Considerations

Performance counter access requires privileges. The verification workflow:

1. **Does not require root for analysis** - Operates on collected data files
2. **Collection requires appropriate permissions** - `perf` needs `perf_event_paranoid` settings
3. **No sensitive data in output** - Only addresses, counts, and source locations

## Future: Continuous Integration

Verification can be part of CI/CD:

```yaml
# .github/workflows/verify.yml
- name: Run performance tests
  run: |
    perf c2c record -o perf.data ./benchmarks
    fidelity-verify perf.data --format=sarif > results.sarif

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: results.sarif
```

This catches cache regressions before merge.

## Related Documentation

| Document | Location |
|----------|----------|
| BAREWire Cache-Aware Layouts | `~/repos/BAREWire/docs/09 Cache-Aware Layouts.md` |
| fsnative Atomic Operations | `~/repos/fsnative-spec/spec/atomic-operations.md` |
| WRENEdit Tooling Integration | `~/repos/WRENEdit/docs/08_tooling_integration.md` |
| FSNAC LSP Extensions | `~/repos/FsNativeAutoComplete/docs/LSP_Extensions.md` |
