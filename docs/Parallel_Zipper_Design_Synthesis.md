# Parallel zipper design synthesis: superseded proposal

Original proposal: January 27, 2026. Status corrected September 25, 2026:
**historical design exploration, not a current implementation plan**.

The proposal considered discovering function roots, grouping them by dependencies,
compiling groups concurrently and combining their outputs. That decomposition was
not validated against the current Alex contracts. The former code recipes,
integration commands, implementation schedule and speedup targets are withdrawn.
The [research record](Parallel_Zipper_Research_Summary.md) retains the prior-art
leads and the distinctions corrected during review.

Current authority is the [Alex architecture](Alex_Architecture_Overview.md),
[backend lowering contract](../../clef-lang-spec/spec/backend-lowering-architecture.md),
[incremental contract](Nanopass_Incremental_Contract_Direction.md) and
[interactive compiler workbench](Interactive_Compiler_Workbench.md).

## Proposal decisions and their limits

| January proposal | Status under the current contract |
|---|---|
| Function-level partitioning | A candidate to investigate, not a proven semantic partition. Scope, initialization, effects, globals and proof dependencies must all be preserved. |
| Derive all dependencies from an SSA coeffect | Insufficient as a general claim. Current names are derived observations; a complete semantic dependency contract cannot be assumed from a presumed allocation pass. |
| Give every worker a fresh ctx and accumulator | Does not isolate compiler-global registration/target state or establish correct occurrence/path and emission scope. |
| Fold worker accumulators into a global result | The sketched record merge is not the current mutable bookkeeping API or an admitted merge contract. |
| Treat associative merge as order-independent | Incorrect. Associativity does not imply commutativity; list append preserves order and map collisions require a policy. |
| Select parallel execution from function counts | Unmeasured heuristic. Useful granularity depends on actual work, dependencies, scheduling, memory and merge costs. |
| Expect a fixed speedup and short completion schedule | No current evidence supports those estimates. |

## Durable constraints

Baker owns elaboration and saturation, including semantic relationships and proof
premises. Alex witnesses the settled graph through ctx pull, codata, coeffects
and its positional Huet zipper. Parallel scheduling cannot create an alternate
semantic construction route, reconstruct missing source behavior late, or replace
an admission failure with a host-language implementation.

The zipper is positional. Current accumulators, scopes and visited state are
physical emission bookkeeping outside it; their existence is not permission to
introduce independent per-worker semantics. Any future partition must preserve
node and occurrence/path identity, selected target, residence, initialization and
effect order, and obligation-to-artifact correspondence.

Independent output fragments need a deterministic combination contract. Keeping
a list's source order may be necessary even when definitions can be referenced
before declaration. Conflicting symbols or registrations must be diagnosed, not
silently resolved by last-writer map updates. A general preservation argument
cannot be replaced by string-equality testing of one example.

## Requirements for a new bounded experiment

1. Select an existing compiler coverage case and record a fresh ordinary
   compilation baseline, target and compiler identities.
2. State which graph facts make each proposed unit independent, including shared
   initialization, storage, effects and proof premises. Keep unknown dependencies
   as an explicit limit on partitioning.
3. Isolate process-global state and define the lifecycle of mutable or lazy
   observations. A shared editor/agent daemon does not imply reentrant compilation.
4. Specify deterministic output combination and validate the same source, graph,
   witness, obligation and native gates, including negative cases.
5. Measure total latency and memory, including discovery, queueing and merge
   overhead. Report the supported workload and limitations; do not infer a
   universal speedup from function count.

This is evidence required to revive the research, not a newly scheduled roadmap.
The warm-host and proof-responsiveness work in WB-01/WB-02 can proceed with
serialized compiler work and isolated solver workers. Native `clefx` and `.clefx`
script execution remain planned consumers of the ordinary
CCS/Baker → PSG → Alex ctx pull → MLIR → LLVM/ORC path, not of this old partition
sketch. FSI may host the F# implementation; it does not execute Clef semantics.

## Related records

- [Architecture investigation](Parallel_Zipper_Architecture.md).
- [Research sources and corrections](Parallel_Zipper_Research_Summary.md).
- [Implementation-claim correction](Parallel_Zipper_Implementation_Summary.md).
