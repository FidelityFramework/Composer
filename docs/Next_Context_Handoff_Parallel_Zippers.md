# Parallel zipper research handoff: superseded

Original handoff: January 27, 2026. Status corrected September 25, 2026:
**historical research intent, not an active continuation task**.

The earlier handoff asked whether Nanopass, Triton/MLIR and Chez Scheme could
inform parallel observation of the PSG through independent zippers. It proposed
studying traversal, dependency boundaries and combining outputs. Those questions
remain useful research leads; the handoff did not establish a parallel Alex
implementation, safe worker partition or measured performance improvement.

Its claims that production compilers prove the proposed zipper architecture,
that every catamorphism is a Huet zipper, and that associativity permits arbitrary
output order are withdrawn. Associativity changes grouping, not ordering. The
former implementation instructions and context-window tasks are superseded.
Do not resume them as an approved integration plan.

The corrected records preserve the research and its limits:

- [Architecture investigation](Parallel_Zipper_Architecture.md): positional
  observation, isolation and dependency questions.
- [Research record](Parallel_Zipper_Research_Summary.md): prior-art leads and
  distinctions between recursive folds, zippers and scheduling.
- [Design synthesis](Parallel_Zipper_Design_Synthesis.md): withdrawn partition
  assumptions and evidence required for a new bounded experiment.
- [Implementation-claim correction](Parallel_Zipper_Implementation_Summary.md):
  current source boundaries and the absent proposed parallel modules.

Current work follows the [Alex architecture](Alex_Architecture_Overview.md) and
[interactive compiler workbench](Interactive_Compiler_Workbench.md). Baker owns
construction and saturation; Alex witnesses compiler-owned graph facts through
ctx pull, codata, coeffects and its positional Huet zipper. Physical emission
bookkeeping is not a second semantic authority. The workbench serializes or
isolates compiler workers and can keep the bootstrap compiler warm without
implementing parallel Alex traversal. Any later concurrency proposal needs its
own provenance, isolation, admission and measured-performance evidence.
