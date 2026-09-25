# Parallel zipper architecture: historical proposal

Original investigation: January 27, 2026. Status corrected September 25, 2026:
**superseded proposal, not an implementation or integration plan**.

The investigation asked whether independent readers of an immutable PSG could
reduce Alex compilation time. That remains a research question. It did not
establish a safe partition, a merge contract, a working parallel backend or a
measured speedup. The former worker/accumulator pseudocode and integration
instructions are withdrawn.

Current authority is the [Alex architecture](Alex_Architecture_Overview.md),
[backend lowering contract](../../clef-lang-spec/spec/backend-lowering-architecture.md)
and [interactive compiler workbench](Interactive_Compiler_Workbench.md). The
workbench does not depend on implementing this proposal.

## What the investigation contributes

Multiple positional zippers can describe distinct observations of one immutable
graph. This can motivate experiments with independent work; it does not prove
that the observations, their lazy values or the compiler process are isolated.
A Huet zipper records focus and path/context for navigation. Ordinary recursive
folding is not sufficient evidence that a traversal implements that zipper.

The former fan-out/fold-in sketch combined worker outputs into one MLIR module.
A future design must identify what is independent, what remains ordered, and
what provenance survives combining outputs. Associativity permits changing
parenthesization; it does not permit arbitrary reordering. List concatenation
is associative and order-sensitive. A map union also requires an explicit
collision policy; presumed disjoint keys are not a proof of isolation.

## Current integrity boundary

Baker constructs and saturates graph-resident semantics. Alex observes through
ctx pull, codata, coeffects and its positional Huet zipper, then witnesses
admitted MLIR. A parallel worker cannot reconstruct missing semantic facts from
source or invent a second emitter. Effect order, residence, occurrence identity,
target obligations and proof correspondence remain part of admission.

The current implementation has mutable emission bookkeeping outside the zipper,
including accumulators, scopes and visited state. That is not a semantic
construction service. The January record-shaped accumulator merge is not its
current API. Existing registry and target selection state also prevents assuming
that several clients can compile safely in one process at once. The workbench
serializes or isolates compiler workers until an explicit isolation contract is
validated.

## Questions retained for a future experiment

- Can a partition preserve occurrence/path identity, effects, ownership and all
  cross-partition dependencies, including globals and initialization?
- Which outputs have a deterministic merge, and how are conflicting symbols or
  registrations rejected rather than overwritten?
- Can isolated work preserve the same graph, witness and proof boundaries as a
  fresh sequential compilation, including negative admission cases?
- Does the measured workload benefit after discovery, scheduling, merge and
  memory costs are included?

No function-count threshold or speed multiplier is established. The
[workbench measurements](Interactive_Compiler_Workbench.md#proof-responsiveness)
and [incremental contract](Nanopass_Incremental_Contract_Direction.md) govern any
related scheduling or reuse experiment.

## Research references

These are the original investigation's research leads, not proof that Alex can
adopt the same partition or scheduling model:

- Huet, G. (1997), *The Zipper*.
- Ramsey, N. and Dias, J. (2006), *An Applicative Control-Flow Graph Based on
  Huet's Zipper*.
- McBride, C. (2001), *The Derivative of a Regular Type is its Type of One-Hole
  Contexts*.
- Sarkar, Waddell and Dybvig's Nanopass framework papers and the local
  `nanopass-framework-scheme` checkout.
- The local `triton-cpu` checkout and its MLIR pass pipeline; parallel operation
  scheduling there requires its own documented isolation assumptions.
- [Research record](Parallel_Zipper_Research_Summary.md) and
  [revised design record](Parallel_Zipper_Design_Synthesis.md).
