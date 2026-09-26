# Required compiler architecture

These are owner requirements for all work in this repository.

- CCS/Baker settles source semantics in the PSG through elaboration and saturation.
  Preserve nanopass ingredients/recipes, joint constraints, complete premises and
  the intermediate rewrite record.
- Alex is a passive witness of settled PSG facts, using the Huet zipper and
  Elements/Patterns/Witnesses. Do not add semantic analysis, recursive emission,
  post-witness repair, declaration hoisting, or MLIR-to-MLIR transforms to the
  middle end. Emit declarations at their settled scope directly.
- Remove all custom MLIR plugins, their loaders, pass injection, environment
  discovery and compatibility dependencies. Do not restore deleted plugin
  repositories or retain a plugin until its callers are migrated.
- Target-specific MLIR/LLVM transformations belong to the backend. They realize
  settled source contracts; they do not recover missing source semantics.
- A failing gate never authorizes violating these boundaries. Expose the failure,
  repair the owning PSG contract and witness, and retain a discriminating test.
  Do not weaken an oracle to preserve an obsolete implementation.
- Use .NET for host/tooling automation. Do not introduce or run Python drivers.
- Report actual compiler revisions, scope, failures and unrun gates. A compiler
  build or focused pass is not completion of a feature area.

`build/WitnessArchitecture.targets` rejects plugin-loading references and the
retired middle-end rewrite entry points during every Composer compilation.
These mechanical checks supplement architectural review; changing names does
not make a prohibited design acceptable.
