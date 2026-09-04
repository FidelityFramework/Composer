# Witness Boundary Audit — Composer/Alex

> Audit of record, 2026-09-03. Measures the actual state of the Alex witness
> boundary against the doctrine that governs it. **Resolved since (2026-09-04):**
> `TypeSizing.fs` deleted (§4a); `cf`/`vector` dead vocabulary deleted (§5);
> `ProofObligations.fs` out of the build — obligations are graph citizens read
> from $F$ (`SMTTransfer` rewired); `pSysReadline`'s `1024L` and unconditional
> trim now read the site's declared-buffer annotation (§4c). The rest stands as
> measured; see `clef/docs/fidelity/phg/`. Companion to
> [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md),
> [Single_Flattening_Design.md](./Single_Flattening_Design.md), and
> `clef/docs/fidelity/phg/PSG_to_PHG_Plan.md`, whose *Draining Alex* inventory
> is derived from Section 4 below.

The rule under test, stated most sharply at `docs/CCS_Architecture.md:204-205`:

> The Zipper traversal in Alex is **purely navigational** — it observes
> pre-computed coeffects and emits the corresponding MLIR. **It does not
> compute, infer, or decide.**

## 1. What a witness is handed

`src/MiddleEnd/Alex/Traversal/TransferTypes.fs`

`TransferCoeffects` (43–66) — "Pre-computed coeffects - computed ONCE before
traversal, NEVER modified" (42). Fourteen fields:

| Field | Line | Source module |
|---|---|---|
| `SSA` | 44 | `PSGElaboration.SSAAssignment` |
| `Platform` | 45 | `PlatformConfig.PlatformResolutionResult` |
| `Mutability` | 46 | `MutabilityAnalysis` |
| `PatternBindings` | 47 | `PatternBindingAnalysis` |
| `Strings` | 48 | `StringCollection` |
| `YieldStates` | 49 | `YieldStateIndices` |
| `EscapeAnalysis` | 50 | `EscapeAnalysis` |
| `CurryFlattening` | 51 | `CurryFlattening` |
| `DeclarationRootLambdas` | 52 | PSG `DeclRoot` map |
| `TargetPlatform` | 55 | `Core.Types.Dialects` |
| `PinMapping` | 58 | `Coeffects.PlatformPinMapping` (FPGA) |
| `WidthInference` | 61 | `IntervalAnalysis` (FPGA) |
| `ValuePosition` | 65 | `ValuePositionAnalysis` |

**None of these come from CCS.** All are computed in
`src/MiddleEnd/PSGElaboration/*.fs` — Composer computes its own coeffects over
the Clef PSG. What Alex consumes *from Clef* is `SemanticGraph` +
`SemanticNode` and nothing else. This contradicts `CCS_Architecture.md:191-198`,
whose table claims CCS computes SSA pre-assignment, capture analysis, lifetime
and emission strategy.

`WitnessContext` (498–508) hands a witness the whole graph plus five pieces of
mutable state (a reference-type accumulator with `get, set`, two scope refs, two
visited refs). That is the structural enabler for everything in Section 4: a
witness that can walk `ctx.Graph` freely and mutate shared state is not
constrained to observation.

## 2. The hard-stop rule, and the vocabulary that was never enumerated

`Thin_Middle_End_Design.md:22`:

> **no llvm dialect, and no semantic dialect, in the MLIR witnessed out of the
> PSG.** The standard dialects, `func`, `memref`, `arith`, `scf`, `index`, are
> the lingua franca of the witnessed region.

`:48`:

> **The count is zero.** … the witness layer emits from a fixed vocabulary, and
> additions to that vocabulary are additions to this document first.

**Finding.** Section 5 asserts a fixed vocabulary and never enumerates it. The
only enumeration is line 22's five dialects, which omits `cf` and omits
everything Alex actually emits (Section 5). The rule is therefore unenforceable
as written, and has not held.

## 3. What Single_Flattening permits

`Single_Flattening_Design.md:42-46` (its own caveat at `:10` — "not normative"):

> **Alex is more constrained than earlier envisioned: it witnesses and elides
> what the graph has settled, and it performs no semantic transformation.**
> Baker gains significant structure. The recipes carry the semantic inventory:
> closures, suspension, dual pairs, and nets.
>
> … **structure discharges in the graph before the zipper runs, and the zipper
> carries settled residue only.**

And the coeffect contract itself, `PSGElaboration/Coeffects.fs:47-49`:

> Witnesses observe this coeffect; they do NOT compute layout during emission.

## 4. Where Alex computes rather than observes

### 4a. TypeSizing.fs — computes sizes by string-parsing, and is dead

`Alex/CodeGeneration/TypeSizing.fs`. `:23` `wordSize = 8L` hardcoded; `:27-61`
`computeSize` parses `"i32"`, `"tuple<...>"`, `"memref<...>"` out of rendered
type *text*; `:56-57` `memref<` → `wordSize * 4L` with its own comment calling
it "approximate" and "backend-dependent"; `:64-108` a hand-rolled
angle-bracket-balancing parser. Self-described at `:7-9` as temporary pending
BAREWire.

**Compiled (`src/Composer.fsproj:82`) with zero callers.** Dead code carrying a
third, inconsistent size model.

### 4b. Three inconsistent size models in one compilation

| Function | File:line | `TIndex`/ptr | `memref` | Arch-aware |
|---|---|---|---|---|
| `mlirTypeSize` | `Dialects/Core/Types.fs:59-71` | 8 | **40** (5 words) | no |
| `mlirTypeSizeForArch` | `CodeGeneration/TypeMapping.fs:55-71` | `wordBytes` | 5 × word | yes |
| `computeSize` | `CodeGeneration/TypeSizing.fs:27` | 8 | **32** (4 words) | no |

`RecordPatterns.fs:29` and `OptionWitness.fs:38` use the arch-independent one;
`ClosurePatterns.fs:247-248` and `LambdaWitness.fs:318,547` use the arch-aware
one. **Record layouts and closure layouts compute from different size models in
the same compilation.**

### 4c. Hardcoded resource and ABI decisions

- `Patterns/PlatformPatterns.fs:304-305, 313` — the `Sys.readline` buffer is
  `1024L`, authored in the witness, read from no coeffect.
- `:326-327` — `SubI(trimmedLen, bytesReadIdx, oneConst)`: Alex decides
  unconditionally that the last byte is a newline and trims it. An I/O framing
  semantic decided below the graph.
- `:422-430`, duplicated `:693-694` — **SysV x86_64 ABI classification in the
  middle end** (`structs > 16 bytes are MEMORY class`), `AlignBytes = 8`
  hardcoded, `size` from the arch-independent sizer, emitted as the
  non-standard attribute `ffi.byval` (`Serialize.fs:528-529`).
- `:110-111` — `Syscall`/`InlineAsm` coeffect values silently discarded for a
  fallback name.

### 4d. Layout computation — the largest violation class

`CaptureSlot` (`Coeffects.fs:57-68`) carries no byte offset, so every offset is
computed in the witness/pattern layer:

- `LambdaWitness.fs:317-327` — closure prefix offset with `// Approximate`
  **twice**, on load-bearing values.
- `LambdaWitness.fs:546-548, 593, 632` — a running mutable `captureByteOffset`.
- `ClosurePatterns.fs:226, 246-248, 267-268` — the same offset walk implemented
  a **second** time. Construction and extraction compute layout independently;
  nothing guarantees they agree.
- `RecordPatterns.fs:28-29` — record field offsets, arch-independent sizer.
- `OptionWitness.fs:38` — option representation (1-byte tag + payload).
- `TypeMapping.fs:422-470` — Alex owns physical layout of tuples, anonymous
  records, `TLazy`, `TSeq`, `TSeqEnumerator`, `TUnion`.
- `TypeMapping.fs:368-396` — **DU layout synthesized in Alex, and wrongly**:
  `payloadType = casePayloadTypes |> List.choose id |> List.tryHead` — *first*
  case, not largest. `:387` admits the cause ("would need layout info").
  `:244` in the same file uses `max` for `Result`. The two paths disagree, and
  any union whose first payload case is not its biggest is under-sized.
- `TypeMapping.fs:260-267` — outright estimation: `FieldCount * wordSize`.
- `TypeMapping.fs:413-414` — `mapNativeType = mapNativeTypeForArch X86_64`, a
  default target commitment, called at `:775, 788, 790`.

**The ignored channel.** `SemanticNode.LayoutHint`, `ArenaAffinity` and
`SRTPResolution` have **zero references in Alex** (the only `SRTPResolution` hit
in Composer is `FrontEnd/CCS/Integration.fs:54`). The three channels the PSG
carries for this purpose are unread.

**The doctrinal split.** `clef/.../NativeTypes.fs:655-656` says *"CCS preserves
type identity; **Alex resolves to concrete size**"*, while
`CCS_Architecture.md:205` says Alex does not compute or decide. Not reconcilable
as written; the PHG plan resolves it in the graph's favour.

### 4e. Branching on self-derived facts

- `VarRefWitness.fs:68-73` — function-ness derived by
  `childNode.Kind.ToString().StartsWith("Lambda")`. **Eleven lines later**,
  `:82-84` does it correctly off a coeffect and quotes the rule while doing so.
  Both postures in one function.
- Same stringify idiom: `BindingWitness.fs:41`, `LambdaWitness.fs:172, 464`,
  `MatchWitness.fs:85`.
- `KernelModuleWitness.fs:223-226` — float-vs-int selection by inspecting the
  *rendered MLIR type string* (`s.StartsWith("f")`).
- `TypeMapping.fs:286-289` — name matching, explicitly forbidden by
  `CCS_Architecture.md:225`.
- `ClosurePatterns.fs:94` — linkage decided by `if name = "main"`.
- `EscapeAnalysis.fs:291-293` — `getEscapeKindOrDefault` defaults to
  `StackScoped`; when the coeffect is absent the witness picks an allocation
  strategy itself (`LambdaWitness.fs:508`, `MemoryPatterns.fs:468`).

### 4f. Witness-minted SSAs

- `LambdaWitness.fs:615-617` — `V (10000 + tempIdx)`, "high range to avoid
  collision with pre-computed SSAs".
- `ClosurePatterns.fs:103-105` — same trick.
- `TransferTypes.fs:277` — `MLIRTempCounter` is a first-class accumulator
  member, i.e. this is sanctioned.
- `HardwareModulePatterns.fs:110-111, 117, 297` — the whole `hw.module` body
  uses its own counter.
- `LambdaWitness.fs:295-307` — the witness snapshots the accumulator because it
  distrusts the coeffect's SSAs ("may not match emission").

### 4g. Structure synthesized with no PSG counterpart

- `KernelModuleWitness.fs:262-371` — a ~110-line **string template** emitting a
  complete `aie.device(npu2)` module: device, tile rows, objectfifo depth, loop
  bound `4294967295`, burst length, DMA stride descriptors. No coeffect behind
  any of it. `:99` also *validates* shape — a judgment.
- `HardwareModulePatterns.fs:14-28, 112-232, 282-520` — Alex synthesizes a Mealy
  machine (POR register, `comb.xor` first-cycle reset, per-field `seq.compreg`
  with feedback, struct wiring, flat pin ports). The PSG has a `Design<'S,'R>`
  record. That is lowering, not witnessing.
- `ClosurePatterns.fs:140-177` — an entire `func.func @<name>_as_closure` thunk
  that does not exist in the graph.
- `LambdaWitness.fs:184-188` — entry point name, signature and linkage decided
  in the witness; `"main"` + `argv: memref<?xi8>` is a C-runtime commitment.
- `MemoryPatterns.fs:319-334` — `pArenaAlloc` returns `[]` ops and the arena
  memref itself, silently substituting different allocation semantics.

None of this carries `Elaboration.Kind` metadata, so it is invisible to Lattice
and unreachable by the "pierce the veil" debugging
`PSG_Enrichment_Architecture.md:91-131` describes — breaking that document's
"Transparency Over Magic" principle (`:222-224`).

## 5. Dialects actually emitted

**Permitted** (`Thin_Middle_End_Design.md:22`): `arith` (26 ops), `memref` (15),
`func` (6), `scf` (5), `index` (4).

**Beyond the documented bound:**

- `cf` and `vector` — union cases exist, never serialized; fall through to
  `sprintf "// TODO: Serialize %A"` (`Serialize.fs:637-639`). Dead vocabulary.
- `builtin.unrealized_conversion_cast` — 6 sites (`Serialize.fs:499, 501, 504,
  506, 549, 554`). `Thin_Middle_End_Design.md:32` claims this "resolved into the
  named materialize and scatter pair governed by a round-trip law." **It did
  not** — the pairs still serialize to the anonymous cast with the meaning in a
  comment.
- CIRCT `hw` / `comb` / `seq` — emitted **at** the witness; `:41` places them
  below the boundary.
- `smt` — proof obligations rendered as IR in the middle end.
- `aie` / `aiex` — **raw unparsed text** via `MLIROp.RawMLIR`, bypassing the
  typed op representation entirely.
- an `ffi.` private namespace fence downstream plugins must strip.

**`llvm.*`: not emitted. The first ban holds.**

**On "flat generic ops"** — neither half holds today. The op stream is *nested*
(`FuncDef` carries a body list; `TransferTypes.fs:423` needs a recursive
`countOperations`), and the serializer emits *custom* assembly form, not generic
form. The "Flat accumulator" comment at `TransferTypes.fs:267` is stale,
contradicted at `:411`.

## 6. The division of labour, as documented versus as built

`PSG_Nanopass_Architecture.md:3-6` states CCS builds the PSG and "Composer
consumes the PSG as 'correct by construction'". `:372-384` concludes Alex's job
"simplifies" because SRTP is resolved and types attached, and that the scribe is
"a **pure transcription layer** … **No resolution logic**".

Against the code: `SRTPResolution` is never read; `TypeMapping.fs` is 857 lines
of re-derivation including estimation and DU synthesis — the re-inference `:376`
says is unnecessary. There is no `PSGScribe`; the transcription layer is the
12,081-line Alex.

**The mechanical explanation is in the same document.** `:425-536` describes the
PSG→PHG evolution and hyperedge promotion and places it at "Mid-term". The
structure `Single_Flattening_Design.md:46` relies on — "structure discharges in
the graph before the zipper runs" — is roadmap, not present. **Layout sits in
Alex because the graph does not yet carry it.**

## 7. Verdict

**Held.** The `llvm` ban. SSA-as-coeffect on the common path. `ValuePosition`
(`VarRefWitness.fs:82-84`), `CurryFlattening` (`ApplicationWitness.fs:63`,
`BindingWitness.fs:53`), `WidthInference` (`PSGCombinators.fs:110-155`, which
**fails loudly at `:121-125` rather than guessing**), `PinMapping`
(`HardwareModuleWitness.fs:365`), and target-gated registration
(`WitnessRegistry.fs:84-124`) are genuine coeffect reads.

**`Traversal/XDCTransfer.fs` is the model citizen:** 72 lines, pure
`PlatformPinMapping → string`, header stating the posture exactly — *"The
coeffect IS the pre-computed data. This transfer is serialization."* Every
transfer should look like this one.

**Violated.** Memory layout in its entirety (4d); ABI classification and
resource sizing (4c); entry-point and linkage conventions (4g); structure
synthesis for NPU and FPGA (4g); witness-side SSA minting (4f); name- and
string-based dispatch throughout (4e). The dialect vocabulary exceeds its
documented bound (5).

## 8. Standing conflict to resolve

`src/MiddleEnd/Alex/Pipeline/MLIRNanopass.fs:1-18` is an MLIR→MLIR pass
**already running post-witness**, and its roadmap states the intent to add
*"DCont lowering (sequential/effectful patterns → stack-based async)"* and
*"Inet lowering (parallel/pure patterns → graph reduction)"* **there**.

That collides with `Thin_Middle_End_Design.md:22` and `:30`, and with
`Single_Flattening_Design.md:38` and `:52` ("The count is one"). Two live
documents plan opposite futures for the same file. The PHG plan takes the graph
side; this roadmap should be retired in writing before that work begins.
