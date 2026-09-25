# Bidirectional Composition: Coordination and Acceptance

**Opened:** 2026-09-21. **Documentation coordination: Complete. Native implementation: Planned.**

Start here to resume the work. This is the single change ledger and acceptance
plan for the bidirectional-computation case study across Composer, Clef, the
language specification, the site, and the research manuscripts. The existing
[NFT architecture](Negative_Fractional_Types_Architecture.md) owns engineering
semantics. This record owns placement, status and next actions, so each other
document can remain focused on its audience.

## Current position

The objective is an ordinary Clef API whose composition retains enough typed
information for Baker to check resource use and select an admissible execution
strategy. Haskell's Tardis supplies a reference example of two-way dependency.
Its name will not become the Clef feature name. A native interpretation must
establish its own semantics and must not identify its two state channels with
negative and fractional types by declaration.

Bidirectional dependency, directed duality, and inverse execution are distinct
properties. A downstream recovery requirement can participate in joint
constraints without adding a runtime history to every result. Bounded-memory
retracing is meaningful for a checked bijection with bounded workspace, or
under an explicitly costed replay contract. A Lyapunov error window alone does
not reconstruct information lost by a finite-precision update.

Chen and Sabry's 2021 erratum concerns Theorem 25's inverse-category claim,
not Tardis and not a retraction of their paper. The missing uniqueness condition
has a counterexample. Compact closure remains available in the stated reference
model. Its translation into Clef still needs a preservation argument.

**Scope of this batch:** source documentation and manuscript corrections, the
new blog article, and making the already-requested *Cold Half of
Concurrency* publicly reachable. No NFT syntax or runtime is implemented by
this batch. All manuscript edits are Markdown only. Generated TeX/PDF review
and publication are separate work, explicitly excluded by the author.

## Authority and placement

| Question | Owning document | Other documents' role |
|---|---|---|
| What changed, what remains, where to resume? | This record | Link here; avoid a second task list |
| How would Clef elaborate and implement it? | [NFT architecture](Negative_Fractional_Types_Architecture.md) | Narrow semantic bridges and local constraints |
| What does the proposed calculus justify? | [NFT manuscript](../../arxiv-papers/negative-fractional-types-in-fidelity.md) | Cite the admitted fragment and its limits |
| How are computations and joint premises represented? | [PHG manuscript](../../arxiv-papers/program-hypergraph-paper.md) and [PHG spec](../../clef-lang-spec/spec/program-hypergraph.md) | Preserve the distinction between graph facts and runtime history |
| What survives saturation and lowering? | [FPS manuscript](../../arxiv-papers/fixed-point-scaffolding.md) and [M-01](PRDs/M-01-DialectAdmission.md) | Name correspondence, runtime obligations and evidence |
| Why should an application developer care? | [A Path Less Traveled](../../clef-lang-site/hugo/content/blog/a-path-less-traveled.md) | Quiet API, bounded-storage example, links to details |
| Which compiler capabilities are accepted? | [PRD index](PRDs/README.md) and [coverage waypoints](Language_Coverage_Waypoints.md) | Documentation completion never promotes feature status |

A separate new internals overview is deferred. Existing PHG, coeffect, and
proof-composition pages provide the necessary entry points. This avoids adding
another semantic authority to maintain.

## Decisions and unresolved choices

| Topic | Position | Status |
|---|---|---|
| Feature name and source style | Native bidirectional composition; ordinary ML domain functions; Tardis as reference only | Agreed direction |
| Graph model | Executable spine, local coeffects, ordered joint relations and evidence in one semantic graph | Agreed framing |
| Lattice metaphor | An ordering of analysis information with separately justified convergence | Agreed correction |
| Dual resources | Carrier, immutable value index, dynamic pairing identity, permitted use and equality evidence | Proposed engineering contract |
| Fractional meaning | Value-indexed core distinct from inverse physical units and rational measure exponents | Agreed correction |
| Source-site accounting | Dynamic instances and all relevant paths must satisfy the selected use policy | Required acceptance condition |
| Quiet API | Downstream uses constrain earlier realization; implementation reports unsupported or over-budget requirements | Proposed interface behavior |
| Memory | Region inverse recipe plus current state/workspace where proved; residual/replay costs otherwise explicit | Agreed accounting |
| Monad interpretation | Preserve bind/demand/effect semantics and laws; a solver model does not prove monad laws | Required acceptance condition |
| Initial fragment | Pure finite bidirectional equations, finite indexed resources, then one exact reversible target example | Planned exercises |
| Surface syntax | No new parser spelling approved here | Open |
| General index equality and dynamic mismatch policy | Admit a bounded equality theory first; retain runtime checks where required | Open beyond first fragment |
| Consumable lazy results and multi-shot captures | No implicit duplication; exact rule and diagnostics need exercise | Open implementation |
| Approximate numeric reconstruction | Requires validated amplification/error bounds and a stated recovery objective | Separate extension |

## Change ledger

**Complete** in this table means the source edit and its stated checks are
complete. It does not mean native implementation, paper publication, or live
publication, which has its own checks below. Dependencies are semantic review order,
not a claim that one document proves another.

| ID | Source and scope | Depends on | Status | Completion evidence |
|---|---|---|---|---|
| D01 | [NFT architecture](Negative_Fractional_Types_Architecture.md): operational core, library/native mapping, identity, memory, target contract | Primary references | Complete | Independent semantic review; total/pure helper and step-count premises explicit |
| D02 | [PRD index](PRDs/README.md), [waypoints](Language_Coverage_Waypoints.md): scoped dependencies and design checkpoint | D01 | Complete | Planned status retained; links/anchors checked |
| D03 | [NFT manuscript](../../arxiv-papers/negative-fractional-types-in-fidelity.md): erratum, case study, ordinary API, exact inverse and limits | D01 | Complete | Source/citation review; prefix/suffix example and 8,192 small-domain inverse identities checked |
| D04 | [PHG manuscript](../../arxiv-papers/program-hypergraph-paper.md): computation/proof incidence, dynamic use, fixed-point and inverse limits | D03 | Complete | Focused Markdown review and baseline comparison |
| D05 | [FPS manuscript](../../arxiv-papers/fixed-point-scaffolding.md): preservation across the graph/target boundary | D03, D04 | Complete | Focused Markdown review and baseline comparison |
| D06 | [Corpus README](../../arxiv-papers/research/corpus/README.md): coordinator and Markdown release boundary | D03–D05 | Complete | Cross-repository coordinator link resolves |
| D07 | [PHG spec](../../clef-lang-spec/spec/program-hypergraph.md), [glossary](../../clef-lang-spec/spec/terms-and-definitions.md): ordered relations, proposed dual core, inference scope | D01, D04 | Complete | Focused review; local links and whitespace checked |
| D08 | [Baker architecture](../../clef/docs/fidelity/Baker_Saturation_Architecture.md), [PHG index](../../clef/docs/fidelity/phg/README.md), [plan](../../clef/docs/fidelity/phg/PSG_to_PHG_Plan.md), [horizons](../../clef/docs/fidelity/phg/Horizon_Requirements.md), [supersession register](../../clef/docs/fidelity/phg/Design_Supersession_Register.md) | D01, D07 | Complete | Dated correction retains historical inventories; links checked |
| D09 | [SemanticGraph Types.fs](../../clef/src/Compiler/PSGSaturation/SemanticGraph/Types.fs): two stale comments only | D08 | Complete | Non-comment source identical to baseline |
| D10 | Site [NFT design](../../clef-lang-site/hugo/content/docs/design/types/negative-fractional-types.md) and [Lyapunov window](../../clef-lang-site/hugo/content/docs/design/types/lyapunov-window.md) | D01, D03 | Complete | Production rendering and local links checked |
| D11 | Site [arithmetic](../../clef-lang-site/hugo/content/docs/internals/numerics/arithmetic-construction-and-placement.md), [proof composition](../../clef-lang-site/hugo/content/docs/internals/verification/proof-composition-and-tooling.md), [Baker](../../clef-lang-site/hugo/content/docs/internals/pipeline/baker-saturation-engine.md), [PHG](../../clef-lang-site/hugo/content/docs/internals/pipeline/hyping-hypergraphs.md), [coeffects](../../clef-lang-site/hugo/content/docs/internals/concepts/coeffects-and-codata.md) | D04, D05, D07 | Complete | Contextual links reviewed; no standalone internals page |
| D12 | [A Path Less Traveled](../../clef-lang-site/hugo/content/blog/a-path-less-traveled.md), subtitle: *How Bidirectional Computation Can Lead to Higher Integrity Computation with a Quieter API* | D01–D11 | Complete | HTTP 200, exact subtitle, contextual links, Atlas and search verified |
| D13 | [The Cold Half of Concurrency](../../clef-lang-site/hugo/content/blog/cold-half-of-concurrency.md): clear draft flag and verify public route | Existing reviewed post | Complete | Draft flag corrected; HTTP 200/title verified after isolated deployment; contextual links included in combined release |
| D14 | Site [Native Reactivity](../../clef-lang-site/hugo/content/blog/native-reactivity-in-clef.md): contextual incoming reference to Cold Half | D13 | Complete | ML-lineage paragraph develops the linked topic; rendered link checked |
| D15 | Site [Pitch, touch and demand](../../clef-lang-site/hugo/content/blog/pitch-slew-and-demand.md): remove remaining historical draft flag; contextual Cold Half link back to the interactive study | D13 | Complete | Local links, shortcode/DOM/assets and both JS syntax checks pass; HTTP 200, Atlas and search verified |
| D16 | Site [README](../../clef-lang-site/README.md) and [blog archetype](../../clef-lang-site/hugo/archetypes/blog.md): record no-draft-blog policy and make new blog entries published by default | Author's publication clarification | Complete | No blog source remains draft; future-date exclusion checked in production build |

## Executable work

These entries remain **Planned** until their executable evidence is recorded.
Each gate may produce findings that revise the proposed contract before a
normative language chapter is admitted.

| ID | Exercise and prerequisites | Acceptance evidence | Status |
|---|---|---|---|
| E01 | Finite pure two-channel reference interpreter; specify observation and demand semantics | Productive example equals the reference bind equations; circular demanded value has defined nonproductive behavior; pure composition laws checked in the fragment | Planned |
| E02 | Finite value-indexed resource core; fixed carrier/equality theory, scoped instance creation and use rules | Match succeeds; wrong value has specified failure; missing/duplicate use rejected or handled by the explicit policy; repeated force and copied captures cannot counterfeit a resource | Planned |
| E03 | Bridge E01/E02 to Baker and ordered PHG occurrences; applicable C-01/C-02/C-05 contracts | Carrier versus index diagnostics; repeated source-site execution creates distinct scoped instances; branch/loop/cancellation obligations verified; quiet source example elaborates without user-threaded proof objects | Planned |
| E04 | One exact modular update region and one native profile; E03 where NFT integration is claimed, M-01 witness gate | Discrete inverse proved; target round-trip oracle; no growing predecessor list; peak live memory measured over increasing step counts with fixed state/direction sizes and bounded outputs | Planned |
| E05 | Numeric extension; validated local-error/amplification bounds and requested recovery tolerance | Reconstruction windows justified for the admitted model/precision; checkpoint and replay costs included; exact bitwise recovery distinguished from physical/model accuracy | Planned |
| E06 | Reactive/effectful extension; relevant R-04 through R-06 and boundary contracts | Invalidation preserves or withdraws dependent evidence; effects obey their order; resource lifetimes survive suspension and cancellation | Planned |

E01 and E02 can develop independently. E04's inverse kernel can also be studied
independently, but that study alone does not accept native NFT integration.
Neither all reactive PRDs nor arbitrary quantum hardware is a prerequisite for
those finite experiments. Compiler tests are introduced with the corresponding
implementation, not as tests that merely restate documentation.

## Validation and publication record

The starting revisions are Clef `e94fa90`, Composer `c464850`, spec `2813371`,
site `52ca6de`, and arXiv-papers `42c58b3`. The first four working trees were
clean. The manuscript repository already contained author edits, including
changes in the three papers revised here. Its existing changes must be retained.
Session snapshots, initial patches and tracked-file hashes are retained under
`/tmp/clef-bidirectional-20260921/`; this temporary location is supporting session
evidence, not the durable coordination record.

Final source review covers **29 files across five repositories**, including this
record. The checks passed for 62 new local documentation links/anchors and 80
rendered site links/anchors. The production build contains all 69 blog entries.
Hugo reports existing configuration/theme deprecations, with no build failure.
Whitespace checks pass in each repository. The small finite reference examples
validate their equations; these checks are not native compiler acceptance.

Baseline hashes confirm that the four manuscript-repository files changed by
this session are Markdown, and that other pre-existing manuscript edits remain
untouched by the session. No `.tex` or `.pdf` was changed or regenerated.
D09's non-comment source is identical to the starting revision. No compiler
regression run was needed for this documentation/comment-only batch.

The author clarified during implementation that **all blog entries belong to the
published set**. The remaining historical flag on *Pitch, touch and demand* and
the new article's initial draft flag were cleared. The README records this
policy and a blog-specific archetype avoids inheriting the generic draft flag.
The new article has a publication time preceding the build, so Hugo's separate
future-date filter cannot hide it.

An initial isolated deployment made *Cold Half* reachable with an incoming link
from *Native Reactivity*. The combined release includes the new article and the
contextually reviewed links. In particular, the chart-readiness discussion in
*Cold Half* links to *Pitch, touch and demand* because its hidden plot and active
audio demonstrate separate consumers. The new article and Cold Half link to
each other through their shared native-composition discussion. The site docs
provide additional relevant incoming links. Public pages, search, and Atlas
have separate acceptance checks; their deployment results are recorded below.

Publication used a captured site working tree and a local module replacement
pointing to a captured current specification, so the corrected PHG and glossary
were included in the same release. During the work, the site sources were
committed as `39374f8` and the specification as `504d8f7`; those concurrent
commits were preserved. The final preprint links and the total/pure-function
clarification are additional site working-tree edits. The public article names
the internal coordinator and links to the published design page, avoiding a
link to an unpublished source file on GitHub.

**Module-pin follow-up:** refresh `clef-lang-site/hugo/go.mod` and `go.sum` to
`504d8f7` after that specification revision is available upstream, before a
subsequent normal deployment. At this check GitHub `main` remained `2813371`,
and the new revision was not available through its API. The configured forge
SSH route was unreachable from this session. The site currently serves the
corrected snapshot; the tracked pin still names the earlier spec revision.
This is a source-publication dependency, separate from native feature work.

| Release check | Evidence |
|---|---|
| Final Pages deployment | `18f1b34a`, [deployment URL](https://18f1b34a.clef-lang.pages.dev/); production build passed |
| New article | [A Path Less Traveled](https://clef-lang.com/blog/a-path-less-traveled/): HTTP 200, exact requested title/subtitle, three preprint links verified |
| Renewed older entries | [Cold Half](https://clef-lang.com/blog/cold-half-of-concurrency/) and [Pitch, touch and demand](https://clef-lang.com/blog/pitch-slew-and-demand/): HTTP 200 and expected titles |
| Publication policy | All 69 blog sources appear in the production build; no blog has `draft: true` |
| Content synchronization | 69 blog posts uploaded; zero failures |
| Search | 2,350 sections across 250 pages synchronized; final pass updated 2 sections, retained 2,348 unchanged, zero failures; all three article queries return their pages |
| Atlas | [Live graph](https://clef-search.engineering-0c5.workers.dev/graph): 268 nodes, 1,939 edges; 1,511 prose links, 100 paper citations, 328 tag links |
| Narrative connectivity | New article: 8 incoming/9 outgoing prose links plus 3 preprint citations. Cold Half: 3 incoming/6 outgoing. Pitch: 1 incoming/3 outgoing. Bidirectional pairs verified from live graph data |

The blog's preprint links identify the published editions explicitly as older
than the working Markdown corrections. They add three real citation edges in
Atlas without implying that TeX/PDF regeneration or arXiv publication occurred.
The linked paragraphs were reviewed for relevance in context, including the
Cold Half/Pitch example of visual demand ending while audio remains active.

The session evidence directory contains `final-source-inventory.json`,
`publication-source-hashes.json`, `rendered-link-check.json`, and the live page,
search and Atlas verification results. The final deployment/index logs are
supporting evidence; this table retains the durable outcome.

## Sources and claim boundaries

- [Chen and Sabry, POPL 2021](https://homes.luddy.indiana.edu/sabry/files/2021/02/popl.pdf), with [erratum](https://homes.luddy.indiana.edu/sabry/files/2021/02/errata.pdf): operational core and corrected categorical scope.
- [Tardis transformer source](https://raw.githubusercontent.com/DanBurton/tardis/master/src/Control/Monad/Trans/Tardis.hs): two-state equations and lazy recursive binding, not an inverse-execution theorem.
- [GHC recursive-do documentation](https://ghc.gitlab.haskell.org/ghc/doc/users_guide/exts/recursive_do.html): recursive monadic binding and its constraints.
- [Stam, exact reversible integration](https://research.nvidia.com/labs/prl/stam2023reversible/reversible2022.pdf): a concrete reference for discrete retracing without a per-step state tape; physical accuracy remains separate.
- [Maclaurin et al., reversible learning](https://proceedings.mlr.press/v37/maclaurin15.pdf): residual information in a particular finite-precision optimization construction.
- [Forward Gradient](https://arxiv.org/abs/2202.08587): directional derivatives as a different technique from inverse execution or dual-resource discharge.

## Resuming the work

First read the current position and the ledger here, then the engineering
architecture's matching section. Check the owning PRD's present status before
implementation. Start with E01/E02 and keep the source fragment explicit.
Record an executable oracle and exact revision when a gate is accepted; a new
paper paragraph is not completion evidence. Update this record and add a dated
coverage waypoint for accepted compiler behavior. Manuscript publication must
follow its own comprehensive review of generated projections.
