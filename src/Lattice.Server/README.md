# Local Lattice server

This .NET 10 stdio server connects the [CCS editor read service](../CCS.Editor/README.md) to standard LSP. Build from the Composer directory so the inherited editor repository's SDK settings do not select the build environment:

```sh
dotnet build src/Lattice.Server/Lattice.Server.fsproj
dotnet src/Lattice.Server/bin/Debug/net10.0/Lattice.Server.dll \
  --project /absolute/path/to/HelloDimensionsProof.fidproj \
  --solver /absolute/path/to/cvc5
```

The second command is for an LSP client, which supplies framed requests on stdin. Compiler tracing goes to stderr. Without `--project`, initialization requires exactly one `.fidproj` in the client's workspace directory. Without `--solver`, dispatch uses `cvc5` on `PATH`.

The [VSCode client walkthrough](https://github.com/FidelityFramework/lattice-vscode/blob/fidelity/client/README.md) prepares a copy of [HelloDimensionsProof](../../samples/lattice/HelloDimensionsProof/README.md) and opens it in an Extension Development Host. There is no published server package to install yet.

## Implemented boundary

The server advertises full-document synchronization, UTF-16 positions, hover, definition and `experimental.clefProofs.version = 1`. CCS loads ordered sources and dependencies, resolves dimensions, constructs obligations and supplies source positions. The adapter publishes its located diagnostics with the effective severity chosen by CCS. Original severities and reachability remain in the editor snapshot; unreachable library diagnostics can appear as information.

Document edits atomically replace their text/version and invalidate the previous check and proof results. The server watches the snapshot's root, platform and dependency manifests and sources, including files outside the workspace. When the watch inventory changes, a settling check runs with the new watchers installed. This is local file observation, not an atomic filesystem snapshot or an incremental compiler. Missing parent directories and remote dependency acquisition need further workspace-service work.

Checks start after a 150 ms pause in input notifications, coalescing rapid edits while invalidation remains immediate. Identical proof queries share one in-flight or completed task within that check generation. Cancelling an individual request stops its wait; changing inputs cancels the shared solver work and clears the cache. Results are not reused across generations.

CCS parser failures currently have file-associated messages but no structured diagnostic ranges. They appear in Lattice output and make semantic/proof requests report a check failure; the adapter does not invent a source position. Completion, references, semantic tokens, multiple projects and native build/debug commands are not advertised.

## Proof view contract, version 1

After the capability is negotiated, `clef/proofs` accepts `{ "textDocument": { "uri": "file:///…", "version": 1 } }`. The document must be open at that version. The response contains that same document identity, a session-local `checkGeneration`, the compiler assembly SHA-256 `compilerIdentity`, and `obligations` associated with the document through their source or premise ranges.

Each obligation carries its compiler `id`, `kind`, `logic`, `statement`, `source`, `refs`, rendered graph `premises`, optional `location`, exact compiler-generated `smtLib`, and SHA-256 `queryHash`. Its `status` contains `phase: "source"`, `state`, `detail`, and the configured solver executable. The server executes cvc5 with a two-second query limit and five-second wall limit, with at most two dispatches in parallel.

| State | Meaning |
| --- | --- |
| `proved` | cvc5 returned `unsat` for the negated obligation under its encoded premises |
| `counterexample` | cvc5 returned `sat`; the encoded premises admit a violation |
| `unknown` | cvc5 could not decide the query or exceeded its execution limit |
| `error` | Dispatch failed, including a missing solver or rejected query |
| `not-dispatched` | Compiler errors prevent dispatch for this snapshot |

The client also accepts `running` for a future progress response; this server currently returns completed batches. `clef/proofsChanged` invalidates displayed evidence when checking starts and requests a refresh when it completes. Superseded requests return LSP `ContentModified` (`-32801`); they cannot update the new snapshot. Client cancellation remains cancellation, not a verdict.

These results are temporary external dispatch evidence linked to compiler-authored graph obligations. They establish source properties under the encoded facts. They do not certify native lowering, emitted JavaScript, runtime inputs outside the premises, or the correctness of the SMT encoding itself. Preservation through lowering remains a separate compiler obligation.

## Focused validation

Run the CCS session and real cvc5 checks from Composer:

```sh
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj
```

From `lattice-vscode/client`, `node test/server-scheduling.cjs` checks shared dispatch, caller cancellation and edit invalidation against the built server with a gated solver stub; its verdicts are scheduling fixtures. Run `npm run test:ccs-host` for actual cvc5 discharge through a real VSCode Extension Development Host. The client README distinguishes this semantic test from the separate protocol-fixture test. The [integration design](../../docs/Lattice_Integration.md) records the broader gates, including compiler-branch reconciliation and the remaining editor surfaces.
