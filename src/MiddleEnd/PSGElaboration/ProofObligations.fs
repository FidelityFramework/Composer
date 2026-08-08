/// ProofObligations: proof obligations as a COEFFECT
///
/// ARCHITECTURAL PRINCIPLE (Codata/Coeffect Model):
///
/// Obligations are contextual facts ABOUT the program, computed by observing
/// the saturated PSG. The PSG is unchanged. Like every coeffect they are
/// PRESENT downstream, never queried or recomputed:
///
///   - Design-time form: serialized here as a ledger (JSON) and a solver-ready
///     SMT-LIB artifact, dispatched by an external solver (cvc5).
///   - Build-time form: Alex's SMTTransfer transcribes the SAME records into
///     `smt` dialect IR. One birth, two dispatches: obligation identity
///     (the anchor name) is preserved across both.
///
/// Refutation style throughout: each obligation is a named Boolean anchor;
/// its definition and its NEGATION are asserted. `unsat` = obligation HOLDS.
///
/// Scope (initial): string-literal storage discipline and data-layout facts
/// for literals written in the entry compilation unit: the strings the
/// developer typed, carrying their exact source positions.
module PSGElaboration.ProofObligations

open System.IO
open System.Text.Json
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// OBLIGATION TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// What an obligation asserts. All facts are PRE-COMPUTED here;
/// downstream transfers only transcribe, never compute.
type ObligationBody =
    /// storage = len + 1 (the NUL byte is reserved at allocation)
    | StorageReservation of len: int * storage: int
    /// view = len AND view < storage (the terminator is never written)
    | ViewContainment of view: int * len: int * storage: int
    /// final storage byte is 0x00
    | NulSentinel of lastByte: int
    /// consecutive layout of the given storages is pairwise disjoint
    /// and spans exactly `span` bytes
    | ConsecutiveLayout of storages: int list * span: int

/// A proof obligation recorded in the PSG
type Obligation = {
    /// Stable anchor name: the identity that travels through both dispatches
    Id: string
    Kind: string
    /// SMT-LIB logic fragment: QF_LIA or QF_BV
    Logic: string
    /// Human-readable statement (ledger and demo surface)
    Statement: string
    /// Origin: file:line:col, from the PSG node range
    Source: string
    /// External rule cross-references (CWE ids), the auditor-facing vocabulary
    Refs: string list
    Body: ObligationBody
}

/// The obligation coeffect
type ObligationSet = { Obligations: Obligation list }

/// The empty coeffect (obligation analysis not requested)
let empty : ObligationSet = { Obligations = [] }

// ═══════════════════════════════════════════════════════════════════════════
// ANALYSIS (pure observation of the PSG)
// ═══════════════════════════════════════════════════════════════════════════

/// Derive a stable, human-readable slug from string content
let private slug (content: string) : string =
    match content with
    | "" -> "empty"
    | "\n" -> "newline"
    | ", " -> "comma_space"
    | "!" -> "bang"
    | c ->
        let words =
            System.Text.RegularExpressions.Regex.Matches(c.ToLowerInvariant(), "[a-z0-9]+")
            |> Seq.map (fun m -> m.Value)
            |> Seq.truncate 3
            |> Seq.toList
        if List.isEmpty words then
            sprintf "str_%08x" (StringCollection.deriveByteLength c)
        else
            String.concat "_" words

let private describe (content: string) : string =
    match content with
    | "" -> "the empty string"
    | c -> sprintf "\"%s\"" (c.Replace("\n", "\\n").Replace("\r", "\\r"))

let private fmtRange (r: SourceRange) : string =
    sprintf "%s:%d:%d" r.File r.Start.Line r.Start.Column

/// Compute the obligation coeffect from the saturated PSG.
/// Observes reachable string literals in the entry compilation unit;
/// the emission invariants (NUL reservation, view trimming) come from
/// the same contract LiteralPatterns emits under.
let analyze (graph: SemanticGraph) : ObligationSet =
    match graph.DeclarationRoots with
    | [] -> { Obligations = [] }
    | (entryId, _) :: _ ->

    let entryFile =
        match Map.tryFind entryId graph.Nodes with
        | Some n -> n.Range.File
        | None -> ""

    if entryFile = "" then { Obligations = [] } else

    // The strings the developer wrote, in source order, deduplicated by content
    let userLiterals =
        graph.Nodes.Values
        |> Seq.filter (fun n -> n.IsReachable && n.Range.File = entryFile)
        |> Seq.choose (fun n ->
            match n.Kind with
            | SemanticKind.Literal (NativeLiteral.String s) -> Some (s, n.Range)
            | _ -> None)
        |> Seq.sortBy (fun (_, r) -> r.Start.Line, r.Start.Column)
        |> Seq.distinctBy fst
        |> Seq.toList

    // Slugs are display names, not identities: distinct contents can collide
    // ("hello world" vs "Hello, World"). The anchor name is the identity that
    // travels through both dispatches, so it MUST be unique: uniquify in
    // deterministic source order.
    let named =
        let seen = System.Collections.Generic.Dictionary<string, int>()
        [ for (content, range) in userLiterals ->
            let s = slug content
            let n = match seen.TryGetValue s with | true, c -> c + 1 | _ -> 1
            seen[s] <- n
            (content, range, (if n = 1 then s else sprintf "%s_%d" s n)) ]

    let perString =
        [ for (content, range, name) in named do
            let len = StringCollection.deriveByteLength content
            let storage = len + 1   // NUL reservation invariant (LiteralPatterns contract)
            let born = fmtRange range
            yield { Id = sprintf "storage_%s" name
                    Kind = "storage-reservation"
                    Logic = "QF_LIA"
                    Statement = sprintf "storage for %s is exactly its logical length %d plus one terminator byte (%d = %d + 1)" (describe content) len storage len
                    Source = born
                    Refs = ["CWE-131"]
                    Body = StorageReservation (len, storage) }
            yield { Id = sprintf "view_%s" name
                    Kind = "view-containment"
                    Logic = "QF_LIA"
                    Statement = sprintf "the view handed to write() for %s is exactly %d bytes and strictly inside its %d-byte storage (the terminator is never written)" (describe content) len storage
                    Source = born
                    Refs = ["CWE-787"]
                    Body = ViewContainment (len, len, storage) }
            yield { Id = sprintf "sentinel_%s" name
                    Kind = "terminator-sentinel"
                    Logic = "QF_BV"
                    Statement = sprintf "the final storage byte of %s is the 0x00 terminator" (describe content)
                    Source = born
                    Refs = ["CWE-170"]
                    Body = NulSentinel 0 } ]

    let layout =
        match userLiterals with
        | [] | [_] -> []
        | _ ->
            let storages = userLiterals |> List.map (fun (c, _) -> StringCollection.deriveByteLength c + 1)
            let span = List.sum storages
            [ { Id = "layout_user_strings"
                Kind = "memory-map-disjointness"
                Logic = "QF_LIA"
                Statement = sprintf "the %d user string storages, laid out consecutively, occupy pairwise-disjoint ranges spanning exactly %d bytes" storages.Length span
                Source = sprintf "%s (all reachable string literals)" (Path.GetFileName entryFile)
                Refs = ["CWE-787"; "CWE-125"]
                Body = ConsecutiveLayout (storages, span) } ]

    { Obligations = perString @ layout }

// ═══════════════════════════════════════════════════════════════════════════
// DESIGN-TIME FORM: SMT-LIB rendering (serialization of pre-computed facts)
// ═══════════════════════════════════════════════════════════════════════════

/// Render one obligation as an SMT-LIB scope (refutation style, named anchor)
let private bodyToSmtLib (id: string) (body: ObligationBody) : string list =
    match body with
    | StorageReservation (len, storage) ->
        [ sprintf "(assert (= %s (= %d (+ %d 1))))" id storage len
          sprintf "(assert (not %s))" id ]
    | ViewContainment (view, len, storage) ->
        [ sprintf "(assert (= %s (and (= %d %d) (< %d %d))))" id view len view storage
          sprintf "(assert (not %s))" id ]
    | NulSentinel lastByte ->
        [ sprintf "(assert (= %s (= #x%02x #x00)))" id lastByte
          sprintf "(assert (not %s))" id ]
    | ConsecutiveLayout (storages, span) ->
        let n = List.length storages
        let bases = [ for i in 0 .. n - 1 -> sprintf "b%d" i ]
        let sizes = List.toArray storages
        let decls = [ for b in bases -> sprintf "(declare-const %s Int)" b ]
        let adjacency =
            [ for i in 1 .. n - 1 ->
                sprintf "(assert (= %s (+ %s %d)))" bases[i] bases[i-1] sizes[i-1] ]
        let disjoint =
            [ for i in 0 .. n - 1 do
                for j in i + 1 .. n - 1 ->
                    sprintf "(or (<= (+ %s %d) %s) (<= (+ %s %d) %s))" bases[i] sizes[i] bases[j] bases[j] sizes[j] bases[i] ]
        decls
        @ [ sprintf "(assert (>= %s 0))" bases[0] ]
        @ adjacency
        @ [ sprintf "(assert (= %s (and %s (= (+ %s %d) (+ %s %d)))))"
                id (String.concat " " disjoint) bases[n-1] sizes[n-1] bases[0] span
            sprintf "(assert (not %s))" id ]

/// Render the full obligation set as a solver-ready SMT-LIB artifact.
/// One scope per obligation; `unsat` on every (check-sat) = all obligations hold.
let toSmtLib (obs: ObligationSet) : string =
    obs.Obligations
    |> List.map (fun ob ->
        String.concat "\n"
            ([ sprintf "; %s: %s" ob.Id ob.Statement
               sprintf "; origin: %s" ob.Source
               sprintf "(set-logic %s)" ob.Logic
               sprintf "(declare-const %s Bool)" ob.Id ]
             @ bodyToSmtLib ob.Id ob.Body
             @ [ "(check-sat)"; "(reset)" ]))
    |> String.concat "\n\n"

// ═══════════════════════════════════════════════════════════════════════════
// SERIALIZATION (intermediates: ledger + solver form)
// ═══════════════════════════════════════════════════════════════════════════

/// Write the design-time artifacts:
///   06a_obligations.json: the ledger (id, kind, statement, source)
///   06b_obligations.smt2: the solver-ready SMT-LIB form
let serialize (intermediatesDir: string) (obs: ObligationSet) : unit =
    if not (List.isEmpty obs.Obligations) then
        let ledger =
            {| version = "1.0"
               description = "Proof obligations coeffect: born in the PSG, design-time form"
               obligations =
                 [ for ob in obs.Obligations ->
                     {| id = ob.Id; kind = ob.Kind; logic = ob.Logic
                        statement = ob.Statement; source = ob.Source
                        refs = ob.Refs |} ] |}
        let options = JsonSerializerOptions(WriteIndented = true)
        File.WriteAllText(
            Path.Combine(intermediatesDir, "06a_obligations.json"),
            JsonSerializer.Serialize(ledger, options))
        File.WriteAllText(
            Path.Combine(intermediatesDir, "06b_obligations.smt2"),
            toSmtLib obs + "\n")
