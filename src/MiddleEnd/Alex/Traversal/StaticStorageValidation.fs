/// Correspondence boundary between settled BAREWire storage and emitted MLIR.
/// Validates the final structured ops after middle-end passes, before serialization.
module Alex.Traversal.StaticStorageValidation

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types

let rec private flatten ops =
    ops |> List.collect (fun op ->
        op ::
        (match op with
         | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _))
         | MLIROp.NoUnwindFunction (FuncOp.FuncDef (_, _, _, body, _))
         | MLIROp.HWOp (HWOp.HWModule (_, _, _, body))
         | MLIROp.Block (_, body) | MLIROp.Region body -> flatten body
         | MLIROp.SCFOp (SCFOp.If (_, yes, no, _)) -> flatten (yes @ Option.defaultValue [] no)
         | MLIROp.SCFOp (SCFOp.While (condition, body)) -> flatten (condition @ body)
         | MLIROp.SCFOp (SCFOp.For (_, _, _, body)) -> flatten body
         | MLIROp.SCFOp (SCFOp.IndexSwitch (_, cases, fallback, _)) -> flatten ((cases |> List.collect snd) @ fallback)
         | _ -> []))

let validate (graph: SemanticGraph) (ops: MLIROp list) : Result<unit, string> =
    let all = flatten ops
    let pools = all |> List.choose (function MLIROp.GlobalBytePool (n,b,a,o) -> Some (n,b,a,o) | _ -> None)
    let errors = ResizeArray<string>()
    match graph.StaticStringPool with
    | None -> if not pools.IsEmpty then errors.Add "emitted a byte pool without a settled BAREWire plan"
    | Some pool ->
        let expectedBody =
            ObligationBody.StaticStorageLayout (
                pool.Entries |> List.map (fun e -> e.Offset, e.StorageLength, 1),
                pool.UsedSize, pool.Size, pool.Alignment, pool.Capacity, pool.SpaceAlignment, pool.Granularity)
        let layoutAnchors =
            graph.Nodes |> Map.toList |> List.choose (fun (_, node) ->
                match node.Kind with SemanticKind.Obligation ob when ob.Body = expectedBody -> Some ob.Id | _ -> None)
        if layoutAnchors.IsEmpty then errors.Add "settled pool has no matching compiler layout obligation"
        match pools with
        | [name, bytes, alignment, anchors] when name = pool.Symbol ->
            if bytes <> pool.Bytes || bytes.Length <> pool.Size then errors.Add "emitted pool bytes/extent differ from the settled plan"
            if alignment <> pool.Alignment then errors.Add "emitted pool alignment differs from the settled plan"
            if layoutAnchors |> List.exists (fun anchor -> not (List.contains anchor anchors)) then
                errors.Add "emitted pool lost its compiler layout obligation anchor"
        | _ -> errors.Add "expected exactly one allocation for the settled BAREWire string pool"
        let entries =
            pool.Entries |> List.collect (fun entry -> entry.NodeIds |> List.map (fun id -> id, entry)) |> Map.ofList
        let poolBytes = Array.ofList pool.Bytes
        for entry in pool.Entries do
            let logical = System.Text.Encoding.UTF8.GetBytes entry.Content
            let ending = int64 entry.Offset + int64 entry.StorageLength
            if entry.Offset < 0 || ending > int64 poolBytes.Length || entry.StorageLength <> logical.Length + 1 || entry.Length <> logical.Length then
                errors.Add "source literal extent differs from the settled pool"
            elif Array.sub poolBytes entry.Offset logical.Length <> logical || poolBytes[entry.Offset + logical.Length] <> 0uy then
                errors.Add "source literal bytes/sentinel differ from the settled pool"
            for id in entry.NodeIds do
                match graph.Nodes.TryFind id with
                | Some { Kind = SemanticKind.Literal (Clef.Compiler.NativeTypedTree.NativeTypes.NativeLiteral.String content) } when content = entry.Content -> ()
                | _ -> errors.Add "pool entry does not refer to its source string"
                let sourceSSA = Values.value id 0
                let definitions = all |> List.choose (function
                    | MLIROp.MemRefOp (MemRefOp.GetGlobal (ssa, name, ty)) when ssa = sourceSSA -> Some (name, ty)
                    | _ -> None)
                if definitions.IsEmpty || definitions |> List.exists (fun (name, _) -> name <> pool.Symbol) then
                    errors.Add "source literal does not reference the settled pool allocation"
        let gets = all |> List.choose (function
            | MLIROp.MemRefOp (MemRefOp.GetGlobal (ssa, name, ty)) when name = pool.Symbol -> Some (ssa, ty)
            | _ -> None)
        if gets.IsEmpty then errors.Add "settled pool has no emitted views"
        for get, actualType in gets do
            match get with
            | V (id, 0) ->
                match entries.TryFind (NodeId id) with
                | None -> errors.Add "pool view has no corresponding source literal"
                | Some entry ->
                    let s k = V(id, k)
                    let storage = TMemRefStatic(pool.Size, TInt(IntWidth 8))
                    let content = TMemRefStatic(entry.Length, TInt(IntWidth 8))
                    let dynamic = TMemRef(TInt(IntWidth 8))
                    if actualType <> storage then errors.Add "pool get_global has the wrong allocation extent"
                    let offsets = all |> List.choose (function MLIROp.IndexOp (IndexOp.IndexConst (ssa, value)) when ssa = s 3 -> Some value | _ -> None)
                    if offsets.IsEmpty || offsets |> List.exists ((<>) (int64 entry.Offset)) then errors.Add "literal byte offset differs from the settled plan"
                    let views = all |> List.choose (function MLIROp.MemRefOp (MemRefOp.View (ssa, src, off, fromTy, toTy)) when ssa = s 1 -> Some (src,off,fromTy,toTy) | _ -> None)
                    if views.IsEmpty || views |> List.exists ((<>) (s 0,s 3,storage,content)) then errors.Add "literal view differs from the settled plan"
                    let casts = all |> List.choose (function MLIROp.MemRefOp (MemRefOp.Cast (ssa, src, fromTy, toTy)) when ssa = s 2 -> Some (src,fromTy,toTy) | _ -> None)
                    if casts.IsEmpty || casts |> List.exists ((<>) (s 1,content,dynamic)) then errors.Add "literal cast lost its settled content view"
            | _ -> errors.Add "pool get_global is not attached to a source literal"
    if errors.Count = 0 then Result.Ok ()
    else Result.Error ("BAREWire storage correspondence failed: " + String.concat "; " errors)

/// Compare the complete emitted writable inventory before a backend can place
/// it. This proves identity/type correspondence only, not linked-region fit.
let validateWritable arch (graph: SemanticGraph) (ops: MLIROp list) =
    let all = flatten ops
    let declarations = all |> List.choose (function MLIROp.GlobalMemref(name, ty, authority) -> Some(name, ty, authority) | _ -> None)
    let errors = ResizeArray<string>()
    let observed = ResizeArray<string * ProgramStorageEntry>()
    let current = Clef.Compiler.PSGSaturation.SemanticGraph.ProgramStorage.read graph
    match current with
    | None -> errors.Add "the source program-storage inventory is stale"
    | Some inventory ->
        for KeyValue(_, reason) in inventory.Unresolved do errors.Add reason
        let mutable names = Set.empty
        let mutable identities = Set.empty
        for name, ty, authority in declarations do
            if names.Contains name then errors.Add ("duplicate writable declaration: " + name)
            names <- names.Add name
            match authority with
            | None -> errors.Add ("writable declaration lacks source allocation authority: " + name)
            | Some entry ->
                if identities.Contains entry.Identity then errors.Add (sprintf "source storage %A was emitted more than once" entry.Identity)
                identities <- identities.Add entry.Identity
                if inventory.Entries.TryFind entry.Identity <> Some entry then
                    errors.Add ("writable declaration carries a stale or foreign inventory entry: " + name)
                elif Alex.Patterns.MemoryPatterns.programStorageType arch graph entry <> Some ty then
                    errors.Add ("writable declaration type disagrees with source storage: " + name)
                else observed.Add(name, entry)
                let reads = all |> List.choose (function
                    | MLIROp.MemRefOp(MemRefOp.GetGlobal(_, symbol, actual)) when symbol = name -> Some actual
                    | _ -> None)
                if reads.IsEmpty || reads |> List.exists ((<>) ty) then
                    errors.Add ("writable allocation views disagree with its declaration: " + name)
        for KeyValue(identity, _) in inventory.Entries do
            if not (identities.Contains identity) then errors.Add (sprintf "source writable allocation %A has no emitted declaration" identity)
    if errors.Count = 0 then Ok(List.ofSeq observed)
    else Error("Writable storage correspondence failed: " + String.concat "; " errors)
