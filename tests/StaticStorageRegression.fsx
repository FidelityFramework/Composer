// Build Composer first. Exercises the final-emission correspondence boundary.
#I "../src/bin/Debug/net10.0"
#r "Fidelity.Data.dll"
#r "Clef.Compiler.Service.dll"
#r "Composer.dll"
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
let builder = NodeBuilder()
let first = builder.Create(SemanticKind.Literal(NativeLiteral.String "A"), Types.stringType, dummyRange)
let second = builder.Create(SemanticKind.Literal(NativeLiteral.String "λ"), Types.stringType, dummyRange)
let entries = [
    { NodeIds = [first.Id]; Content = "A"; Offset = 0; Length = 1; StorageLength = 2 }
    { NodeIds = [second.Id]; Content = "λ"; Offset = 2; Length = 2; StorageLength = 3 } ]
let pool : StaticStringPool = {
    Symbol = "test_pool"; Bytes = [65uy;0uy;206uy;187uy;0uy;0uy;0uy;0uy]
    Alignment = 8; Size = 8; UsedSize = 5; Entries = entries
    SpaceName = "rodata"; Capacity = 8L; SpaceAlignment = 8; Granularity = 8; DeclarationNode = NodeId.fresh () }
let obligation = { Id = "layout_pool"; Kind = "static-storage-layout"; Logic = "QF_LIA"; Statement = "test"
                   Source = "test"; Refs = []; Body = ObligationBody.StaticStorageLayout([(0,2,1);(2,3,1)],5,8,8,8L,8,8) }
builder.Create(SemanticKind.Obligation obligation, Types.unitType, dummyRange) |> ignore
let graph = { builder.Build [] with StaticStringPool = Some pool }
let literalOps = entries |> List.collect (fun e ->
    Alex.Patterns.LiteralPatterns.stringPoolView pool e (Alex.Traversal.Values.values e.NodeIds.Head) |> fst)
let allocation = MLIROp.GlobalBytePool(pool.Symbol,pool.Bytes,pool.Alignment,[obligation.Id])
let ops = allocation :: literalOps
let check label expectSuccess graph ops =
    let result = Alex.Traversal.StaticStorageValidation.validate graph ops
    if Result.isOk result <> expectSuccess then failwithf "%s: %A" label result
check "settled pool" true graph ops
let mutate f = ops |> List.map f
check "changed bytes" false graph (mutate (function
    | MLIROp.GlobalBytePool(n,_,a,o) -> MLIROp.GlobalBytePool(n,[0uy;0uy;206uy;187uy;0uy;0uy;0uy;0uy],a,o)
    | op -> op))
check "changed alignment" false graph (mutate (function
    | MLIROp.GlobalBytePool(n,b,_,o) -> MLIROp.GlobalBytePool(n,b,1,o) | op -> op))
check "changed offset" false graph (mutate (function
    | MLIROp.IndexOp(IndexOp.IndexConst(s,2L)) -> MLIROp.IndexOp(IndexOp.IndexConst(s,1L)) | op -> op))
check "changed symbol" false graph (mutate (function
    | MLIROp.MemRefOp(MemRefOp.GetGlobal(s,n,t)) when s = Alex.Traversal.Values.value first.Id 0 ->
        MLIROp.MemRefOp(MemRefOp.GetGlobal(s,"independent_string",t)) | op -> op))
check "missing anchor" false graph (mutate (function
    | MLIROp.GlobalBytePool(n,b,a,_) -> MLIROp.GlobalBytePool(n,b,a,[]) | op -> op))
check "missing proof" false {graph with Nodes = graph.Nodes |> Map.filter (fun _ node -> match node.Kind with SemanticKind.Obligation _ -> false | _ -> true)} ops
check "duplicate pool" false graph (allocation :: ops)
check "missing plan" false {graph with StaticStringPool = None} ops
let wrongBytesPool = { pool with Bytes = [0uy;0uy;206uy;187uy;0uy;0uy;0uy;0uy] }
check "corrupt plan and emitted bytes agree" false {graph with StaticStringPool = Some wrongBytesPool}
    (MLIROp.GlobalBytePool(pool.Symbol,wrongBytesPool.Bytes,pool.Alignment,[obligation.Id]) :: literalOps)
printfn "PASS 10 static-storage correspondence cases (including offset, bytes, alignment, symbol, anchor and plan mutations)"
