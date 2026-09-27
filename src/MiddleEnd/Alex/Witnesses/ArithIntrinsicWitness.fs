/// ArithIntrinsicWitness - Witness arithmetic/conversion intrinsic operations
///
/// The intrinsic identity at the application's callee selects the one
/// per-operation parser that owns the node. That parser either witnesses the
/// settled operation or reports the premise it lacks; its refusal is never
/// discarded as a skip.
///
/// NANOPASS: Handles Operators.*, Convert.* and Math.truncate intrinsic applications.
module Alex.Witnesses.ArithIntrinsicWitness

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ApplicationPatterns

/// The intrinsic named at the application's callee (seen through its annotation).
let private calleeIntrinsic (ctx: WitnessContext) (node: SemanticNode) =
    match node.Kind with
    | SemanticKind.Application (callee, _) ->
        match ctx.Graph.Nodes.TryFind callee with
        | Some { Kind = SemanticKind.Intrinsic info } -> Some info
        | Some { Kind = SemanticKind.TypeAnnotation (inner, _) } ->
            match ctx.Graph.Nodes.TryFind inner with
            | Some { Kind = SemanticKind.Intrinsic info } -> Some info
            | _ -> None
        | _ -> None
    | _ -> None

/// The one parser that owns this intrinsic; other intrinsics belong to other witnesses.
let private owningPattern (info: IntrinsicInfo) =
    match info.Module, classifyAtomicOp info with
    | IntrinsicModule.Operators, (BinaryArith _ | Comparison _) -> Some pBinaryArithIntrinsic
    | IntrinsicModule.Operators, UnaryArith _ -> Some pUnaryArithIntrinsic
    | IntrinsicModule.Operators, _ when info.Operation = "ignore" -> Some pIgnoreIntrinsic
    | IntrinsicModule.Convert, _ -> Some pTypeConversionIntrinsic
    | IntrinsicModule.Math, _ when info.Operation = "truncate" -> Some pTruncateIntrinsic
    | _ -> None

let private witnessArithIntrinsic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match calleeIntrinsic ctx node |> Option.bind (fun info -> owningPattern info |> Option.map (fun pattern -> info, pattern)) with
    | None -> WitnessOutput.skip
    | Some (info, pattern) ->
        match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error reason ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "ArithIntrinsic") (Some $"%A{info.Module}.{info.Operation}")
                $"PSG settlement did not settle a witnessable operation for intrinsic application %A{info.Module}.{info.Operation} (node {NodeId.value node.Id}): {reason}"

let nanopass : Nanopass = { Name = "ArithIntrinsic"; Witness = witnessArithIntrinsic }
