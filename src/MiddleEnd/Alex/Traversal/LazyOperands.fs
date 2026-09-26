/// Passive typed transport for explicit lazy instances. Source settlement owns
/// the memoization algorithm, lifetime, fields and thunk boundary. This module
/// preserves the actual environment operand at each occurrence.
module Alex.Traversal.LazyOperands

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators

type Shape = private {
    Occurrence: NodeId
    Layout: LazyLayout
    FunctionType: MLIRType
    EnvironmentType: MLIRType
}

let private validated = System.Runtime.CompilerServices.ConditionalWeakTable<SemanticGraph, Map<NodeId, LazyLayout>>()

let layout (ctx: WitnessContext) owner =
    let current = validated.GetValue(ctx.Graph, fun graph ->
        graph.Codata.Value.LazyLayouts |> Map.filter (fun _ layout ->
            Clef.Compiler.Nanopass.LazyRuntime.validate graph layout))
    current.TryFind owner

let layoutAt (ctx: WitnessContext) occurrence =
    ctx.Graph.Codata.Value.LazyOrigins.TryFind occurrence
    |> Option.bind (layout ctx)

let project (ctx: WitnessContext) occurrence : Result<Shape, string> =
    match ctx.Graph.Nodes.TryFind occurrence, layoutAt ctx occurrence with
    | Some { Type = ty }, Some layout ->
        match applySubst ty, Clef.Compiler.PSGSaturation.SemanticGraph.LazyValues.instance ctx.Graph layout.Owner with
        | NativeType.TLazy element, Some contract when applySubst element = applySubst contract.ElementType ->
            try
                let environment = TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
                let result = mapTypeAt contract.ThunkBody contract.ElementType ctx |> narrowType ctx.Coeffects ctx.Graph contract.ThunkBody
                let results = if result = TVoid then [] else [result]
                Result.Ok { Occurrence = occurrence; Layout = layout; EnvironmentType = environment
                            FunctionType = TFunc([environment], results) }
            with ex -> Result.Error ex.Message
        | _ -> Result.Error "Lazy occurrence does not retain its exact settled source element and thunk boundary."
    | _ -> Result.Error "Lazy occurrence has no complete source layout and lifetime contract."

let functionType shape = shape.FunctionType
let environmentType shape = shape.EnvironmentType
let componentTypes shape = [shape.FunctionType; shape.EnvironmentType]
let contract shape = shape.Layout

let create (shape: Shape) (code: Val) (environment: Val) : Result<LazyOperand, string> =
    if code.Type <> shape.FunctionType || environment.Type <> shape.EnvironmentType then
        Result.Error "Lazy thunk and environment disagree with the settled physical boundary."
    elif code.SSA = environment.SSA then Result.Error "Lazy thunk and environment must remain separate operands."
    else Result.Ok { Occurrence = shape.Occurrence; Layout = shape.Layout; Code = code; Environment = environment }

let reproject (ctx: WitnessContext) source destination =
    match MLIRAccumulator.recallLazy source ctx.Accumulator with
    | Some value when value.Occurrence = source ->
        project ctx destination |> Result.bind (fun shape ->
            if value.Layout <> shape.Layout then Result.Error "Lazy alias no longer retains its settled instance schema."
            else create shape value.Code value.Environment)
    | _ -> Result.Error "Lazy source has not been witnessed at this operation scope."

let values (value: LazyOperand) = [value.Code; value.Environment]
let code (value: LazyOperand) = value.Code
let environment (value: LazyOperand) = value.Environment
let occurrence (value: LazyOperand) = value.Occurrence
let layoutOf (value: LazyOperand) = value.Layout
