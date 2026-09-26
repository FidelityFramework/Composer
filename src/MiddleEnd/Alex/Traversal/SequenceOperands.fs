/// Canonical transport for a Baker-settled sequence protocol. This reads exact
/// source participants and placed fields. It never chooses an origin, discovers
/// a frame from a type, emits a generator body or packs a function into data.
module Alex.Traversal.SequenceOperands

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

type Shape = private {
    Flow: SequenceFlow
    Family: SequenceFamily
    FunctionType: MLIRType
    EnvironmentType: MLIRType
}

let project (ctx: WitnessContext) occurrence : Result<Shape, string> =
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryStorage ctx.Graph with
    | Result.Error reason -> Result.Error reason
    | Result.Ok projection ->
        match projection.Sequences.TryFind occurrence with
        | None -> Result.Error "Sequence occurrence has no source-published protocol."
        | Some contract ->
            let environment = TMemRefStatic(contract.Family.Bytes, TInt(IntWidth 8))
            Result.Ok { Flow = contract.Flow; Family = contract.Family; EnvironmentType = environment
                        FunctionType = TFunc([environment], [TInt(IntWidth 1)]) }

let functionType shape = shape.FunctionType
let environmentType shape = shape.EnvironmentType
let family shape = shape.Family
let flow shape = shape.Flow
let componentTypes shape = [shape.FunctionType; shape.EnvironmentType]

/// The source owner has already validated representation, storage and exact
/// initializer participants. Witnessing consumes that immutable decision.
let copyContract (ctx: WitnessContext) acquisition =
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryStorage ctx.Graph with
    | Result.Error reason -> Result.Error reason
    | Result.Ok projection ->
        match projection.SequenceCopies.TryFind acquisition with
        | Some contract -> Result.Ok contract
        | None -> Result.Error "Fresh sequence acquisition has no source-published copy contract."

let create (shape: Shape) (code: Val) (environment: Val) : Result<SequenceOperand, string> =
    if code.Type <> shape.FunctionType || environment.Type <> shape.EnvironmentType then
        Result.Error "Sequence code and environment do not match their settled family signature."
    elif code.SSA = environment.SSA then Result.Error "Sequence code and environment must be separate values."
    else Result.Ok { Flow = shape.Flow; Family = shape.Family; Code = code; Environment = environment }

let bind (ctx: WitnessContext) occurrence code environment =
    project ctx occurrence
    |> Result.bind (fun shape -> create shape code environment)
    |> Result.bind (fun value -> MLIRAccumulator.bindSequence occurrence value ctx.Accumulator)

let reproject (ctx: WitnessContext) source destination =
    match MLIRAccumulator.recallSequence source ctx.Accumulator with
    | None -> Result.Error "Source sequence has not been witnessed in this operation scope."
    | Some value when value.Flow.Occurrence <> source -> Result.Error "Sequence was recalled under another occurrence's identity."
    | Some value ->
        project ctx destination |> Result.bind (fun shape ->
            if value.Family.Identity <> shape.Family.Identity || value.Flow.IsEnumerator <> shape.Flow.IsEnumerator ||
               value.Flow.ElementType <> shape.Flow.ElementType ||
               not (Set.isSubset value.Flow.Owners shape.Flow.Owners) then
                Result.Error "Sequence copy does not preserve its protocol, alternatives and actual value role."
            else create shape value.Code value.Environment)

let copy (ctx: WitnessContext) source destination =
    reproject ctx source destination
    |> Result.bind (fun value -> MLIRAccumulator.bindSequence destination value ctx.Accumulator)

let values (value: SequenceOperand) = [value.Code; value.Environment]
let code (value: SequenceOperand) = value.Code
let environment (value: SequenceOperand) = value.Environment
