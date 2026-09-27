/// CIRCT realization of the complete source Mealy contract. Source declaration
/// interpretation and pin assignment have already ended in Baker.
module BackEnd.CIRCT.HardwareRealization

open System.Globalization
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Core.Types.Pipeline
module Representation = Alex.CodeGeneration.TypeMapping

let private moduleFor (pointer: Result<int,string>) (stepSymbol: string) (plan: HardwareModuleWitness) =
    let operations = ResizeArray<MLIROp>()
    // Names are local to this backend-created hw.module region.
    let mutable next = 0
    let fresh () =
        let result = V(NodeId.value plan.Site, next)
        next <- next + 1
        result
    let append operation = operations.Add operation
    let hw operation = append (MLIROp.HWOp operation)
    let constant value ty =
        let result = fresh ()
        // Decimal text retains the source integer, including unsigned values
        // outside the tooling host's signed 64-bit range.
        append (MLIROp.RawMLIR(sprintf "%s = arith.constant %s : %s"
                    (ssaToString result) ((value:bigint).ToString(CultureInfo.InvariantCulture)) (typeToString pointer ty)))
        result
    let bit = TInt(IntWidth 1)
    let reset = plan.Pins.Reset |> Option.defaultWith (fun () -> invalidOp "Published hardware plan lacks its explicit reset.")
    let clock = Arg 0
    let one = constant 1I bit
    let resetSignal =
        if reset.IsExternal then
            if reset.ActiveHigh then Arg 1 else
            let inverted = fresh ()
            append (MLIROp.CombOp(CombXor(inverted, Arg 1, one, bit)))
            inverted
        else
            let initial, seen, active = fresh (), fresh (), fresh ()
            // CIRCT seq.compreg initial is an explicit immutable value. This
            // realizes the declared one-cycle POR; no device INIT default is assumed.
            // https://circt.llvm.org/docs/Dialects/Seq/#seqinitial-circtseqinitialop
            append (MLIROp.RawMLIR(sprintf "%s = seq.initial () {\n  %%zero = arith.constant 0 : i1\n  seq.yield %%zero : i1\n} : () -> !seq.immutable<i1>" (ssaToString initial)))
            append (MLIROp.RawMLIR(sprintf "%s = seq.compreg %s, %s initial %s : i1"
                        (ssaToString seen) (ssaToString one) (ssaToString clock) (ssaToString initial)))
            append (MLIROp.CombOp(CombXor(active, seen, one, bit)))
            active
    let stateType = Representation.representationType plan.StateRepresentation
    let resultType = Representation.representationType plan.ResultRepresentation
    let stateFields =
        match stateType with TStruct(fields, _) -> fields | _ -> invalidOp "Source hardware state is not a record."
    let registers = stateFields |> List.map (fun (_, ty) -> fresh (), ty)
    let nextFields = stateFields |> List.map (fun _ -> fresh ())
    List.zip3 plan.ResetFields registers nextFields |> List.iter (fun (field, (register, ty), nextState) ->
        let initial = constant field.Reset ty
        append (MLIROp.SeqOp(SeqCompreg(register, nextState, clock, Some(resetSignal, initial), ty))))
    let state = fresh ()
    hw (HWStructCreate(state, registers, stateType))
    let baseArgument = if reset.IsExternal then 2 else 1
    let inputs = plan.InputPorts |> List.mapi (fun index port -> port.Path, (Arg(baseArgument + index), Representation.representationType port.Representation)) |> Map.ofList
    let rec pack path representation =
        let ty = Representation.representationType representation
        match representation with
        | ValueRepresentation.Record(fields, _) ->
            let operands = fields |> List.map (fun (name, field) -> pack (path @ [name]) field)
            let result = fresh ()
            hw (HWStructCreate(result, operands, ty))
            result, ty
        | _ ->
            match inputs.TryFind path with
            | Some(value, actual) when actual = ty -> value, ty
            | _ -> invalidOp "Published input port path does not retain its exact physical value."
    let arguments =
        (state, stateType) :: (plan.InputRepresentation |> Option.map (pack []) |> Option.toList)
    if arguments.Length <> plan.Parameters.Length then invalidOp "Published hardware Step parameter count differs from its state/input contract."
    let instance = fresh ()
    let namedArguments = List.map2 (fun (name, _) (ssa, ty) -> name, ssa, ty) plan.Parameters arguments
    hw (HWInstance(instance, "step", stepSymbol, namedArguments, ["result", resultType]))
    let extract source sourceType name =
        let fieldType =
            match sourceType with
            | TStruct(fields, _) -> fields |> List.tryFind (fst >> (=) name) |> Option.map snd
            | _ -> None
        let fieldType = fieldType |> Option.defaultWith (fun () -> invalidOp "Published hardware port/result path lacks its field representation.")
        let result = fresh ()
        hw (HWStructExtract(result, source, name, sourceType))
        result, fieldType
    let nextState, output =
        if resultType = stateType then (instance, stateType), None
        else extract instance resultType "Item1", Some(extract instance resultType "Item2")
    List.zip stateFields nextFields |> List.iter (fun ((name, _), result) ->
        hw (HWStructExtract(result, fst nextState, name, snd nextState)))
    let outputs = plan.OutputPorts |> List.map (fun port ->
        let source = output |> Option.defaultWith (fun () -> invalidOp "Published output ports lack their Step output component.")
        let value, ty = port.Path |> List.fold (fun (ssa, ty) field -> extract ssa ty field) source
        if ty <> Representation.representationType port.Representation then invalidOp "Published output port changed representation."
        value, ty)
    hw (HWOutput outputs)
    let ports =
        [yield plan.Pins.Clock.PortName, TSeqClock
         if reset.IsExternal then yield reset.PortName, bit
         yield! plan.InputPorts |> List.map (fun port -> port.Name, Representation.representationType port.Representation)]
    let outputPorts = plan.OutputPorts |> List.map (fun port -> port.Name, Representation.representationType port.Representation)
    MLIROp.HWOp(HWModule(plan.Name, ports, outputPorts, List.ofSeq operations))

let private declarations (input: BackEndInput) =
    let catalog = input.Catalog |> Option.defaultWith (fun () -> invalidOp "CIRCT realization requires the current source catalog.")
    let graph = Core.WitnessArtifacts.graph catalog.Scope
    let published = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.trySpatial graph |> Result.defaultWith invalidOp
    let plans = input.Operations |> List.choose (function MLIROp.SpatialModule(SpatialModuleWitness.Hardware plan) -> Some plan | _ -> None)
    if (plans |> List.map _.Site |> List.sort) <> (published.Hardware.Keys |> Seq.toList) then
        invalidOp "CIRCT input lost or duplicated a published hardware declaration."
    if input.Operations |> List.exists (function
        | MLIROp.SpatialModule(SpatialModuleWitness.Kernel _)
        | MLIROp.FuncOp(BoundaryFuncDecl _ | IntrinsicWriteDecl _) -> true
        | _ -> false) then invalidOp "CIRCT has no realization for kernel or foreign runtime declarations."
    plans |> List.map (fun plan ->
        if published.Hardware.TryFind plan.Site <> Some plan then invalidOp "CIRCT hardware plan differs from current source publication."
        let symbol = Alex.CodeGeneration.CallableSymbols.tryBinding graph plan.Implementation
                     |> Option.defaultWith (fun () -> invalidOp "The source Step implementation lacks its published symbol.")
        let parameterTypes = plan.StateRepresentation :: Option.toList plan.InputRepresentation |> List.map Representation.representationType
        let expectedInputs = List.map2 (fun (name, _) ty -> name, ty) plan.Parameters parameterTypes
        let expectedOutputs = ["result", Representation.representationType plan.ResultRepresentation]
        match input.Operations |> List.choose (function MLIROp.HWOp(HWModule(name, inputs, outputs, _)) when name = symbol -> Some(inputs,outputs) | _ -> None) with
        | [inputs,outputs] when inputs = expectedInputs && outputs = expectedOutputs -> ()
        | _ -> invalidOp "The witnessed Step declaration does not match the exact published hardware signature."
        plan.Site, symbol) |> Map.ofList

let validate (input: BackEndInput) (_context: BackEndContext) =
    try declarations input |> ignore; Ok ()
    with error -> Error("CIRCT hardware admission: " + error.Message)

let realize (input: BackEndInput) : Result<BackEndInput,string> =
    try
        let symbols = declarations input
        let operations = input.Operations |> List.map (function
            | MLIROp.SpatialModule(SpatialModuleWitness.Kernel _) -> invalidOp "A kernel module has no CIRCT realization."
            | MLIROp.SpatialModule(SpatialModuleWitness.Hardware plan) ->
                moduleFor input.PointerBits symbols[plan.Site] plan
            | operation -> operation)
        let text =
            match input.ModuleName with
            | Some name -> moduleToString input.PointerBits name operations
            | None -> sprintf "module {\n%s\n}" (opsToString input.PointerBits operations "  ")
        Ok { input with Operations = operations; Text = text }
    with error -> Error("CIRCT hardware realization: " + error.Message)
