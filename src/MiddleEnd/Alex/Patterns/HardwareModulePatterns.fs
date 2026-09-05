/// HardwareModulePatterns - Mealy machine builder for FPGA [<HardwareModule>] bindings
///
/// Models the full Mealy machine: (State × Input) → (State × Output)
///   - State feeds back through seq.compreg registers
///   - Input comes from top-level input ports (Design<'S,'R> with step: 'S -> I -> 'S * 'R)
///   - Output goes to top-level output ports
///   - Step function is instantiated via hw.instance (already emitted by LambdaWitness)
///
/// Design<'State, 'Report> record is compile-time metadata:
///   InitialState → register reset values (NativeLiteral preserves source width)
///   Step         → VarRef to function → hw.instance instantiation
///   Clock        → clock port (in %clk: !seq.clock)
///
/// Every value of the hw.module body is read from the HardwareModuleLayout that
/// SSAAssignment derived for the binding, in the order this file emits them; nothing
/// is numbered here, and a register's feedback operand is its NextFields entry, so
/// no forward reference is computed. A disagreement between the layout and the
/// design's shape is reported, never patched.
///
/// Target MLIR (full Mealy — internal POR, no external rst port):
///   hw.module @name(in %clk: !seq.clock, in %inputs: <inputType>,
///                   out outputs: <outputType>) {
///     %por_one = arith.constant 1 : i1
///     %por_reg = seq.compreg %por_one, %clk : i1         // INIT=0, goes to 1
///     %por_rst = comb.xor %por_reg, %por_one : i1        // active first cycle
///     %init0 = arith.constant <resetVal> : <fieldType>
///     %reg0 = seq.compreg %next0, %clk reset %por_rst, %init0 : <fieldType>
///     %state = hw.struct_create (%reg0, ...) : <stateType>
///     %result = hw.instance "step_inst" @step(state: %state, inputs: %inputs)
///     %nextState = hw.struct_extract %result["Item1"] : <resultType>
///     %outputs = hw.struct_extract %result["Item2"] : <resultType>
///     %next0 = hw.struct_extract %nextState["field0"] : <stateType>
///     hw.output %outputs : <outputType>
///   }
module Alex.Patterns.HardwareModulePatterns

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open PSGElaboration.Coeffects

// ═══════════════════════════════════════════════════════════
// PSG METADATA EXTRACTION (compile-time, no MLIR emission)
// ═══════════════════════════════════════════════════════════

/// Resolve a VarRef to the qualified function name it references
/// Traverses VarRef → definition Binding → parent ModuleDef for qualification
let resolveStepFunctionName (graph: SemanticGraph) (stepNodeId: NodeId) : string option =
    match SemanticGraph.tryGetNode stepNodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.VarRef (name, Some defId) ->
            // Get the definition binding
            match SemanticGraph.tryGetNode defId graph with
            | Some defNode ->
                match defNode.Kind with
                | SemanticKind.Binding (bindingName, _, _, _) ->
                    // Check for ModuleDef parent for qualification
                    match defNode.Parent with
                    | Some moduleId ->
                        match SemanticGraph.tryGetNode moduleId graph with
                        | Some moduleNode ->
                            match moduleNode.Kind with
                            | SemanticKind.ModuleDef (moduleName, _) ->
                                Some (sprintf "%s.%s" moduleName bindingName)
                            | _ -> Some bindingName
                        | None -> Some bindingName
                    | None -> Some bindingName
                | _ -> None
            | None -> None
        | _ -> None
    | None -> None

// ═══════════════════════════════════════════════════════════
// MEALY MACHINE BUILDER
// ═══════════════════════════════════════════════════════════

/// Information extracted from a Design<S,R> record — models a Mealy machine:
/// (State × Input) → (State × Output)
type MealyMachineInfo = {
    /// Qualified name for the hw.module (e.g., "HelloFPGA.counter")
    ModuleName: string
    /// Qualified name of the step function (e.g., "HelloFPGA.step")
    StepFunctionName: string
    /// State type as TStruct
    StateType: MLIRType
    /// State field info: (fieldName, mlirType, resetValue) — NativeLiteral preserves source width
    StateFields: (string * MLIRType * NativeLiteral) list
    /// Input type (from step function's second parameter). None for Design<'S>, Some for Design<'S,'R>.
    InputType: MLIRType option
    /// Output type (from step function's return tuple Item2). None for Design<'S>, Some for Design<'S,'R>.
    OutputType: MLIRType option
}

/// Extract the raw int64 value from a NativeLiteral for ArithOp.ConstI emission.
/// The type width is carried separately by the MLIRType in the state field tuple.
let private literalToInt64 (lit: NativeLiteral) : int64 =
    match lit with
    | NativeLiteral.Int (v, _) -> v
    | NativeLiteral.UInt (v, _) -> int64 v
    | NativeLiteral.Bool true -> 1L
    | NativeLiteral.Bool false -> 0L
    | _ -> failwith $"Unsupported NativeLiteral for FPGA reset value: {lit}"

/// A per-state-field role of the layout must carry one value per state field
let private perField (role: string) (info: MealyMachineInfo) (ssas: SSA list) : SSA[] =
    if ssas.Length <> info.StateFields.Length then
        failwithf "HardwareModulePatterns: %s derived %d %s for %d state fields" info.ModuleName ssas.Length role info.StateFields.Length
    List.toArray ssas

/// The step function's result type: (State × Output) when the step reports, else the state
let private stepResultTypeOf (info: MealyMachineInfo) : MLIRType =
    match info.OutputType with
    | Some outTy -> TStruct [("Item1", info.StateType); ("Item2", outTy)]
    | None -> info.StateType

/// Reset constants: one arith.constant per state field, from the InitialState literals
let private resetOps (info: MealyMachineInfo) (layout: HardwareModuleLayout) : MLIROp list =
    let resets = perField "reset values" info layout.ResetValues
    info.StateFields |> List.mapi (fun i (_, fieldTy, resetLit) ->
        MLIROp.ArithOp (ArithOp.ConstI (resets.[i], literalToInt64 resetLit, fieldTy)))

/// State registers: seq.compreg per state field, fed back from its NextFields value
let private registerOps (info: MealyMachineInfo) (layout: HardwareModuleLayout) (clkSSA: SSA) (rstSSA: SSA) : MLIROp list =
    let regs = perField "registers" info layout.Registers
    let nexts = perField "next-state fields" info layout.NextFields
    let resets = perField "reset values" info layout.ResetValues
    info.StateFields |> List.mapi (fun i (_, fieldTy, _) ->
        MLIROp.SeqOp (SeqOp.SeqCompreg (regs.[i], nexts.[i], clkSSA, Some (rstSSA, resets.[i]), fieldTy)))

/// The current state: hw.struct_create of the registers
let private stateOp (info: MealyMachineInfo) (layout: HardwareModuleLayout) : MLIROp =
    let regs = perField "registers" info layout.Registers
    let fieldVals = info.StateFields |> List.mapi (fun i (_, ty, _) -> (regs.[i], ty))
    MLIROp.HWOp (HWOp.HWStructCreate (layout.State, fieldVals, info.StateType))

/// The step result decomposed: Item1 (the next state) and Item2 (the outputs) when the step
/// reports; otherwise the instance result is the next state
let private stepResult (info: MealyMachineInfo) (layout: HardwareModuleLayout) : SSA * SSA option * MLIROp list =
    let stepResultType = stepResultTypeOf info
    match info.OutputType, layout.StepResult with
    | Some _, Some (nsSA, oSSA) ->
        (nsSA, Some oSSA,
         [ MLIROp.HWOp (HWOp.HWStructExtract (nsSA, layout.Instance, "Item1", stepResultType))
           MLIROp.HWOp (HWOp.HWStructExtract (oSSA, layout.Instance, "Item2", stepResultType)) ])
    | None, None -> (layout.Instance, None, [])
    | Some _, None -> failwithf "HardwareModulePatterns: %s reports an output but no step-result values were derived" info.ModuleName
    | None, Some _ -> failwithf "HardwareModulePatterns: %s reports no output but step-result values were derived" info.ModuleName

/// Next-state field extraction for register feedback
let private nextFieldOps (info: MealyMachineInfo) (layout: HardwareModuleLayout) (nextStateSSA: SSA) : MLIROp list =
    let nexts = perField "next-state fields" info layout.NextFields
    info.StateFields |> List.mapi (fun i (fieldName, _, _) ->
        MLIROp.HWOp (HWOp.HWStructExtract (nexts.[i], nextStateSSA, fieldName, info.StateType)))

/// Build the Mealy machine hw.module (struct ports) from extracted Design info and the
/// binding's layout.
///
///   - State feeds back through seq.compreg registers
///   - Input comes from top-level input ports (when InputType is Some)
///   - Output goes to top-level output ports (when OutputType is Some)
///   - Step function is instantiated via hw.instance
let buildMealyMachineModule (info: MealyMachineInfo) (layout: HardwareModuleLayout) : MLIROp =
    // Input port SSAs: clk (Arg 0), rst (Arg 1), inputs (Arg 2 if present)
    let clkSSA = SSA.Arg 0
    let rstSSA = SSA.Arg 1
    let inputsSSA = if info.InputType.IsSome then Some (SSA.Arg 2) else None

    let stepInputs =
        match inputsSSA, info.InputType with
        | Some iSSA, Some iTy -> [("state", layout.State, info.StateType); ("inputs", iSSA, iTy)]
        | _ -> [("s", layout.State, info.StateType)]
    let instanceOp =
        MLIROp.HWOp (HWOp.HWInstance (layout.Instance, "step_inst", info.StepFunctionName, stepInputs, [("result", stepResultTypeOf info)]))

    let nextStateSSA, outputSSA, decomposeOps = stepResult info layout

    let outputOp =
        match outputSSA, info.OutputType with
        | Some oSSA, Some outTy -> MLIROp.HWOp (HWOp.HWOutput [(oSSA, outTy)])
        | _ -> MLIROp.HWOp (HWOp.HWOutput [(layout.State, info.StateType)])

    // In the layout's order: reset constants, registers, state, instance, step result, next fields
    let bodyOps =
        resetOps info layout
        @ registerOps info layout clkSSA rstSSA
        @ [stateOp info layout; instanceOp]
        @ decomposeOps
        @ nextFieldOps info layout nextStateSSA
        @ [outputOp]

    let inputs =
        [("clk", TSeqClock); ("rst", TInt (IntWidth 1))]
        @ (match info.InputType with Some iTy -> [("inputs", iTy)] | None -> [])
    let outputs =
        match info.OutputType with
        | Some outTy -> [("outputs", outTy)]
        | None -> [("result", info.StateType)]

    MLIROp.HWOp (HWOp.HWModule (info.ModuleName, inputs, outputs, bodyOps))

// ═══════════════════════════════════════════════════════════
// FLAT-PORT MEALY MACHINE BUILDER (Top-level FPGA module)
// ═══════════════════════════════════════════════════════════

/// The flat-port module's output extractions: the walk outputExtractions enumerates, each
/// emitted with its value from the layout; the pinned ones become the flat output ports.
let private flattenOutputStruct
    (info: MealyMachineInfo)
    (pinAttrs: Map<string, string list>)
    (outputSSA: SSA)
    (outputType: MLIRType)
    (layout: HardwareModuleLayout)
    : (string * SSA * MLIRType) list * MLIROp list =
    let steps = outputExtractions pinAttrs outputType
    if steps.Length <> layout.OutputFlatten.Length then
        failwithf "HardwareModulePatterns: %s flattens %d output extractions (%A) but %d values were derived"
            info.ModuleName steps.Length (steps |> List.map (fun st -> st.Field)) layout.OutputFlatten.Length
    let ssas = List.toArray layout.OutputFlatten
    let ops =
        steps |> List.mapi (fun i step ->
            let parentSSA = match step.Parent with None -> outputSSA | Some j -> ssas.[j]
            MLIROp.HWOp (HWOp.HWStructExtract (ssas.[i], parentSSA, step.Field, step.ParentType)))
    let pins =
        steps |> List.mapi (fun i step -> (i, step))
        |> List.choose (fun (i, step) -> step.Pin |> Option.map (fun pin -> (pin, ssas.[i], step.FieldType)))
    (pins, ops)

/// Build a flat-port Mealy machine hw.module for the FPGA top-level.
/// Observes PlatformPinMapping coeffect to expand struct-typed input/output ports
/// into individual ports matching physical pin logical names.
///
/// The step function's inner hw.module keeps struct ports — only the top-level
/// module gets flat ports. This is the residual of observing pin coeffects.
let buildFlatPortMealyModule
    (info: MealyMachineInfo)
    (pinMapping: PlatformPinMapping)
    (pinAttrs: Map<string, string list>)
    (layout: HardwareModuleLayout)
    : MLIROp =

    // ── Flat input ports ──
    // Walk input struct fields, map each to its pin logical name via FieldPinAttributes
    let flatInputPorts =
        match info.InputType with
        | Some (TStruct fields) ->
            fields |> List.collect (fun (fieldName, fieldTy) ->
                match Map.tryFind fieldName pinAttrs with
                | Some [pinName] -> [(pinName, fieldTy)]
                | Some pinNames ->
                    // Multi-pin input field
                    match fieldTy with
                    | TStruct tupleFields ->
                        List.zip pinNames tupleFields
                        |> List.map (fun (pn, (_, eTy)) -> (pn, eTy))
                    | _ -> [((List.head pinNames), fieldTy)]
                | None ->
                    // No pin attr — use field name as-is (shouldn't happen for pin-mapped types)
                    [(fieldName, fieldTy)])
        | _ -> []

    // ── Reset infrastructure ──
    // External: rst is a top-level port (Arg 1). Flat inputs start at Arg 2.
    // Internal POR: no rst port; the power-on reset circuit's values are the layout's.
    // Flat inputs start at Arg 1.
    let resetIsExternal =
        match pinMapping.Reset with
        | Some r -> r.IsExternal
        | None -> false

    let clkSSA = SSA.Arg 0
    let inputArgBase = if resetIsExternal then 2 else 1

    // For internal POR: a 1-bit register that starts at 0 (Xilinx INIT default) and
    // transitions to 1 on the first clock edge. Reset active = NOT por_reg = por_reg XOR 1.
    let porOps, rstSSA =
        match resetIsExternal, layout.PowerOnReset with
        | true, None -> ([], SSA.Arg 1)
        | false, Some (porOneSSA, porRegSSA, porRstSSA) ->
            ([ MLIROp.ArithOp (ArithOp.ConstI (porOneSSA, 1L, TInt (IntWidth 1)))
               MLIROp.SeqOp (SeqOp.SeqCompreg (porRegSSA, porOneSSA, clkSSA, None, TInt (IntWidth 1)))
               MLIROp.CombOp (CombOp.CombXor (porRstSSA, porRegSSA, porOneSSA, TInt (IntWidth 1))) ], porRstSSA)
        | true, Some _ -> failwithf "HardwareModulePatterns: %s has an external reset but power-on reset values were derived" info.ModuleName
        | false, None -> failwithf "HardwareModulePatterns: %s has an internal reset but no power-on reset values were derived" info.ModuleName

    // ── Pack flat input ports → input struct ──
    // A multi-pin tuple field takes its element args and packs them with its value from the
    // layout (InputPacks, in field order); every other field is one arg.
    let inputPackOps, inputStructSSA =
        match info.InputType with
        | Some (TStruct fields as inputType) ->
            let structSSA =
                match layout.InputStruct with
                | Some s -> s
                | None -> failwithf "HardwareModulePatterns: %s has a record input but no input-struct value was derived" info.ModuleName
            let (_, remainingPacks, fieldSSAsRev, packOpsRev) =
                fields |> List.fold (fun (argIdx, packs, fieldSSAs, packOps) (fieldName, fieldTy) ->
                    match Map.tryFind fieldName pinAttrs, fieldTy with
                    | Some pinNames, TStruct tupleFields when pinNames.Length > 1 ->
                        let elemSSAs = tupleFields |> List.mapi (fun k (_, eTy) -> (SSA.Arg (argIdx + k), eTy))
                        match packs with
                        | tupleSSA :: rest ->
                            (argIdx + tupleFields.Length, rest, (tupleSSA, fieldTy) :: fieldSSAs,
                             MLIROp.HWOp (HWOp.HWStructCreate (tupleSSA, elemSSAs, fieldTy)) :: packOps)
                        | [] -> failwithf "HardwareModulePatterns: %s multi-pin input field '%s' has no pack value derived" info.ModuleName fieldName
                    | _ -> (argIdx + 1, packs, (SSA.Arg argIdx, fieldTy) :: fieldSSAs, packOps))
                    (inputArgBase, layout.InputPacks, [], [])
            if not remainingPacks.IsEmpty then
                failwithf "HardwareModulePatterns: %s derived %d input pack values beyond its multi-pin fields" info.ModuleName remainingPacks.Length
            let createOp = MLIROp.HWOp (HWOp.HWStructCreate (structSSA, List.rev fieldSSAsRev, inputType))
            (List.rev packOpsRev @ [createOp], structSSA)
        | _ -> ([], SSA.Arg inputArgBase)

    // ── Instantiate the step function ──
    let stepInputs =
        match info.InputType with
        | Some iTy -> [("state", layout.State, info.StateType); ("inputs", inputStructSSA, iTy)]
        | _ -> [("s", layout.State, info.StateType)]
    let instanceOp =
        MLIROp.HWOp (HWOp.HWInstance (layout.Instance, "step_inst", info.StepFunctionName, stepInputs, [("result", stepResultTypeOf info)]))

    // ── Decompose the step result ──
    let nextStateSSA, outputStructSSA, decomposeOps = stepResult info layout

    // ── Flatten the output struct → individual pin signals ──
    let flatOutputPins, flattenOps =
        match outputStructSSA, info.OutputType with
        | Some oSSA, Some outTy -> flattenOutputStruct info pinAttrs oSSA outTy layout
        | _ -> ([], [])

    // ── hw.output (flat pin signals) ──
    let outputOp =
        if flatOutputPins.IsEmpty then
            match outputStructSSA, info.OutputType with
            | Some oSSA, Some outTy -> MLIROp.HWOp (HWOp.HWOutput [(oSSA, outTy)])
            | _ -> MLIROp.HWOp (HWOp.HWOutput [(layout.State, info.StateType)])
        else
            MLIROp.HWOp (HWOp.HWOutput (flatOutputPins |> List.map (fun (_, ssa, ty) -> (ssa, ty))))

    // In the layout's order: power-on reset, reset constants, registers, input packs and
    // struct, state, instance, step result, output flatten, next fields
    let bodyOps =
        porOps
        @ resetOps info layout
        @ registerOps info layout clkSSA rstSSA
        @ inputPackOps
        @ [stateOp info layout; instanceOp]
        @ decomposeOps
        @ flattenOps
        @ nextFieldOps info layout nextStateSSA
        @ [outputOp]

    let rstPortName =
        match pinMapping.Reset with
        | Some r -> r.PortName
        | None -> "rst"

    let inputs =
        [(pinMapping.Clock.PortName, TSeqClock)]
        @ (if resetIsExternal then [(rstPortName, TInt (IntWidth 1))] else [])
        @ flatInputPorts
    let outputs =
        if flatOutputPins.IsEmpty then
            match info.OutputType with
            | Some outTy -> [("outputs", outTy)]
            | None -> [("result", info.StateType)]
        else
            flatOutputPins |> List.map (fun (name, _, ty) -> (name, ty))

    MLIROp.HWOp (HWOp.HWModule (info.ModuleName, inputs, outputs, bodyOps))
