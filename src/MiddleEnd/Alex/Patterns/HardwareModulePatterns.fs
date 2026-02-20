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

/// Build the Mealy machine hw.module from extracted Design info.
/// Returns the complete hw.module op.
///
/// Models the full Mealy machine: (State × Input) → (State × Output)
///   - State feeds back through seq.compreg registers
///   - Input comes from top-level input ports (when InputType is Some)
///   - Output goes to top-level output ports (when OutputType is Some)
///   - Step function is instantiated via hw.instance
///
/// The body uses programmatic SSA allocation (V 0, V 1, ...) since
/// the hw.module body is entirely synthesized — not from PSG node traversal.
let buildMealyMachineModule (info: MealyMachineInfo) : MLIROp =
    let n = info.StateFields.Length
    let hasInputs = info.InputType.IsSome
    let hasOutputs = info.OutputType.IsSome
    let mutable ssaCounter = 0
    let nextSSA () =
        let ssa = V ssaCounter
        ssaCounter <- ssaCounter + 1
        ssa

    // Input port SSAs: clk (Arg 0), rst (Arg 1), inputs (Arg 2 if present)
    let clkSSA = SSA.Arg 0
    let rstSSA = SSA.Arg 1
    let inputsSSA = if hasInputs then Some (SSA.Arg 2) else None

    // ── Phase 1: Reset value constants ──
    let resetOps, resetSSAs =
        info.StateFields
        |> List.map (fun (_, fieldTy, resetLit) ->
            let ssa = nextSSA()
            let op = MLIROp.ArithOp (ArithOp.ConstI (ssa, literalToInt64 resetLit, fieldTy))
            (op, ssa))
        |> List.unzip

    // ── Phase 2: State registers (seq.compreg) ──
    // Forward-declare next-state SSAs (will be assigned after step instance)
    // In CIRCT hw.module, ops use graph semantics — forward references are valid
    //
    // SSA layout after registers:
    //   stateSSA, instanceSSA, [stateExtractSSAs...], [outputExtractSSA if needed]
    let postRegOpsCount =
        1  // struct_create (state)
        + 1  // hw.instance
        + (if hasOutputs then 1 else 0)  // struct_extract for state from result tuple
        + n  // struct_extract per state field
    // nextStateBaseSSA: where the state field extracts start (for register feedback)
    let nextStateBaseSSA = ssaCounter + n + 1 + 1 + (if hasOutputs then 2 else 0)  // skip regs + struct_create + instance + optional (Item1 + Item2) extracts
    let regOps, regSSAs =
        info.StateFields
        |> List.mapi (fun i (_, fieldTy, _) ->
            let regSSA = nextSSA()
            let nextSSARef = V (nextStateBaseSSA + i)  // Forward ref to struct_extract of next-state field
            let op = MLIROp.SeqOp (SeqOp.SeqCompreg (regSSA, nextSSARef, clkSSA, Some (rstSSA, resetSSAs.[i]), fieldTy))
            (op, regSSA))
        |> List.unzip

    // ── Phase 3: Build current state struct from register outputs ──
    let stateSSA = nextSSA()
    let stateFieldVals = List.zip regSSAs (info.StateFields |> List.map (fun (_, ty, _) -> ty))
    let structCreateOp = MLIROp.HWOp (HWOp.HWStructCreate (stateSSA, stateFieldVals, info.StateType))

    // ── Phase 4: Instantiate step function ──
    let instanceSSA = nextSSA()
    let stepInputs =
        match inputsSSA, info.InputType with
        | Some iSSA, Some iTy -> [("state", stateSSA, info.StateType); ("inputs", iSSA, iTy)]
        | _ -> [("s", stateSSA, info.StateType)]

    // Step function return type: (State × Output) if OutputType present, else just State
    let stepResultType =
        match info.OutputType with
        | Some outTy -> TStruct [("Item1", info.StateType); ("Item2", outTy)]
        | None -> info.StateType
    let instanceOp = MLIROp.HWOp (HWOp.HWInstance (
        instanceSSA,
        "step_inst",
        info.StepFunctionName,
        stepInputs,
        [("result", stepResultType)]
    ))

    // ── Phase 5: Decompose step result ──
    // If OutputType present: result is (State, Output) struct — extract both
    // If no OutputType: result IS the next state directly
    let nextStateSSA, outputSSA, decomposeOps =
        match info.OutputType with
        | Some outTy ->
            let nsSA = nextSSA()
            let nsOp = MLIROp.HWOp (HWOp.HWStructExtract (nsSA, instanceSSA, "Item1", stepResultType))
            let oSSA = nextSSA()
            let oOp = MLIROp.HWOp (HWOp.HWStructExtract (oSSA, instanceSSA, "Item2", stepResultType))
            (nsSA, Some oSSA, [nsOp; oOp])
        | None ->
            (instanceSSA, None, [])

    // ── Phase 6: Extract next-state fields for register feedback ──
    let extractOps =
        info.StateFields
        |> List.map (fun (fieldName, _, _) ->
            let extractSSA = nextSSA()
            MLIROp.HWOp (HWOp.HWStructExtract (extractSSA, nextStateSSA, fieldName, info.StateType)))

    // ── Phase 7: hw.output ──
    let outputOp =
        match outputSSA, info.OutputType with
        | Some oSSA, Some outTy -> MLIROp.HWOp (HWOp.HWOutput [(oSSA, outTy)])
        | _ -> MLIROp.HWOp (HWOp.HWOutput [(stateSSA, info.StateType)])

    // ── Assemble hw.module ──
    let bodyOps =
        resetOps
        @ regOps
        @ [structCreateOp]
        @ [instanceOp]
        @ decomposeOps
        @ extractOps
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

/// Recursively extract pin-mapped signals from an output struct.
/// For each leaf field with a pin attribute, emits hw.struct_extract ops
/// and collects (sanitizedPinName, SSA, type) for flat output ports.
/// Fields without pin attributes are skipped (synthesis will optimize them away).
let private flattenOutputStruct
    (pinAttrs: Map<string, string list>)
    (parentSSA: SSA)
    (parentType: MLIRType)
    (nextSSA: unit -> SSA)
    : (string * SSA * MLIRType) list * MLIROp list =
    let mutable pins = []
    let mutable ops = []

    let rec walk (pSSA: SSA) (pType: MLIRType) =
        match pType with
        | TStruct fields ->
            for (fieldName, fieldTy) in fields do
                match Map.tryFind fieldName pinAttrs with
                | Some pinNames ->
                    let extractSSA = nextSSA()
                    ops <- ops @ [MLIROp.HWOp (HWOp.HWStructExtract (extractSSA, pSSA, fieldName, pType))]
                    match pinNames with
                    | [single] ->
                        pins <- pins @ [(single, extractSSA, fieldTy)]
                    | multiple ->
                        // Multi-pin field (e.g., [<Pins("r","g","b")>]): extract tuple elements
                        match fieldTy with
                        | TStruct tupleFields ->
                            for (pinName, (elemField, elemTy)) in List.zip multiple tupleFields do
                                let elemSSA = nextSSA()
                                ops <- ops @ [MLIROp.HWOp (HWOp.HWStructExtract (elemSSA, extractSSA, elemField, fieldTy))]
                                pins <- pins @ [(pinName, elemSSA, elemTy)]
                        | _ ->
                            // Single value assigned to first pin name
                            pins <- pins @ [((List.head multiple), extractSSA, fieldTy)]
                | None ->
                    // No direct pin attr — recurse into sub-structs that may contain pins
                    match fieldTy with
                    | TStruct _ ->
                        let extractSSA = nextSSA()
                        ops <- ops @ [MLIROp.HWOp (HWOp.HWStructExtract (extractSSA, pSSA, fieldName, pType))]
                        walk extractSSA fieldTy
                    | _ -> ()
        | _ -> ()

    walk parentSSA parentType
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
    : MLIROp =

    let n = info.StateFields.Length
    let hasOutputs = info.OutputType.IsSome
    let mutable ssaCounter = 0
    let nextSSA () =
        let ssa = V ssaCounter
        ssaCounter <- ssaCounter + 1
        ssa

    // ── Compute flat input ports ──
    // Walk input struct fields, map each to its pin logical name via FieldPinAttributes
    let flatInputPorts, inputArgCount =
        match info.InputType with
        | Some (TStruct fields) ->
            let ports =
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
            (ports, ports.Length)
        | _ -> ([], 0)

    // ── Reset infrastructure ──
    // External: rst is a top-level port (Arg 1). Flat inputs start at Arg 2.
    // Internal POR: no rst port. Generate POR circuit. Flat inputs start at Arg 1.
    let resetIsExternal =
        match pinMapping.Reset with
        | Some r -> r.IsExternal
        | None -> false

    let clkSSA = SSA.Arg 0
    let inputArgBase = if resetIsExternal then 2 else 1

    // For internal POR: generate a 1-bit register that starts at 0 (Xilinx INIT default),
    // transitions to 1 on first clock edge. Reset active = NOT por_reg = por_reg XOR 1.
    // 3 ops: arith.constant 1, seq.compreg, comb.xor
    let porOps, rstSSA =
        if resetIsExternal then
            ([], SSA.Arg 1)
        else
            let porOneSSA = nextSSA()
            let porOneOp = MLIROp.ArithOp (ArithOp.ConstI (porOneSSA, 1L, TInt (IntWidth 1)))
            let porRegSSA = nextSSA()
            let porRegOp = MLIROp.SeqOp (SeqOp.SeqCompreg (porRegSSA, porOneSSA, clkSSA, None, TInt (IntWidth 1)))
            let porRstSSA = nextSSA()
            let porRstOp = MLIROp.CombOp (CombOp.CombXor (porRstSSA, porRegSSA, porOneSSA, TInt (IntWidth 1)))
            ([porOneOp; porRegOp; porRstOp], porRstSSA)

    // ── Phase 1: Reset value constants ──
    let resetOps, resetSSAs =
        info.StateFields
        |> List.map (fun (_, fieldTy, resetLit) ->
            let ssa = nextSSA()
            let op = MLIROp.ArithOp (ArithOp.ConstI (ssa, literalToInt64 resetLit, fieldTy))
            (op, ssa))
        |> List.unzip

    // ── Phase 2: State registers ──
    // Forward-reference calculation must account for flat input packing ops
    let inputPackOpsCount =
        match info.InputType with
        | Some (TStruct fields) ->
            // Multi-pin fields need a struct_create for the tuple, plus one for the outer struct
            let multiPinPacks =
                fields |> List.sumBy (fun (fieldName, _) ->
                    match Map.tryFind fieldName pinAttrs with
                    | Some pins when pins.Length > 1 -> 1  // struct_create for tuple
                    | _ -> 0)
            1 + multiPinPacks  // 1 for the input struct + multi-pin tuple packs
        | _ -> 0

    // nextStateBaseSSA: after regs + pack ops + struct_create(state) + instance + decompose + output extract
    let nextStateBaseSSA =
        ssaCounter + n  // skip registers
        + inputPackOpsCount  // input packing
        + 1  // struct_create (state)
        + 1  // hw.instance
        + (if hasOutputs then 2 else 0)  // Item1 + Item2 extract
        // Then: output flatten ops (variable count) — we handle this with a post-fixup

    // We need to know the exact SSA offset for register feedback.
    // Problem: output flatten op count is variable (depends on type structure).
    // Solution: Use two-pass — first compute output flatten count, then build.

    // Count output flatten ops
    let outputFlattenOpCount =
        match info.OutputType with
        | Some outTy ->
            // Count ops that flattenOutputStruct would emit
            let rec countOps (ty: MLIRType) =
                match ty with
                | TStruct fields ->
                    fields |> List.sumBy (fun (fieldName, fieldTy) ->
                        match Map.tryFind fieldName pinAttrs with
                        | Some pinNames ->
                            1 + (if pinNames.Length > 1 then
                                    match fieldTy with TStruct tf -> tf.Length | _ -> 0
                                 else 0)
                        | None ->
                            match fieldTy with
                            | TStruct _ -> 1 + countOps fieldTy
                            | _ -> 0)
                | _ -> 0
            countOps outTy
        | None -> 0

    let actualNextStateBase =
        ssaCounter + n + inputPackOpsCount + 1 + 1
        + (if hasOutputs then 2 else 0)
        + outputFlattenOpCount

    let regOps, regSSAs =
        info.StateFields
        |> List.mapi (fun i (_, fieldTy, _) ->
            let regSSA = nextSSA()
            let nextSSARef = V (actualNextStateBase + i)
            let op = MLIROp.SeqOp (SeqOp.SeqCompreg (regSSA, nextSSARef, clkSSA, Some (rstSSA, resetSSAs.[i]), fieldTy))
            (op, regSSA))
        |> List.unzip

    // ── Phase 3: Pack flat input ports → input struct ──
    let inputPackOps, inputStructSSA =
        match info.InputType with
        | Some (TStruct fields as inputType) ->
            let mutable argIdx = inputArgBase  // after clk (+ rst if external)
            let mutable packOps = []
            let fieldSSAs =
                fields |> List.map (fun (fieldName, fieldTy) ->
                    match Map.tryFind fieldName pinAttrs with
                    | Some [_] ->
                        let ssa = SSA.Arg argIdx
                        argIdx <- argIdx + 1
                        (ssa, fieldTy)
                    | Some pinNames when pinNames.Length > 1 ->
                        // Multi-pin: collect individual args, pack into tuple struct
                        match fieldTy with
                        | TStruct tupleFields ->
                            let elemSSAs =
                                tupleFields |> List.map (fun (_, eTy) ->
                                    let ssa = SSA.Arg argIdx
                                    argIdx <- argIdx + 1
                                    (ssa, eTy))
                            let tupleSSA = nextSSA()
                            packOps <- packOps @ [MLIROp.HWOp (HWOp.HWStructCreate (tupleSSA, elemSSAs, fieldTy))]
                            (tupleSSA, fieldTy)
                        | _ ->
                            let ssa = SSA.Arg argIdx
                            argIdx <- argIdx + 1
                            (ssa, fieldTy)
                    | _ ->
                        let ssa = SSA.Arg argIdx
                        argIdx <- argIdx + 1
                        (ssa, fieldTy))
            let structSSA = nextSSA()
            let createOp = MLIROp.HWOp (HWOp.HWStructCreate (structSSA, fieldSSAs, inputType))
            (packOps @ [createOp], structSSA)
        | _ -> ([], SSA.Arg inputArgBase)

    // ── Phase 4: Build current state struct ──
    let stateSSA = nextSSA()
    let stateFieldVals = List.zip regSSAs (info.StateFields |> List.map (fun (_, ty, _) -> ty))
    let structCreateOp = MLIROp.HWOp (HWOp.HWStructCreate (stateSSA, stateFieldVals, info.StateType))

    // ── Phase 5: Instantiate step function ──
    let instanceSSA = nextSSA()
    let stepInputs =
        match info.InputType with
        | Some iTy -> [("state", stateSSA, info.StateType); ("inputs", inputStructSSA, iTy)]
        | _ -> [("s", stateSSA, info.StateType)]

    let stepResultType =
        match info.OutputType with
        | Some outTy -> TStruct [("Item1", info.StateType); ("Item2", outTy)]
        | None -> info.StateType
    let instanceOp = MLIROp.HWOp (HWOp.HWInstance (
        instanceSSA,
        "step_inst",
        info.StepFunctionName,
        stepInputs,
        [("result", stepResultType)]
    ))

    // ── Phase 6: Decompose step result ──
    let nextStateSSA, outputStructSSA, decomposeOps =
        match info.OutputType with
        | Some outTy ->
            let nsSA = nextSSA()
            let nsOp = MLIROp.HWOp (HWOp.HWStructExtract (nsSA, instanceSSA, "Item1", stepResultType))
            let oSSA = nextSSA()
            let oOp = MLIROp.HWOp (HWOp.HWStructExtract (oSSA, instanceSSA, "Item2", stepResultType))
            (nsSA, Some oSSA, [nsOp; oOp])
        | None ->
            (instanceSSA, None, [])

    // ── Phase 7: Flatten output struct → individual pin signals ──
    let flatOutputPins, flattenOps =
        match outputStructSSA, info.OutputType with
        | Some oSSA, Some outTy ->
            flattenOutputStruct pinAttrs oSSA outTy nextSSA
        | _ -> ([], [])

    // ── Phase 8: Extract next-state fields for register feedback ──
    let extractOps =
        info.StateFields
        |> List.map (fun (fieldName, _, _) ->
            let extractSSA = nextSSA()
            MLIROp.HWOp (HWOp.HWStructExtract (extractSSA, nextStateSSA, fieldName, info.StateType)))

    // ── Phase 9: hw.output (flat pin signals) ──
    let outputOp =
        if flatOutputPins.IsEmpty then
            match outputStructSSA, info.OutputType with
            | Some oSSA, Some outTy -> MLIROp.HWOp (HWOp.HWOutput [(oSSA, outTy)])
            | _ -> MLIROp.HWOp (HWOp.HWOutput [(stateSSA, info.StateType)])
        else
            MLIROp.HWOp (HWOp.HWOutput (flatOutputPins |> List.map (fun (_, ssa, ty) -> (ssa, ty))))

    // ── Assemble hw.module ──
    let bodyOps =
        porOps
        @ resetOps
        @ regOps
        @ inputPackOps
        @ [structCreateOp]
        @ [instanceOp]
        @ decomposeOps
        @ flattenOps
        @ extractOps
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
