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
/// Target MLIR (full Mealy):
///   hw.module @name(in %clk: !seq.clock, in %rst: i1, in %inputs: <inputType>,
///                   out outputs: <outputType>) {
///     %init0 = arith.constant <resetVal> : <fieldType>
///     %reg0 = seq.compreg %next0, %clk reset %rst, %init0 : <fieldType>
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
        [("clk", TSeqClock); ("rst", TInt I1)]
        @ (match info.InputType with Some iTy -> [("inputs", iTy)] | None -> [])
    let outputs =
        match info.OutputType with
        | Some outTy -> [("outputs", outTy)]
        | None -> [("result", info.StateType)]

    MLIROp.HWOp (HWOp.HWModule (info.ModuleName, inputs, outputs, bodyOps))
