/// HardwareModulePatterns - Mealy machine extraction for FPGA [<HardwareModule>] bindings
///
/// Builds hw.module with seq.compreg registers + hw.instance of the step function.
/// The step function is already emitted as a separate hw.module by LambdaWitness.
///
/// Design<'State, 'Report> record is compile-time metadata:
///   InitialState → register reset values (seq.compreg reset argument)
///   Step         → VarRef to function → hw.instance instantiation
///   Clock        → clock port (in %clk: i1)
///
/// Target MLIR:
///   hw.module @name(in %clk: i1, out result: <stateType>) {
///     %reg0 = seq.compreg %next0, %clk reset %init0 : type
///     ...
///     %state = hw.struct_create (%reg0, ...) : <stateType>
///     %next_state = hw.instance "step" @step_func(%state) -> (result: <stateType>)
///     %next0 = hw.struct_extract %next_state["field0"] : <stateType>
///     ...
///     hw.output %state : <stateType>
///   }
module Alex.Patterns.HardwareModulePatterns

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// PSG METADATA EXTRACTION (compile-time, no MLIR emission)
// ═══════════════════════════════════════════════════════════

/// Extract literal integer value from a PSG Literal node (for register reset values)
let private extractLiteralInt (graph: SemanticGraph) (nodeId: NodeId) : int64 option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Literal (NativeLiteral.Int (value, _)) -> Some value
        | SemanticKind.Literal (NativeLiteral.Bool true) -> Some 1L
        | SemanticKind.Literal (NativeLiteral.Bool false) -> Some 0L
        | _ -> None
    | None -> None

/// Extract field values from a RecordExpr as (fieldName, literalValue) pairs
/// Returns None for fields with non-literal initial values
let private extractRecordLiterals (graph: SemanticGraph) (fields: (string * NodeId) list) : (string * int64) list option =
    let results =
        fields |> List.map (fun (name, nodeId) ->
            match extractLiteralInt graph nodeId with
            | Some v -> Some (name, v)
            | None -> None)
    if results |> List.forall Option.isSome then
        Some (results |> List.map Option.get)
    else
        None

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

/// Information extracted from a Design<S,R> record
type MealyMachineInfo = {
    /// Qualified name for the hw.module (e.g., "HelloFPGA.counter")
    ModuleName: string
    /// Qualified name of the step function (e.g., "HelloFPGA.step")
    StepFunctionName: string
    /// State type as TStruct
    StateType: MLIRType
    /// State field info: (fieldName, mlirType, resetValue)
    StateFields: (string * MLIRType * int64) list
}

/// Build the Mealy machine hw.module from extracted Design info.
/// Returns the complete hw.module op.
///
/// The body uses programmatic SSA allocation (V 0, V 1, ...) since
/// the hw.module body is entirely synthesized — not from PSG node traversal.
let buildMealyMachineModule (info: MealyMachineInfo) : MLIROp =
    let n = info.StateFields.Length
    let mutable ssaCounter = 0
    let nextSSA () =
        let ssa = V ssaCounter
        ssaCounter <- ssaCounter + 1
        ssa

    // Clock port SSA (first input), reset port SSA (second input)
    let clkSSA = SSA.Arg 0
    let rstSSA = SSA.Arg 1

    // ── Phase 1: Reset value constants ──
    let resetOps, resetSSAs =
        info.StateFields
        |> List.map (fun (_, fieldTy, resetVal) ->
            let ssa = nextSSA()
            let op = MLIROp.ArithOp (ArithOp.ConstI (ssa, resetVal, fieldTy))
            (op, ssa))
        |> List.unzip

    // ── Phase 2: State registers (seq.compreg) ──
    // Forward-declare next-state SSAs (will be assigned in Phase 4)
    // In CIRCT hw.module, ops use graph semantics — forward references are valid
    let nextStateBaseSSA = ssaCounter + n  // Skip past register SSAs
    let regOps, regSSAs =
        info.StateFields
        |> List.mapi (fun i (_, fieldTy, _) ->
            let regSSA = nextSSA()
            let nextSSARef = V (nextStateBaseSSA + 2 + i)  // Forward ref to struct_extract results (skip struct_create + instance)
            let op = MLIROp.SeqOp (SeqOp.SeqCompreg (regSSA, nextSSARef, clkSSA, Some (rstSSA, resetSSAs.[i]), fieldTy))
            (op, regSSA))
        |> List.unzip

    // ── Phase 3: Build current state struct from register outputs ──
    let stateSSA = nextSSA()
    let stateFieldVals = List.zip regSSAs (info.StateFields |> List.map (fun (_, ty, _) -> ty))
    let structCreateOp = MLIROp.HWOp (HWOp.HWStructCreate (stateSSA, stateFieldVals, info.StateType))

    // ── Phase 4: Instantiate step function ──
    let nextStateSSA = nextSSA()
    let instanceOp = MLIROp.HWOp (HWOp.HWInstance (
        nextStateSSA,
        "step_inst",
        info.StepFunctionName,
        // Step function input: the current state struct
        [("s", stateSSA, info.StateType)],
        // Step function output: the next state struct
        [("result", info.StateType)]
    ))

    // ── Phase 5: Extract next-state fields for register feedback ──
    let extractOps =
        info.StateFields
        |> List.map (fun (fieldName, _, _) ->
            let extractSSA = nextSSA()
            MLIROp.HWOp (HWOp.HWStructExtract (extractSSA, nextStateSSA, fieldName, info.StateType)))

    // ── Phase 6: hw.output (expose current state as output) ──
    let outputOp = MLIROp.HWOp (HWOp.HWOutput [(stateSSA, info.StateType)])

    // ── Assemble hw.module ──
    let bodyOps =
        resetOps
        @ regOps
        @ [structCreateOp]
        @ [instanceOp]
        @ extractOps
        @ [outputOp]

    let inputs = [("clk", TSeqClock); ("rst", TInt I1)]
    let outputs = [("result", info.StateType)]

    MLIROp.HWOp (HWOp.HWModule (info.ModuleName, inputs, outputs, bodyOps))
