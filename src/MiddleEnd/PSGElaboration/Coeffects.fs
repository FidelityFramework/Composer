/// Coeffects - Pre-computed analysis results consumed by witnesses
///
/// Coeffects are the outputs of PSGElaboration passes. They represent
/// information computed BEFORE emission that witnesses OBSERVE (not compute).
///
/// This follows the "mise-en-place" principle: all prep work is done
/// before emission begins. Witnesses only look up pre-computed values.
///
/// Current coeffects:
/// - NodeSSAAllocation: Which SSAs are assigned to each PSG node
/// - ClosureLayout: How closures are structured (captures, SSAs, types)
///
/// Future coeffects (when SeqMoveNext moves to CCS):
/// - SeqMoveNextLayout: State machine structure for sequence MoveNext
module PSGElaboration.Coeffects

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// SSA ASSIGNMENT COEFFECT
// ═══════════════════════════════════════════════════════════════════════════

/// SSA allocation for a PSG node - supports multi-SSA expansion
/// One PSG node may expand to multiple MLIR ops, each needing an SSA.
/// SSAs are in emission order; Result is the final SSA (what gets used downstream).
type NodeSSAAllocation = {
    /// All SSAs for this node in emission order
    SSAs: SSA list
    /// The result SSA (always the last one)
    Result: SSA
}

module NodeSSAAllocation =
    let single (ssa: SSA) = { SSAs = [ssa]; Result = ssa }
    let multi (ssas: SSA list) =
        match ssas with
        | [] -> failwith "NodeSSAAllocation requires at least one SSA"
        | _ -> { SSAs = ssas; Result = List.last ssas }

// ═══════════════════════════════════════════════════════════════════════════
// CLOSURE LAYOUT COEFFECT
// ═══════════════════════════════════════════════════════════════════════════
//
// For Lambdas with captures, we pre-compute the complete closure layout.
// This is deterministic - derived from CaptureInfo list in PSG (from CCS).
// Witnesses observe this coeffect; they do NOT compute layout during emission.

/// How a variable is captured in a closure
type CaptureMode =
    | ByValue  // Immutable variable: copy value into closure struct
    | ByRef    // Mutable variable: store pointer to alloca in closure struct

/// Layout information for a single captured variable
type CaptureSlot = {
    /// Name of the captured variable
    Name: string
    /// Index in the closure struct (0 = code_ptr, 1+ = captures)
    SlotIndex: int
    /// MLIR type of the slot (value type for ByValue, ptr for ByRef)
    SlotType: MLIRType
    /// Source NodeId of the captured binding (for SSA lookup)
    SourceNodeId: NodeId option
    /// How the variable is captured
    Mode: CaptureMode
    /// The slot holds the base index of a memref value (a record, an array, a mutable cell):
    /// construction extracts the base pointer first, with its own value from CaptureInsertSSAs
    ExtractsBasePointer: bool
}

/// Complete closure layout for a Lambda with captures.
/// This coeffect tells LambdaWitness exactly how to construct and extract closures.
type ClosureLayout = {
    /// The Lambda node this layout is for
    LambdaNodeId: NodeId
    /// Ordered list of capture slots (matches closure struct field order)
    Captures: CaptureSlot list

    // ─────────────────────────────────────────────────────────────────────────
    // FLAT STRUCT CONSTRUCTION SSAs
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for addressof code_ptr
    CodeAddrSSA: SSA
    /// SSA for undef closure struct
    ClosureUndefSSA: SSA
    /// SSA for insertvalue of code_ptr at [0]
    ClosureWithCodeSSA: SSA
    /// SSAs for insertvalue of each capture at [1..N] (one per capture)
    CaptureInsertSSAs: SSA list

    // ─────────────────────────────────────────────────────────────────────────
    // HEAP ALLOCATION SSAs (for escaping closures)
    // ─────────────────────────────────────────────────────────────────────────
    /// SSAs for heap arena allocation (5 SSAs)
    HeapPosPtrSSA: SSA
    HeapPosSSA: SSA
    HeapBaseSSA: SSA
    HeapResultPtrSSA: SSA
    HeapNewPosSSA: SSA

    /// SSAs for size computation (3 SSAs - compile-time size, no null GEP trick)
    /// Size is computed from ClosureStructType at compile time
    SizeGepSSA: SSA
    SizeSSA: SSA
    SizeOneSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // UNIFORM PAIR CONSTRUCTION SSAs
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for undef uniform pair {ptr, ptr}
    PairUndefSSA: SSA
    /// SSA for insertvalue code_ptr at [0]
    PairWithCodeSSA: SSA
    /// SSA for final closure result (insertvalue env_ptr at [1])
    ClosureResultSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // CAPTURE EXTRACTION SSAs (for callee/inner function)
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for loading the closure struct from env_ptr (Arg 0) in inner function
    StructLoadSSA: SSA
    /// The callee prologue's values per capture, in the order pExtractCaptures consumes them:
    /// the capture's work values (view and zero; a decomposed memref's seven) then its result
    CaptureExtractionSSAs: SSA list list
    /// The callee prologue's env reconstruction: the memref view of Arg 0, then its static cast
    EnvReconstructionSSAs: SSA * SSA

    // ─────────────────────────────────────────────────────────────────────────
    // TYPE INFORMATION
    // ─────────────────────────────────────────────────────────────────────────
    /// MLIR type of the environment struct (kept for compatibility)
    EnvStructType: MLIRType
    /// MLIR type of the closure struct: {ptr, T0, T1, ...} = {code_ptr, captures...}
    ClosureStructType: MLIRType
    /// Lambda context: determines extraction base index and load struct type
    Context: LambdaContext
    /// For LazyThunk: the full lazy struct type {i1, T, ptr, cap0, cap1, ...}
    LazyStructType: MLIRType option
}

/// Get the struct type to load when extracting captures from this closure
let closureLoadStructType (layout: ClosureLayout) : MLIRType =
    match layout.Context with
    | LambdaContext.RegularClosure -> layout.ClosureStructType
    | LambdaContext.LazyThunk ->
        match layout.LazyStructType with
        | Some lazyType -> lazyType
        | None -> failwith "LazyThunk context requires LazyStructType"
    | LambdaContext.SeqGenerator -> layout.ClosureStructType

/// Get the base index for capture extraction based on context
/// Regular closure: captures at indices 1, 2, ... (after code_ptr at [0])
/// Lazy thunk: captures at indices 3, 4, ... (after computed[0], value[1], code_ptr[2])
let closureExtractionBaseIndex (layout: ClosureLayout) : int =
    match layout.Context with
    | LambdaContext.RegularClosure -> 1
    | LambdaContext.LazyThunk -> 3
    | LambdaContext.SeqGenerator -> 3

// ═══════════════════════════════════════════════════════════════════════════
// DU LAYOUT COEFFECT
// ═══════════════════════════════════════════════════════════════════════════
//
// For heterogeneous DUs (like Result<'T, 'E>) that need arena allocation,
// we pre-compute the complete DU layout. This follows the flat closure model:
// build case-specific struct inline, then store to arena, return pointer.
//
// Homogeneous DUs (like Option<'T>) use inline struct representation and
// don't need a DULayout - they're handled directly by witnessDUConstruct.

/// Complete DU layout for a DUConstruct node that needs arena allocation.
/// This coeffect tells MemoryWitness exactly how to construct arena-allocated DUs.
type DULayout = {
    /// The DUConstruct node this layout is for
    DUConstructNodeId: NodeId
    /// Case name (e.g., "Ok", "Error")
    CaseName: string
    /// Case index (0 for first case, 1 for second, etc.)
    CaseIndex: int
    /// Whether this case has a payload
    HasPayload: bool

    // ─────────────────────────────────────────────────────────────────────────
    // CASE-SPECIFIC STRUCT CONSTRUCTION SSAs
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for undef case struct
    StructUndefSSA: SSA
    /// SSA for tag constant
    TagConstSSA: SSA
    /// SSA for insertvalue of tag at [0]
    WithTagSSA: SSA
    /// SSA for insertvalue of payload at [1] (only used if HasPayload)
    WithPayloadSSA: SSA option

    // ─────────────────────────────────────────────────────────────────────────
    // SIZE COMPUTATION SSAs (compile-time size, no null GEP trick)
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for constant 1 (if needed)
    SizeOneSSA: SSA
    /// SSA for GEP result (if needed)
    SizeGepSSA: SSA
    /// SSA for size in bytes (computed from CaseStructType at compile time)
    SizeSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // ARENA ALLOCATION SSAs (uses closure_heap arena)
    // ─────────────────────────────────────────────────────────────────────────
    /// SSA for addressof closure_pos
    HeapPosPtrSSA: SSA
    /// SSA for load current position
    HeapPosSSA: SSA
    /// SSA for addressof closure_heap
    HeapBaseSSA: SSA
    /// SSA for GEP heap_base + pos (result pointer)
    HeapResultPtrSSA: SSA
    /// SSA for pos + size (new position)
    HeapNewPosSSA: SSA

    // ─────────────────────────────────────────────────────────────────────────
    // TYPE INFORMATION
    // ─────────────────────────────────────────────────────────────────────────
    /// MLIR type of the case-specific struct: {i8, PayloadType}
    CaseStructType: MLIRType
    /// MLIR type of the payload (if HasPayload)
    PayloadType: MLIRType option
}

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM PIN MAPPING COEFFECT (FPGA targets only)
// ═══════════════════════════════════════════════════════════════════════════
//
// Pre-computed mapping from [<Pin>]/[<Pins>] attributes on record fields
// to physical PinEndpoint data from the platform bindings. Observed by:
//   1. HardwareModulePatterns — flat hw.module port generation
//   2. XDCTransfer — XDC constraint file generation
// Two observers, one truth, two residuals.

/// Single pin constraint for XDC generation
type PinConstraint = {
    /// hw.module port name = PinEndpoint.LogicalName (e.g., "sw[0]")
    PortName: string
    /// Xilinx package pin (e.g., "A8")
    PackagePin: string
    /// I/O standard (e.g., "LVCMOS33")
    IOStandard: string
    /// Pin direction
    Direction: string
}

/// Clock constraint for XDC generation
type ClockConstraint = {
    /// Port name (e.g., "sys_clk")
    PortName: string
    /// Package pin (e.g., "E3")
    PackagePin: string
    /// I/O standard
    IOStandard: string
    /// Frequency in Hz (e.g., 100_000_000L) — period = 1e9 / FrequencyHz
    FrequencyHz: int64
}

/// Reset constraint — platform infrastructure parallel to ClockConstraint.
/// External: physical pin on the board. Internal: compiler generates POR circuit.
type ResetConstraint = {
    /// Port name (e.g., "rst")
    PortName: string
    /// True = external pin (XDC constraint needed). False = internal POR (no port).
    IsExternal: bool
    /// Package pin (only meaningful when IsExternal = true)
    PackagePin: string
    /// I/O standard (only meaningful when IsExternal = true)
    IOStandard: string
    /// True = reset is active-high. False = active-low (inverter needed).
    ActiveHigh: bool
}

/// Complete pin mapping for a HardwareModule Design
type PlatformPinMapping = {
    /// All I/O pin constraints (inputs + outputs)
    Pins: PinConstraint list
    /// Clock constraint
    Clock: ClockConstraint
    /// Reset constraint (None → no reset endpoint declared, same as internal POR)
    Reset: ResetConstraint option
    /// Platform device string (e.g., "xc7a100tcsg324-1" for Xilinx)
    DevicePart: string
    /// Record field name → pin logical names (from [<Pin>]/[<Pins>] attributes)
    /// Used by HardwareModulePatterns for struct ↔ flat port mapping
    FieldPinAttrs: Map<string, string list>
}

// ═══════════════════════════════════════════════════════════════════════════
// HARDWARE MODULE LAYOUT COEFFECT (FPGA)
// ═══════════════════════════════════════════════════════════════════════════
//
// The Mealy machine an [<HardwareModule>] binding describes is synthesised by
// HardwareModulePatterns from the Design record. Every value in its body is a
// deterministic function of the design's structure and the platform's pin facts,
// so SSAAssignment derives them and the pattern reads them. One walk of the
// output record (outputExtractions) serves the derivation and the emission.

/// One hw.struct_extract the flat-port module emits while flattening the step's
/// output record into pin-mapped ports, in emission order.
type OutputExtraction = {
    /// The extraction this one reads from (None: the step result's output struct)
    Parent: int option
    /// The field extracted
    Field: string
    /// The type of the struct extracted from
    ParentType: MLIRType
    /// The type of the field
    FieldType: MLIRType
    /// The flat output port this extraction feeds, if it feeds one
    Pin: string option
}

/// The extractions the flat-port module emits for an output type under the platform's
/// pin attributes: a pinned field is extracted (a multi-pin tuple field then extracts each
/// element for its pin); an unpinned record field is extracted and walked; any other
/// unpinned field is left for synthesis to drop.
let outputExtractions (pinAttrs: Map<string, string list>) (outputType: MLIRType) : OutputExtraction list =
    let rec walk (parent: int option) (parentType: MLIRType) (acc: OutputExtraction list) : OutputExtraction list =
        match parentType with
        | TStruct fields ->
            fields |> List.fold (fun (acc: OutputExtraction list) (fieldName, fieldTy) ->
                let step pin = { Parent = parent; Field = fieldName; ParentType = parentType; FieldType = fieldTy; Pin = pin }
                match Map.tryFind fieldName pinAttrs with
                | Some [single] -> acc @ [ step (Some single) ]
                | Some multiple ->
                    match fieldTy with
                    | TStruct tupleFields ->
                        let acc' = acc @ [ step None ]
                        let idx = acc'.Length - 1
                        acc' @ (List.zip multiple tupleFields |> List.map (fun (pinName, (elemField, elemTy)) ->
                            { Parent = Some idx; Field = elemField; ParentType = fieldTy; FieldType = elemTy; Pin = Some pinName }))
                    | _ -> acc @ [ step (Some (List.head multiple)) ]
                | None ->
                    match fieldTy with
                    | TStruct _ ->
                        let acc' = acc @ [ step None ]
                        walk (Some (acc'.Length - 1)) fieldTy acc'
                    | _ -> acc) acc
        | _ -> acc
    walk None outputType []

/// The values of the Mealy machine's hw.module body, derived by SSAAssignment for the
/// [<HardwareModule>] binding and read by HardwareModulePatterns; numbered from zero in
/// emission order, so a register's feedback operand is simply its NextFields entry.
type HardwareModuleLayout = {
    /// The binding this layout is for
    BindingNodeId: NodeId
    /// The internal power-on reset (constant one, register, xor) of the flat-port module when
    /// the platform declares no external reset; None with an external reset port, and for the
    /// struct-port module, which takes rst as a port
    PowerOnReset: (SSA * SSA * SSA) option
    /// One reset constant per state field
    ResetValues: SSA list
    /// One seq.compreg per state field
    Registers: SSA list
    /// One hw.struct_create per multi-pin tuple input field, in field order (flat-port module)
    InputPacks: SSA list
    /// The packed input struct of the flat-port module with a record input
    InputStruct: SSA option
    /// hw.struct_create of the current state from the registers
    State: SSA
    /// hw.instance of the step function
    Instance: SSA
    /// Item1 (the next state) and Item2 (the outputs) of the step result, when the step reports
    StepResult: (SSA * SSA) option
    /// The flat-port module's output extractions, in outputExtractions order
    OutputFlatten: SSA list
    /// One hw.struct_extract per state field: the next value fed back to its register
    NextFields: SSA list
}
