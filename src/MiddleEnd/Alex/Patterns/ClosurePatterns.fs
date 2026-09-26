/// ClosurePatterns - Closure and lambda operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns for lambda and closure operations.
/// Patterns compose Elements into semantic closure/lambda operations.
module Alex.Patterns.ClosurePatterns

open XParsec
open XParsec.Parsers     // preturn, fail
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.XParsec.Extensions // sequence combinator
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements
open Alex.Elements.FuncElements
open Alex.Elements.HWElements   // pHWModule, pHWOutput (FPGA function wrapping)
open Alex.CodeGeneration.TypeMapping
open Core.Types.Dialects        // TargetPlatform (codata-dependent elision)
open Clef.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════
// FUNCTION DEFINITION
// ═══════════════════════════════════════════════════════════

/// Wrap already witnessed, ordered result components. A callable result stays
/// two independent values; this pattern performs no graph or closure analysis.
let pFunctionDefResults (visibility: FuncVisibility) (name: string) (parameters: (SSA * MLIRType) list)
                        (parameterNames: string list option) (resultTypes: MLIRType list)
                        (bodyOps: MLIROp list) (results: Val list) : PSGParser<MLIROp> =
    parser {
        do! ensure ((results |> List.map _.Type) = resultTypes) $"pFunctionDefResults: function '{name}' has mismatched result components"
        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            let names = parameterNames |> Option.defaultValue (parameters |> List.mapi (fun i _ -> sprintf "in%d" i))
            do! ensure (names.Length = parameters.Length) $"pFunctionDefResults: function '{name}' has mismatched input names"
            let inputs = List.map2 (fun name (_, ty) -> name, ty) names parameters
            let outputs = resultTypes |> List.mapi (fun index ty -> sprintf "result%d" index, ty)
            let! terminal = pHWOutput (results |> List.map (fun value -> value.SSA, value.Type))
            return! pHWModule name inputs outputs (bodyOps @ [terminal])
        | _ ->
            let! terminal = pFuncReturnResults results
            return! pFuncDefResults name parameters resultTypes (bodyOps @ [terminal]) visibility
    }

/// Create function definition (func.func for named calls, llvm.func for closures)
/// Coeffect-aware function definition wrapping.
/// Observes TargetPlatform to elide to func.func (CPU) or hw.module (FPGA).
/// Handles the function terminator internally: func.return (CPU) or hw.output (FPGA).
///
/// `paramNames`: optional port names for hw.module (defaults to "in0", "in1", ...)
/// `returnSSA`: the SSA of the return value (None for a unit function)
/// `unitReturnSSA`: for a unit function, the zero constant it returns, derived by SSAAssignment
let pFunctionDef (visibility: FuncVisibility) (name: string) (params': (SSA * MLIRType) list) (paramNames: string list option)
                 (retTy: MLIRType) (bodyOps: MLIROp list) (returnSSA: SSA option) (unitReturnSSA: SSA option)
                 : PSGParser<MLIROp> =
    parser {
        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            // hw.module with named input/output ports
            let names = paramNames |> Option.defaultValue (params' |> List.mapi (fun i _ -> sprintf "in%d" i))
            let inputs = List.map2 (fun pname (_, ty) -> (pname, ty)) names params'
            let outputs =
                match returnSSA with
                | Some _ -> [("result", retTy)]
                | None -> []
            let outputOp = MLIROp.HWOp (HWOp.HWOutput (match returnSSA with
                                                        | Some ssa -> [(ssa, retTy)]
                                                        | None -> []))
            let body = bodyOps @ [outputOp]
            return! pHWModule name inputs outputs body
        | _ ->
            // func.func with positional parameters
            // Preserve earlier witness diagnostics when a non-unit body failed
            // to produce a value, instead of throwing over the accumulated errors.
            do! ensure (returnSSA.IsSome || unitReturnSSA.IsSome) $"pFunctionDef: function '{name}' has no witnessed return value"
            // A unit function: returnSSA = None but retTy is concrete (e.g. i32), and
            // func.return needs an operand of that type. The zero constant it returns is the
            // value SSAAssignment derived for the Lambda (the first of its body's scope).
            let actualReturnSSA, extraOps =
                match returnSSA, unitReturnSSA with
                | Some _, _ -> returnSSA, []
                | None, Some zeroSSA ->
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, retTy))
                    (Some zeroSSA, [zeroOp])
                | None, None ->
                    failwithf "pFunctionDef: function '%s' returns no value and SSAAssignment derived no unit-return value for its Lambda; the derivation covers every unit-typed body" name
            let! returnOp = pFuncReturn actualReturnSSA (Some retTy)
            let body = bodyOps @ extraOps @ [returnOp]
            return! pFuncDef name params' retTy body visibility
    }

// ═══════════════════════════════════════════════════════════
// CAPTURE EXTRACTION
// ═══════════════════════════════════════════════════════════

/// Extract captures from closure struct at function entry via typed reinterpret_cast.
/// The closure struct is a byte-level memref (memref<Nxi8>). Each capture is extracted
/// by reinterpret_casting to a typed view at the correct byte offset, then loading.
///
/// Slot type dispatches extraction strategy (pattern match on coeffect):
///   - Scalar (TIndex, TInt, TFloat): pTypedExtract — 3 SSAs (view, zero, result)
///   - Decomposed memref (TStruct [ptr; len]): load ptr + len separately,
///     reconstruct memref via IndexToMemRef + ReinterpretCastDynamic — 8 SSAs total
///
/// Each capture's slot type and byte offset, and its values (its work values then its result,
/// `ClosureLayout.CaptureExtractionSSAs`), are read from the closure layout SSAAssignment
/// derived; nothing is counted or summed here.
let pExtractCaptures (captures: (MLIRType * int * MLIRType) list) (structType: MLIRType) (envSSA: SSA) (ssas: SSA list list) : PSGParser<MLIROp list> =
    parser {
        let envPtrSSA = envSSA  // Reconstructed memref<Nxi8> from caller

        let! extractOpLists =
            List.zip captures ssas
            |> List.map (fun ((capTy, byteOffset, valueTy), captureSSAs) ->
                parser {
                    match capTy, captureSSAs with
                    | TStruct ([("ptr", TIndex); ("len", TIndex)], bytes), [ ptrViewSSA; ptrZeroSSA; ptrSSA; lenViewSSA; lenZeroSSA; lenSSA; rawMemrefSSA; resultSSA ] ->
                        // Decomposed memref capture: load the base index and the extent, reconstruct the memref
                        let ptrByteOffset = byteOffset
                        let lenByteOffset =
                            match bytes with
                            | Some b -> byteOffset + b.Offsets.[1]
                            | None -> failwith "pExtractCaptures: a decomposed string slot with no derived layout"
                        // Load ptr (TIndex) at ptrByteOffset
                        let! ptrOps = pTypedExtract ptrSSA envPtrSSA ptrByteOffset ptrViewSSA ptrZeroSSA TIndex structType
                        // Load len (TIndex) at lenByteOffset
                        let! lenOps = pTypedExtract lenSSA envPtrSSA lenByteOffset lenViewSSA lenZeroSSA TIndex structType
                        // Preserve the captured buffer's settled element type and actual extent.
                        let dynMemrefTy = valueTy
                        let castOp = MLIROp.MemRefOp(MemRefOp.IndexToMemRef(rawMemrefSSA, ptrSSA, dynMemrefTy))
                        // Set size: reinterpret_cast with dynamic length
                        let sizeOp = MLIROp.MemRefOp(MemRefOp.ReinterpretCastDynamic(resultSSA, rawMemrefSSA, 0, lenSSA, dynMemrefTy, dynMemrefTy))
                        return ptrOps @ lenOps @ [castOp; sizeOp]

                    | TIndex, [ viewSSA; zeroSSA; ptrSSA; rawMemrefSSA; resultSSA ] ->
                        match valueTy with
                        | TMemRefStatic (count, element) ->
                            let! ptrOps = pTypedExtract ptrSSA envPtrSSA byteOffset viewSSA zeroSSA TIndex structType
                            let rawTy = TMemRef element
                            let castOp = MLIROp.MemRefOp(MemRefOp.IndexToMemRef(rawMemrefSSA, ptrSSA, rawTy))
                            let sizeOp = MLIROp.MemRefOp(MemRefOp.ReinterpretCast(resultSSA, rawMemrefSSA, 0, count, rawTy, valueTy))
                            return ptrOps @ [castOp; sizeOp]
                        | TStruct (_, Some bytes) ->
                            // Records retain their settled field layout for body accesses;
                            // their physical carrier is the same bounded byte view used at construction.
                            let! ptrOps = pTypedExtract ptrSSA envPtrSSA byteOffset viewSSA zeroSSA TIndex structType
                            let rawTy = TMemRef (TInt (IntWidth 8))
                            let castOp = MLIROp.MemRefOp(MemRefOp.IndexToMemRef(rawMemrefSSA, ptrSSA, rawTy))
                            let sizeOp = MLIROp.MemRefOp(MemRefOp.ReinterpretCast(resultSSA, rawMemrefSSA, 0, bytes.Size, rawTy, valueTy))
                            return ptrOps @ [castOp; sizeOp]
                        | _ -> return! fail (Message $"pExtractCaptures: address slot has no settled static view: {valueTy}")

                    | _, [ viewSSA; zeroSSA; resultSSA ] ->
                        // Scalar capture: standard typed extraction at the slot's settled offset
                        return! pTypedExtract resultSSA envPtrSSA byteOffset viewSSA zeroSSA capTy structType

                    | _, values ->
                        return! fail (Message $"pExtractCaptures: a slot of type {capTy} was derived {values.Length} values; the derivation and the pattern disagree")
                })
            |> sequence

        return List.concat extractOpLists
    }

// ═══════════════════════════════════════════════════════════
// LAZY PATTERNS (PRD-14)
// ═══════════════════════════════════════════════════════════

/// Lazy struct: {computed: i1, value: T, code_ptr, captures...}
/// SSA layout: [0] = undef, [1] = falseConst, [2-3] = insert computed (offset, result), [4-5] = insert code_ptr (offset, result), then for each capture: [6+2*i] = offsetSSA, [7+2*i] = resultSSA
let pLazyStruct (valueTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let arch = state.Platform.TargetArch
        do! ensure (ssas.Length >= 6 + 2 * captures.Length) $"pLazyStruct: Expected at least {6 + 2 * captures.Length} SSAs, got {ssas.Length}"

        // Compute lazy type: {computed: i1, value: T, code_ptr: ptr, captures...}
        let fieldTypes = [TInt (IntWidth 1); valueTy; codePtrTy] @ (captures |> List.map (fun cap -> cap.Type))
        let totalBytes = fieldTypes |> List.sumBy (mlirTypeSize arch)
        let lazyTy = TMemRefStatic(totalBytes, TInt (IntWidth 8))

        // Create undef struct
        let! undefOp = pUndef ssas.[0] lazyTy

        // Insert computed = false at index 0
        let computedTy = TInt (IntWidth 1)
        let! falseConstOp = pConstI ssas.[1] 0L computedTy
        let! insertComputedOps = pInsertValue ssas.[3] ssas.[0] ssas.[1] 0 ssas.[2] lazyTy

        // Insert code_ptr at index 2
        let! insertCodeOps = pInsertValue ssas.[5] ssas.[3] codePtr 2 ssas.[4] lazyTy

        // Insert captures starting at index 3
        let! captureOpLists =
            captures
            |> List.mapi (fun i cap ->
                parser {
                    let offsetSSA = ssas.[6 + 2*i]
                    let targetSSA = ssas.[7 + 2*i]
                    let sourceSSA = if i = 0 then ssas.[5] else ssas.[5 + 2*i]
                    return! pInsertValue targetSSA sourceSSA cap.SSA (i + 3) offsetSSA lazyTy
                })
            |> sequence

        return [undefOp; falseConstOp] @ insertComputedOps @ insertCodeOps @ List.concat captureOpLists
    }

/// Build lazy struct: High-level pattern for witnesses
/// Combines pLazyStruct with proper result construction
let pBuildLazyStruct (valueTy: MLIRType) (codePtrTy: MLIRType) (codePtr: SSA) (captures: Val list)
                     (ssas: SSA list) (arch: Architecture)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Call low-level pattern to build struct
        let! ops = pLazyStruct valueTy codePtrTy codePtr captures ssas

        // Final SSA is the last one (after all insertions: undef + falseConst + 2*insertComputed + 2*insertCode + 2*captures)
        let finalSSA = ssas.[5 + 2 * captures.Length]

        // Lazy type is {computed: i1, value: T, code_ptr, captures...}
        let fieldTypes = [TInt (IntWidth 1); valueTy; codePtrTy] @ (captures |> List.map (fun cap -> cap.Type))
        let totalBytes = fieldTypes |> List.sumBy (mlirTypeSize arch)
        let mlirType = TMemRefStatic(totalBytes, TInt (IntWidth 8))

        return (ops, TRValue { SSA = finalSSA; Type = mlirType })
    }

/// Build lazy force: Call lazy thunk via struct pointer passing
///
/// LazyForce is a SIMPLE operation (not elaborated by CCS).
/// SSA cost: Fixed 5 (extract code_ptr, const 1, alloca, index, store, call)
///
/// Calling convention: Thunk receives pointer to lazy struct
/// 1. Extract code_ptr from lazy struct [2]
/// 2. Alloca space for lazy struct on stack (const 1 for size)
/// 3. Store lazy struct to get pointer
/// 4. Call thunk with pointer -> result
///
/// The thunk extracts captures internally using LazyLayout coeffect.
///
/// Lazy struct: {computed: i1, value: T, code_ptr: ptr, capture0, capture1, ...}
let pBuildLazyForce (lazySSA: SSA) (lazyTy: MLIRType) (resultSSA: SSA) (resultTy: MLIRType)
                    (ssas: SSA list) (arch: Architecture)
                    : PSGParser<MLIROp list * TransferResult> =
    parser {
        // SSAs: [0-1] = code_ptr extract (offset, result), [2] = const 1, [3] = alloca'd ptr, [4] = index
        do! ensure (ssas.Length >= 5) $"pBuildLazyForce: Expected at least 5 SSAs, got {ssas.Length}"

        let codeOffsetSSA = ssas.[0]
        let codePtrSSA = ssas.[1]
        let constOneSSA = ssas.[2]
        let ptrSSA = ssas.[3]
        let indexSSA = ssas.[4]

        // Extract code_ptr from lazy struct [2] - it's always a ptr type
        let codePtrTy = TIndex
        let! extractCodePtrOps = pExtractValue codePtrSSA lazySSA 2 codeOffsetSSA codePtrTy

        // Alloca space for lazy struct
        let constOneTy = TInt (IntWidth 64)
        let! constOneOp = pConstI constOneSSA 1L constOneTy
        let! allocaOp = pAlloca ptrSSA 1 lazyTy None

        // Store lazy struct to alloca'd space
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for 1-element memref
        let memrefType = TMemRefStatic (1, lazyTy)  // 1-element lazy value storage
        let! storeOp = pStore lazySSA ptrSSA [indexSSA] lazyTy memrefType

        // Call thunk with pointer -> result
        let argVals = [{ SSA = ptrSSA; Type = TIndex }]
        let! callOp = pFuncCallIndirect (Some resultSSA) codePtrSSA argVals resultTy

        return (extractCodePtrOps @ [constOneOp; allocaOp; indexOp; storeOp; callOp], TRValue { SSA = resultSSA; Type = resultTy })
    }
