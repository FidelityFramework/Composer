/// A declared acquisition/callback/release boundary. The compiler owns native
/// out-cells and the stack view header; no source pointer conversion is exposed.
module Alex.Patterns.MappedViewPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Patterns.PlatformPatterns
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.MappedBindings

let pMappedCall : PSGParser<MLIROp list * TransferResult> = parser {
    let! functionId, applicationArgs = pApplication
    let! node = getCurrentNode
    let! state = getUserState
    let! mapping = match tryFindCall state.Graph functionId with Some m -> preturn m | None -> fail (Message "Not a declared mapping scope")
    let args = applicationArgs
    do! ensure (args.Length = mapping.Parameters.Length) "A mapping scope must be fully applied"
    let! ssas = getNodeSSAs node.Id
    let mutable cursor = 1
    let fresh () =
        if cursor >= ssas.Length then failwith "Mapping boundary exceeds its assigned SSA family"
        let value = ssas.[cursor]
        cursor <- cursor + 1
        value
    let result = ssas.[0]
    let resultTy = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph node.Type |> narrowForCurrent state
    let! pointerBits =
        match Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.cAbiOfGraph state.Graph with
        | [(_, bits, _)] -> preturn bits
        | _ -> fail (Message "A native mapping scope requires one platform C ABI declaration")
    do! ensure (pointerBits > 0 && pointerBits <= 64) "Native mapping scope requires a supported pointer representation"
    let wordTy = TInt (IntWidth pointerBits)
    let bitTy = TInt (IntWidth 1)
    let a op = MLIROp.ArithOp op
    let m op = MLIROp.MemRefOp op
    let i op = MLIROp.IndexOp op
    let f op = MLIROp.FuncOp op
    let yieldValue value = MLIROp.SCFOp (SCFOp.Yield [value, resultTy])
    let conditional cond yes no target = MLIROp.SCFOp (SCFOp.If (cond, yes, Some no, Some (target, resultTy)))
    let constant value ty = let s = fresh () in s, a (ArithOp.ConstI (s, value, ty))
    let zero, zeroOp = constant 0L wordTy
    let zeroIndex, zeroIndexOp = constant 0L TIndex
    let one, oneOp = constant 1L wordTy
    let failure, failureOp = constant (int64 mapping.FailureStatus) resultTy
    let originalInputs = List.zip mapping.Parameters args |> List.map (fun ((name,ty,id), actual) -> name,(ty,id,actual)) |> Map.ofList
    let callbackName = mapping.Parameters |> List.find (fun (_,_,id) -> id = mapping.CallbackParameter) |> fun (name,_,_) -> name
    let _, callbackType, callbackId =
        mapping.Parameters |> List.mapi (fun n (name,ty,id) -> name,ty,args.[n]) |> List.find (fun (name,_,_) -> name = callbackName)
    let! callback, callbackTy = pRecallNode callbackId
    do! ensure (callbackTy = TMemRefStatic (2, TIndex)) "Mapping callback must use the Clef code/environment pair"
    let! viewType =
        match Clef.Compiler.NativeTypedTree.UnionFind.applySubst callbackType with
        | NativeType.TFun (view, _) -> preturn view
        | _ -> fail (Message "Mapping callback lacks its borrowed view parameter")
    let! layout =
        match Clef.Compiler.PSGSaturation.SemanticGraph.BorrowedViews.layout state.Graph viewType with
        | Ok l when l.Schema = mapping.Layout -> preturn l
        | Ok _ -> fail (Message "Mapping callback schema differs from the declared native view")
        | Result.Error e -> fail (Message e)
    let! span =
        match Clef.Compiler.PSGSaturation.SemanticGraph.MappedSpans.forLayout state.Graph layout with
        | Ok span -> preturn span
        | Result.Error e -> fail (Message e)
    do! ensure (span.PointerBits = pointerBits) "Mapped span and native ABI pointer widths disagree"
    let viewTy = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph viewType
    let! header = match viewTy with TStruct (_, Some bytes) -> preturn bytes | _ -> fail (Message "Mapping view header is not settled")
    let dataTy = TMemRef (TInt (IntWidth layout.ElementBits))
    let inputOps = ResizeArray<MLIROp>()
    let inputChecks = ResizeArray<SSA>()
    // Values are checked at their observed source width before ABI narrowing.
    let scalarInput actual (declared: Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.DeclaredParameter) value ty =
        let sourceRange = nodeRange state.Graph actual |> Option.defaultValue ValueRange.Unbounded
        let unsigned = ValueRange.isNonNegative sourceRange
        match ty with
        | TInt (IntWidth sourceBits) when sourceBits > 0 ->
            let checkBits = max (sourceBits + (if unsigned then 1 else 0))
                                (declared.Bits + (if ValueRange.isNonNegative declared.Range then 1 else 0))
            let checkTy = TInt (IntWidth checkBits)
            let checkedValue =
                if checkBits = sourceBits then value else
                let extended = fresh ()
                inputOps.Add(a ((if unsigned then ArithOp.ExtUI else ArithOp.ExtSI) (extended, value, ty, checkTy)))
                extended
            match ValueRange.endpoints declared.Range with
            | Some (ValueRange.Endpoint.Finite lo, ValueRange.Endpoint.Finite hi) when hi <= bigint System.Int64.MaxValue ->
                let low, lowOp = constant (int64 lo) checkTy
                let high, highOp = constant (int64 hi) checkTy
                let lowOk, highOk = fresh (), fresh ()
                inputOps.AddRange [lowOp; highOp
                                   a (ArithOp.CmpI (lowOk, ICmpPred.Sge, checkedValue, low, checkTy))
                                   a (ArithOp.CmpI (highOk, ICmpPred.Sle, checkedValue, high, checkTy))]
                inputChecks.Add lowOk
                inputChecks.Add highOk
            | _ -> failwith "Mapped scalar input requires finite native bounds supported by its adapter"
            let destTy = TInt (IntWidth declared.Bits)
            if ty = destTy then { SSA = value; Type = ty } else
            let converted = fresh ()
            inputOps.Add(a ((if sourceBits > declared.Bits then ArithOp.TruncI elif unsigned then ArithOp.ExtUI else ArithOp.ExtSI)
                                (converted, value, ty, destTy)))
            { SSA = converted; Type = destTy }
        | _ -> failwith "Mapped scalar input has no observed integer representation"
    let! nativeInputs =
        let rec collect parameters = parser {
            match parameters with
            | [] -> return []
            | (name, _, nativeId) :: rest ->
                let scalarRef = mapping.Acquire.References |> List.tryFind (fst >> (=) nativeId)
                let pointerRef = mapping.Acquire.PointerReferences |> List.tryFind (fst >> (=) nativeId)
                match scalarRef, pointerRef with
                | Some _, _ | _, Some _ ->
                    let! remaining = collect rest
                    return remaining
                | _ ->
                    let sourceTy, _, actual = originalInputs.[name]
                    let! value, ty = pRecallNode actual
                    let! native = parser {
                        match mapping.Acquire.Parameters |> List.tryFind (fst >> (=) nativeId) |> Option.bind snd with
                        | Some scalar -> return scalarInput actual scalar value ty
                        | None when ty = TIndex ->
                            let native = fresh ()
                            inputOps.Add(i (IndexOp.IndexCastU (native, value, TIndex, wordTy)))
                            return { SSA = native; Type = wordTy }
                        | None when isNullableHandle sourceTy ->
                            let start = cursor
                            cursor <- cursor + 13
                            let! ops, native = pUnwrapOptionArgForFFI sourceTy value ty wordTy ssas start
                            inputOps.AddRange ops
                            return native
                        | _ -> return! fail (Message "Mapping native input lacks a supported ABI descriptor")
                    }
                    let! remaining = collect rest
                    return (name, native) :: remaining
        }
        collect mapping.AcquireParameters
    let nativeInputs = Map.ofList nativeInputs
    let cellOps = ResizeArray<MLIROp>()
    let outputLoads = ResizeArray<MLIROp>()
    let! outputs =
        let rec collect parameters = parser {
            match parameters with
            | [] -> return []
            | (name, _, nativeId) :: rest ->
                let scalarRef = mapping.Acquire.References |> List.tryFind (fst >> (=) nativeId)
                let pointerRef = mapping.Acquire.PointerReferences |> List.tryFind (fst >> (=) nativeId)
                let representation =
                    match scalarRef, pointerRef with
                    | Some (_, scalar), _ -> Some (TInt (IntWidth scalar.Bits), ValueRange.isNonNegative scalar.Range)
                    | _, Some (_, pointer) -> Some (TInt (IntWidth pointer.Bits), true)
                    | _ -> None
                match representation with
                | None -> return! collect rest
                | Some (ty, unsigned) ->
                    let cell, address, nativeAddress, loaded = fresh (), fresh (), fresh (), fresh ()
                    let cellTy = TMemRefStatic (1, ty)
                    let! allocate = pAlloca cell 1 ty (Some (mlirTypeSize state.Platform.TargetArch ty))
                    let! addressOp = pExtractBasePtr address cell cellTy
                    let initial, initialOp = constant 0L ty
                    cellOps.AddRange [allocate; initialOp; m (MemRefOp.Store (initial, cell, [zeroIndex], ty, cellTy)); addressOp
                                      i (IndexOp.IndexCastU (nativeAddress, address, TIndex, wordTy))]
                    outputLoads.Add(m (MemRefOp.Load (loaded, cell, [zeroIndex], ty, cellTy)))
                    let! remaining = collect rest
                    return (name, ({ SSA = nativeAddress; Type = wordTy }, { SSA = loaded; Type = ty }, unsigned)) :: remaining
        }
        collect mapping.AcquireParameters
    let outputs = Map.ofList outputs
    let acquireArgs = mapping.AcquireParameters |> List.map (fun (name,_,_) ->
        match Map.tryFind name outputs with Some (address,_,_) -> address | None -> nativeInputs.[name])
    let releaseArgs = mapping.ReleaseArguments |> List.map (function
        | Input name -> nativeInputs.[name]
        | Output name -> let _, value, _ = outputs.[name] in value)
    let acquireName, releaseName = "ffi." + mapping.Acquire.CName, "ffi." + mapping.Release.CName
    let linked binding =
        state.Graph.Nodes.[binding].Metadata |> Map.tryFind "FidelityExtern.Library"
        |> Option.exists (function MetadataValue.String library -> isLinkedExtern state.Platform library | _ -> false)
    do! ensure (linked mapping.AcquireBinding && linked mapping.ReleaseBinding) "Mapped acquisition and release must name explicitly linked native libraries"
    let acquireDecl = f (FuncOp.FuncDecl (acquireName, List.map (fun (v:Val) -> v.Type) acquireArgs, wordTy, FuncVisibility.Private, []))
    let releaseDecl = f (FuncOp.FuncDecl (releaseName, List.map (fun (v:Val) -> v.Type) releaseArgs, TVoid, FuncVisibility.Private, []))
    let raw, acquired = fresh (), fresh ()
    let acquire = f (FuncOp.FuncCall (Some raw, acquireName, acquireArgs, wordTy))
    let release () = f (FuncOp.FuncCall (None, releaseName, releaseArgs, TVoid))
    let shapeOps = ResizeArray<MLIROp>()
    let wordValue = function
        | Input name -> nativeInputs.[name], true
        | Output name -> let _, value, unsigned = outputs.[name] in value, unsigned
    let asWord selector =
        let value, unsigned = wordValue selector
        if value.Type = wordTy then value.SSA else
        let widened = fresh ()
        match value.Type with
        | TInt (IntWidth bits) when bits < pointerBits ->
            shapeOps.Add(a ((if unsigned then ArithOp.ExtUI else ArithOp.ExtSI) (widened, value.SSA, value.Type, wordTy)))
            widened
        | _ -> failwith "Mapped extent exceeds the native pointer representation"
    let stride, rows, width = asWord mapping.RowStride, asWord mapping.RowCount, asWord mapping.RowWidth
    let elementBytes, elementBytesOp = constant (int64 span.ElementBytes) wordTy
    let alignment, alignmentOp = constant (int64 span.BaseAlignment) wordTy
    let maxExtent, maxExtentOp = constant (int64 span.MaximumExtent) wordTy
    shapeOps.AddRange [elementBytesOp; alignmentOp; maxExtentOp]
    let checks = ResizeArray<SSA>()
    let compare pred lhs rhs = let s = fresh () in shapeOps.Add(a (ArithOp.CmpI (s, pred, lhs, rhs, wordTy))); checks.Add s; s
    compare ICmpPred.Ugt stride zero |> ignore
    let rowsPositive = compare ICmpPred.Ugt rows zero
    compare ICmpPred.Ugt width zero |> ignore
    let safeRows, limit, elementsPerRow, remainder, baseAlignment, rowAlignment = fresh (), fresh (), fresh (), fresh (), fresh (), fresh ()
    shapeOps.AddRange [a (ArithOp.Select (safeRows, rowsPositive, rows, one, wordTy));
                       a (ArithOp.DivUI (limit, maxExtent, safeRows, wordTy));
                       a (ArithOp.DivUI (elementsPerRow, stride, elementBytes, wordTy));
                       a (ArithOp.RemUI (remainder, stride, elementBytes, wordTy));
                       a (ArithOp.RemUI (baseAlignment, raw, alignment, wordTy));
                       a (ArithOp.RemUI (rowAlignment, stride, alignment, wordTy))]
    compare ICmpPred.Ule stride limit |> ignore
    compare ICmpPred.Ule width elementsPerRow |> ignore
    compare ICmpPred.Eq remainder zero |> ignore
    compare ICmpPred.Eq baseAlignment zero |> ignore
    compare ICmpPred.Eq rowAlignment zero |> ignore
    let combine (ops:ResizeArray<MLIROp>) (values:ResizeArray<SSA>) =
        if values.Count = 0 then let s, op = constant 1L bitTy in ops.Add op; s
        else values |> Seq.skip 1 |> Seq.fold (fun current value -> let s = fresh () in ops.Add(a (ArithOp.AndI (s, current, value, bitTy))); s) values.[0]
    let shapeValid = combine shapeOps checks
    let inputValid = combine inputOps inputChecks
    let bytes, count, countIndex, pointerIndex, strideIndex, rawView, data, view = fresh (), fresh (), fresh (), fresh (), fresh (), fresh (), fresh (), fresh ()
    let extentOps = [a (ArithOp.MulI (bytes, stride, rows, wordTy)); a (ArithOp.DivUI (count, bytes, elementBytes, wordTy))
                     i (IndexOp.IndexCastU (countIndex, count, wordTy, TIndex)); i (IndexOp.IndexCastU (pointerIndex, raw, wordTy, TIndex))
                     i (IndexOp.IndexCastU (strideIndex, stride, wordTy, TIndex))
                     m (MemRefOp.IndexToMemRef (rawView, pointerIndex, dataTy))
                     m (MemRefOp.ReinterpretCastDynamic (data, rawView, 0, countIndex, dataTy, dataTy))]
    let! allocateHeader = pAlloca view header.Size (TInt (IntWidth 8)) (Some header.Align)
    let! storeData = pTypedInsertView view data header.Offsets.[0] (fresh ()) (fresh ()) (fresh ()) dataTy viewTy
    let! storeStride = pTypedInsertView view strideIndex header.Offsets.[1] (fresh ()) (fresh ()) (fresh ()) TIndex viewTy
    let oneIndex, oneIndexOp = constant 1L TIndex
    let code, environment, functionPointer, callbackResult = fresh (), fresh (), fresh (), fresh ()
    let callbackOps = [oneIndexOp
                       m (MemRefOp.Load (code, callback, [zeroIndex], TIndex, callbackTy))
                       m (MemRefOp.Load (environment, callback, [oneIndex], TIndex, callbackTy))
                       f (FuncOp.IndexToFunc (functionPointer, code, [TIndex; viewTy], resultTy))
                       f (FuncOp.FuncCallIndirect (Some callbackResult, functionPointer, [{ SSA = environment; Type = TIndex }; { SSA = view; Type = viewTy }], resultTy))]
    let validResult, acquiredResult = fresh (), fresh ()
    let baseLimit, addressValid, addressResult = fresh (), fresh (), fresh ()
    // The extent product is used only after its division guard has established
    // representability. Also exclude wraparound of base plus the full extent.
    let addressOps = [a (ArithOp.SubI (baseLimit, maxExtent, bytes, wordTy))
                      a (ArithOp.CmpI (addressValid, ICmpPred.Ule, raw, baseLimit, wordTy))]
    let addressBranch = conditional addressValid ([allocateHeader] @ storeData @ storeStride @ callbackOps @ [release (); yieldValue callbackResult])
                                                [release (); yieldValue failure] addressResult
    let shapeBranch = conditional shapeValid (extentOps @ addressOps @ [addressBranch; yieldValue addressResult])
                                            [release (); yieldValue failure] validResult
    let acquiredBranch = conditional acquired (List.ofSeq outputLoads @ List.ofSeq shapeOps @ [shapeBranch; yieldValue validResult]) [yieldValue failure] acquiredResult
    let acquireBody = List.ofSeq cellOps @ [acquire; a (ArithOp.CmpI (acquired, ICmpPred.Ne, raw, zero, wordTy)); acquiredBranch; yieldValue acquiredResult]
    let ops = [acquireDecl; releaseDecl; zeroOp; zeroIndexOp; oneOp; failureOp] @ List.ofSeq inputOps @
              [conditional inputValid acquireBody [yieldValue failure] result]
    return ops, TRValue { SSA = result; Type = resultTy }
}
