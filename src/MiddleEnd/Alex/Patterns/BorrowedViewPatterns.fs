/// BAREWire's schema supplies physical element storage. A borrowed view never
/// inherits the physical width selected for unrelated source integer arrays.
module Alex.Patterns.BorrowedViewPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.CodeGeneration.TypeMapping
open Alex.Elements.MLIRAtomics
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.BorrowedViews

let pBorrowedViewIntrinsic : PSGParser<MLIROp list * TransferResult> = parser {
    let! info, args = pIntrinsicApplication IntrinsicModule.BorrowedView
    let! node = getCurrentNode
    let! state = getUserState
    let! ssas = getNodeSSAs node.Id
    do! ensure (not args.IsEmpty && ssas.Length >= 40) "BorrowedView requires its view and assigned work values"
    let s i = ssas.[i]
    let! layout = match layout state.Graph state.Graph.Nodes.[args.Head].Type with Ok l -> preturn l | Result.Error e -> fail (Message e)
    let! span =
        match Clef.Compiler.PSGSaturation.SemanticGraph.MappedSpans.forLayout state.Graph layout with
        | Ok model -> preturn model
        | Result.Error e -> fail (Message e)
    let! view, viewTy = pRecallNode args.Head
    let! header = match viewTy with TStruct (_, Some b) -> preturn b | _ -> fail (Message "BorrowedView has no settled scope header")
    let dataTy = TMemRef (TInt (IntWidth layout.ElementBits))
    let resultTy = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph node.Type |> narrowForCurrent state
    let! dataOps = pTypedExtractView (s 1) view header.Offsets.[0] (s 20) (s 21) (s 22) dataTy viewTy
    let zero = MLIROp.ArithOp (ArithOp.ConstI (s 4, 0L, TIndex))
    let length = MLIROp.MemRefOp (MemRefOp.Dim (s 5, s 1, s 4, dataTy))
    match info.Operation, args with
    | "length", [_] ->
        return dataOps @ [zero; length; MLIROp.IndexOp (IndexOp.IndexCastU (s 0, s 5, TIndex, resultTy))], TRValue { SSA = s 0; Type = resultTy }
    | "stride", [_] ->
        let! ops = pTypedExtractView (s 5) view header.Offsets.[1] (s 20) (s 21) (s 22) TIndex viewTy
        return ops @ [MLIROp.IndexOp (IndexOp.IndexCastU (s 0, s 5, TIndex, resultTy))], TRValue { SSA = s 0; Type = resultTy }
    | ("get" | "set" as operation), _ when args.Length = (if info.Operation = "get" then 2 else 3) ->
        do! ensure ((operation = "get" && layout.Access <> "WriteOnly") || (operation = "set" && layout.Access <> "ReadOnly"))
                   "BorrowedView operation violates the declared access permission"
        let! index, indexTy = pRecallNode args.[1]
        let beforeIndexCast =
            match indexTy with
            | TInt (IntWidth bits) when bits > span.PointerBits ->
                [MLIROp.ArithOp (ArithOp.ConstI (s 23, int64 span.MaximumExtent, indexTy))
                 MLIROp.ArithOp (ArithOp.CmpI (s 24, ICmpPred.Ule, index, s 23, indexTy))
                 MLIROp.Assert (s 24, "BorrowedView index cannot be represented in the mapped address space")]
            | _ -> []
        let indexOps, index =
            if indexTy = TIndex then [], index
            else [Alex.Patterns.MemoryPatterns.indexCastForRange
                    (nodeRange state.Graph args.[1] |> Option.defaultValue ValueRange.Unbounded) (s 6) index indexTy], s 6
        let valid = MLIROp.ArithOp (ArithOp.CmpI (s 7, ICmpPred.Ult, index, s 5, TIndex))
        let prefix = dataOps @ [zero; length] @ beforeIndexCast @ indexOps @ [valid; MLIROp.Assert (s 7, "BorrowedView index is outside the mapped extent")]
        let elementTy = TInt (IntWidth layout.ElementBits)
        if operation = "get" then
            return prefix @ [MLIROp.MemRefOp (MemRefOp.LoadAligned (s 0, s 1, [index], elementTy, dataTy, span.ElementAlignment))],
                   TRValue { SSA = s 0; Type = elementTy }
        else
            // Check the original value before narrowing it to physical storage.
            let! value, valueTy = pRecallNode args.[2]
            let! bits = match valueTy with TInt (IntWidth b) when b > 0 -> preturn b | _ -> fail (Message "BorrowedView.set requires an observed integer range")
            let! lo, hi =
                match ValueRange.endpoints layout.Range with
                | Some (ValueRange.Endpoint.Finite lo, ValueRange.Endpoint.Finite hi) -> preturn (lo, hi)
                | _ -> fail (Message "BorrowedView.set requires finite schema bounds")
            let sourceRange = nodeRange state.Graph args.[2] |> Option.defaultValue ValueRange.Unbounded
            let unsigned = ValueRange.isNonNegative sourceRange
            let checkBits = max (bits + (if unsigned then 1 else 0))
                                (layout.ElementBits + (if ValueRange.isNonNegative layout.Range then 1 else 0))
            let checkTy = TInt (IntWidth checkBits)
            let extend =
                if bits = checkBits then [] else
                [MLIROp.ArithOp ((if unsigned then ArithOp.ExtUI else ArithOp.ExtSI) (s 8, value, valueTy, checkTy))]
            let checkedValue = if bits = checkBits then value else s 8
            let constant id temporary number =
                if number <= bigint System.Int64.MaxValue then [MLIROp.ArithOp (ArithOp.ConstI (id, int64 number, checkTy))]
                else
                    let raw = int64 (number - (1I <<< 64))
                    [MLIROp.ArithOp (ArithOp.ConstI (temporary, raw, TInt (IntWidth 64)))
                     MLIROp.ArithOp (ArithOp.ExtUI (id, temporary, TInt (IntWidth 64), checkTy))]
            let predLow, predHigh = ICmpPred.Sge, ICmpPred.Sle
            let checks = constant (s 9) (s 15) lo @ constant (s 10) (s 16) hi @
                         [MLIROp.ArithOp (ArithOp.CmpI (s 11, predLow, checkedValue, s 9, checkTy))
                          MLIROp.ArithOp (ArithOp.CmpI (s 12, predHigh, checkedValue, s 10, checkTy))
                          MLIROp.ArithOp (ArithOp.AndI (s 13, s 11, s 12, TInt (IntWidth 1)))
                          MLIROp.Assert (s 13, "BorrowedView value is outside the declared element range")]
            let conversion =
                if bits = layout.ElementBits then []
                elif bits > layout.ElementBits then [MLIROp.ArithOp (ArithOp.TruncI (s 14, value, valueTy, elementTy))]
                else [MLIROp.ArithOp ((if unsigned then ArithOp.ExtUI else ArithOp.ExtSI) (s 14, value, valueTy, elementTy))]
            let stored = if conversion.IsEmpty then value else s 14
            return prefix @ extend @ checks @ conversion @
                   [MLIROp.MemRefOp (MemRefOp.StoreAligned (stored, s 1, [index], elementTy, dataTy, span.ElementAlignment))
                    MLIROp.ArithOp (ArithOp.ConstI (s 0, 0L, TInt (IntWidth 32)))],
                   TRValue { SSA = s 0; Type = TInt (IntWidth 32) }
    | _ -> return! fail (Message "Unsupported BorrowedView intrinsic arity")
}
