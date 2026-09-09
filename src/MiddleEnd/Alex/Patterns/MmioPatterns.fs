module Alex.Patterns.MmioPatterns
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution

let pMmioIntrinsic : PSGParser<MLIROp list * TransferResult> = parser {
    let! info, args = pIntrinsicApplication IntrinsicModule.Mmio
    let! node = getCurrentNode
    let! state = getUserState
    let! ssas = getNodeSSAs node.Id
    let s i = ssas.[i]
    let isReg = info.Operation.StartsWith("reg")
    let isRead = info.Operation.StartsWith("read")
    let bits = int (info.Operation.Substring(if isReg then 3 elif isRead then 4 else 5))
    let! pointerBits = match state.Platform.TargetArch.Pointer with Ok b -> preturn b | Result.Error e -> fail (Message e)
    do! ensure (bits = 8 || bits = 16 || bits = 32) "Unsupported MMIO access width"
    if isReg then
        let! address =
            match args with
            | [id] ->
                match int64Of state.Graph id with
                | Some a -> preturn a
                | _ -> fail (Message "MMIO address must be a statically declared integer")
            | _ -> fail (Message "MMIO constructor requires one address")
        do! ensure (address > 0L && bigint address + bigint (bits / 8) <= (1I <<< pointerBits) && address % int64 (bits / 8) = 0L)
                   "MMIO address is null, outside the platform address space, or misaligned"
        return [MLIROp.ArithOp (ArithOp.ConstI (s 0, address, TIndex))], TRValue { SSA = s 0; Type = TIndex }
    else
        do! ensure (args.Length = (if isRead then 1 else 2)) "Invalid MMIO accessor arity"
        let handleType = applySubst state.Graph.Nodes.[args.Head].Type
        do! ensure (match handleType with NativeType.TApp(tc, []) -> tc.Name = "Mmio" + string bits | _ -> false)
                   "MMIO access width does not match its opaque register handle"
        let! address, addressTy = pRecallNode args.Head
        do! ensure (addressTy = TIndex) "MMIO handle must have the platform pointer representation"
        let elementTy = TInt (IntWidth bits)
        if isRead then
            return [MLIROp.MmioLoad (s 0, address, s 1, s 2, bits)], TRValue { SSA = s 0; Type = elementTy }
        else
            // No implicit narrowing at this boundary. Require the complete
            // source range to fit; users can mask explicitly for bit fields.
            let range = nodeRange state.Graph args.[1] |> Option.defaultValue ValueRange.Unbounded
            do! ensure (match ValueRange.endpoints range with
                        | Some (ValueRange.Endpoint.Finite lo, ValueRange.Endpoint.Finite hi) -> lo >= 0I && hi < (1I <<< bits)
                        | _ -> false) "MMIO write value is not proven within the unsigned register width"
            let! value, ty = pRecallNode args.[1]
            let! width = match ty with TInt (IntWidth w) -> preturn w | _ -> fail (Message "MMIO write requires an integer")
            let conversion =
                if width = bits then []
                elif width < bits then [MLIROp.ArithOp (ArithOp.ExtUI (s 3, value, ty, elementTy))]
                else [MLIROp.ArithOp (ArithOp.TruncI (s 3, value, ty, elementTy))]
            let stored = if conversion.IsEmpty then value else s 3
            let unitTy = TInt (IntWidth 32)
            return conversion @ [MLIROp.MmioStore (stored, address, s 1, s 2, bits); MLIROp.ArithOp (ArithOp.ConstI (s 0, 0L, unitTy))],
                   TRValue { SSA = s 0; Type = unitTy }
}
