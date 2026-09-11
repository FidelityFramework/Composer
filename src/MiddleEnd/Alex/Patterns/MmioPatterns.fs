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
    let isReg = info.Operation.StartsWith("reg") || info.Operation.StartsWith("bind")
    let isRead = info.Operation.StartsWith("read")
    let! evidence =
        match Map.tryFind node.Id state.Graph.Codata.Value.Mmio with
        | Some evidence -> preturn evidence
        | None -> fail (Message "MMIO operation has no established CCS access evidence")
    let bits = evidence.Bits
    do! ensure (bits = 8 || bits = 16 || bits = 32) "Unsupported MMIO access width"
    if isReg then
        // ConstI carries signed bits; a high 64-bit CPU address is the same
        // pointer bit pattern after index-to-pointer conversion.
        let address = int64 (if evidence.Address > bigint System.Int64.MaxValue then evidence.Address - (1I <<< 64) else evidence.Address)
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
            // CCS established the complete source range before selecting this
            // transaction. These conversions implement that established meet.
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
