/// Application Witness - Witness function applications to MLIR
///
/// Observes application PSG nodes and generates corresponding MLIR calls.
/// Handles intrinsics, platform bindings, primitive ops, and curried calls.
/// Follows the codata/photographer principle: observe, don't compute.
module Alex.Witnesses.ApplicationWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open FSharp.Native.Compiler.Checking.Native.NativeGlobals
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes
open Alex.Patterns.SemanticPatterns
open Alex.Templates.TemplateTypes
open Alex.Templates.ArithTemplates
open Alex.Templates.MemoryTemplates

module LLVMTemplates = Alex.Templates.LLVMTemplates

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE OPERATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Try to emit a binary primitive operation (arithmetic, comparison, bitwise)
/// Uses quotation-based templates for principled MLIR generation.
let tryEmitPrimitiveBinaryOp (opName: string) (arg1SSA: string) (arg1Type: string) (arg2SSA: string) (arg2Type: string) (zipper: MLIRZipper) : (string * string * MLIRZipper) option =
    if arg1Type <> arg2Type then None
    elif not (isIntegerType arg1Type || isFloatType arg1Type) then None
    else
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let isInt = isIntegerType arg1Type
        
        // Helper to emit binary ops via templates
        let emitBinaryOp (template: MLIRTemplate<BinaryOpParams>) resultType =
            let binaryParams: BinaryOpParams = { Result = resultSSA; Lhs = arg1SSA; Rhs = arg2SSA; Type = arg1Type }
            let opText = render template binaryParams
            let zipper'' = MLIRZipper.witnessOpWithResult opText resultSSA (Serialize.deserializeType resultType) zipper'
            Some (resultSSA, resultType, zipper'')
        
        // Helper to emit comparison ops via templates
        let emitCmpOp pred isCmpF =
            let cmpParams: Quot.Compare.CmpParams = { Result = resultSSA; Predicate = pred; Lhs = arg1SSA; Rhs = arg2SSA; Type = arg1Type }
            let template = if isCmpF then Quot.Compare.cmpF else Quot.Compare.cmpI
            let opText = render template cmpParams
            let zipper'' = MLIRZipper.witnessOpWithResult opText resultSSA (Integer I1) zipper'
            Some (resultSSA, "i1", zipper'')
        
        // Select template based on operation and type
        match opName, isInt with
        // Integer arithmetic operations
        | "op_Addition", true -> emitBinaryOp Quot.IntBinary.addI arg1Type
        | "op_Subtraction", true -> emitBinaryOp Quot.IntBinary.subI arg1Type
        | "op_Multiply", true -> emitBinaryOp Quot.IntBinary.mulI arg1Type
        | "op_Division", true -> emitBinaryOp Quot.IntBinary.divSI arg1Type
        | "op_Modulus", true -> emitBinaryOp Quot.IntBinary.remSI arg1Type
        // Float arithmetic operations
        | "op_Addition", false -> emitBinaryOp Quot.FloatBinary.addF arg1Type
        | "op_Subtraction", false -> emitBinaryOp Quot.FloatBinary.subF arg1Type
        | "op_Multiply", false -> emitBinaryOp Quot.FloatBinary.mulF arg1Type
        | "op_Division", false -> emitBinaryOp Quot.FloatBinary.divF arg1Type
        // Integer comparisons
        | "op_LessThan", true -> emitCmpOp "slt" false
        | "op_LessThanOrEqual", true -> emitCmpOp "sle" false
        | "op_GreaterThan", true -> emitCmpOp "sgt" false
        | "op_GreaterThanOrEqual", true -> emitCmpOp "sge" false
        | "op_Equality", true -> emitCmpOp "eq" false
        | "op_Inequality", true -> emitCmpOp "ne" false
        // Float comparisons
        | "op_LessThan", false -> emitCmpOp "olt" true
        | "op_LessThanOrEqual", false -> emitCmpOp "ole" true
        | "op_GreaterThan", false -> emitCmpOp "ogt" true
        | "op_GreaterThanOrEqual", false -> emitCmpOp "oge" true
        | "op_Equality", false -> emitCmpOp "oeq" true
        | "op_Inequality", false -> emitCmpOp "one" true
        // Bitwise operations (int only)
        | "op_BitwiseAnd", true -> emitBinaryOp Quot.IntBitwise.andI arg1Type
        | "op_BitwiseOr", true -> emitBinaryOp Quot.IntBitwise.orI arg1Type
        | "op_ExclusiveOr", true -> emitBinaryOp Quot.IntBitwise.xorI arg1Type
        | "op_LeftShift", true -> emitBinaryOp Quot.IntBitwise.shlI arg1Type
        | "op_RightShift", true -> emitBinaryOp Quot.IntBitwise.shrSI arg1Type
        | _ -> None

/// Try to emit a unary primitive operation
/// Uses quotation-based templates for principled MLIR generation.
let tryEmitPrimitiveUnaryOp (opName: string) (argSSA: string) (argType: string) (zipper: MLIRZipper) : (string * string * MLIRZipper) option =
    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
    
    // Use active pattern for type-safe unary operation classification
    match opName with
    | UnaryOp BoolNot when isBoolType argType ->
        // Boolean NOT: XOR with true
        let trueSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let constParams: ConstantParams = { Result = trueSSA; Value = "true"; Type = "i1" }
        let trueOp = render Quot.Constant.intConst constParams
        let zipper''' = MLIRZipper.witnessOpWithResult trueOp trueSSA (Integer I1) zipper''
        let xorParams: BinaryOpParams = { Result = resultSSA; Lhs = argSSA; Rhs = trueSSA; Type = "i1" }
        let notOp = render Quot.IntBitwise.xorI xorParams
        let zipper4 = MLIRZipper.witnessOpWithResult notOp resultSSA (Integer I1) zipper'''
        Some (resultSSA, "i1", zipper4)
        
    | UnaryOp IntNegate when isIntegerType argType ->
        // Integer negation: 0 - x
        let zeroSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let constParams: ConstantParams = { Result = zeroSSA; Value = "0"; Type = argType }
        let zeroOp = render Quot.Constant.intConst constParams
        let zipper''' = MLIRZipper.witnessOpWithResult zeroOp zeroSSA (Serialize.deserializeType argType) zipper''
        let subParams: BinaryOpParams = { Result = resultSSA; Lhs = zeroSSA; Rhs = argSSA; Type = argType }
        let negOp = render Quot.IntBinary.subI subParams
        let zipper4 = MLIRZipper.witnessOpWithResult negOp resultSSA (Serialize.deserializeType argType) zipper'''
        Some (resultSSA, argType, zipper4)
        
    | UnaryOp BitwiseNot when isIntegerType argType ->
        // Bitwise NOT: XOR with -1 (all ones)
        let onesSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let constParams: ConstantParams = { Result = onesSSA; Value = "-1"; Type = argType }
        let onesOp = render Quot.Constant.intConst constParams
        let zipper''' = MLIRZipper.witnessOpWithResult onesOp onesSSA (Serialize.deserializeType argType) zipper''
        let xorParams: BinaryOpParams = { Result = resultSSA; Lhs = argSSA; Rhs = onesSSA; Type = argType }
        let notOp = render Quot.IntBitwise.xorI xorParams
        let zipper4 = MLIRZipper.witnessOpWithResult notOp resultSSA (Serialize.deserializeType argType) zipper'''
        Some (resultSSA, argType, zipper4)
        
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM BINDING DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a platform binding call
let witnessPlatformBinding (entryPoint: string) (argSSAs: (string * MLIRType) list) (returnType: NativeType) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = argSSAs
        ReturnType = mapType returnType
        BindingStrategy = Static
    }

    let zipper', result = PlatformDispatch.dispatch prim zipper

    match result with
    | WitnessedValue (ssa, ty) ->
        zipper', TRValue (ssa, Serialize.mlirType ty)
    | WitnessedVoid ->
        zipper', TRVoid
    | NotSupported reason ->
        zipper', TRError (sprintf "Platform binding '%s' not supported: %s" entryPoint reason)

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness an application and generate corresponding MLIR
let witness (funcNodeId: NodeId) (argNodeIds: NodeId list) (returnType: NativeType) (graph: SemanticGraph) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match SemanticGraph.tryGetNode funcNodeId graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.PlatformBinding entryPoint ->
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    match SemanticGraph.tryGetNode nodeId graph with
                    | Some argNode ->
                        match MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper with
                        | Some (ssa, _) -> Some (ssa, mapType argNode.Type)
                        | None -> None
                    | None -> None)
            
            let expectedParamCount =
                match entryPoint with
                | "writeBytes" | "readBytes" -> 3
                | "getCurrentTicks" -> 0
                | "sleep" -> 1
                | _ -> 0
            
            if List.length argSSAs < expectedParamCount then
                let argsEncoded = 
                    argSSAs 
                    |> List.collect (fun (ssa, ty) -> [ssa; Serialize.mlirType ty]) 
                    |> String.concat ":"
                let marker = 
                    if argsEncoded.Length > 0 then sprintf "$platform:%s:%s" entryPoint argsEncoded
                    else sprintf "$platform:%s" entryPoint
                ()
                zipper, TRValue (marker, "func")
            else
                witnessPlatformBinding entryPoint argSSAs returnType zipper

        | SemanticKind.Intrinsic intrinsicInfo ->
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)

            match intrinsicInfo, argSSAs with
            // NativePtr operations - type-safe dispatch via NativePtrOpKind
            // Uses quotation-based templates for principled MLIR generation.
            | NativePtrOp op, argSSAs ->
                match op, argSSAs with
                | PtrToNativeInt, [(argSSA, _)] ->
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let convParams: Quot.Conversion.PtrIntParams = { Result = ssaName; Operand = argSSA; IntType = "i64" }
                    let text = render Quot.Conversion.ptrToInt convParams
                    let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Integer I64) zipper'
                    zipper'', TRValue (ssaName, "i64")
                | PtrOfNativeInt, [(argSSA, _)] ->
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let convParams: Quot.Conversion.PtrIntParams = { Result = ssaName; Operand = argSSA; IntType = "i64" }
                    let text = render Quot.Conversion.intToPtr convParams
                    let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
                    zipper'', TRValue (ssaName, "!llvm.ptr")
                | PtrToVoidPtr, [(argSSA, _)] ->
                    zipper, TRValue (argSSA, "!llvm.ptr")
                | PtrOfVoidPtr, [(argSSA, _)] ->
                    zipper, TRValue (argSSA, "!llvm.ptr")
                | PtrGet, [(ptrSSA, _); (idxSSA, _)] ->
                    let elemType = Serialize.mlirType (mapType returnType)
                    let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepParams: GepParams = { Result = gepSSA; Base = ptrSSA; Offset = idxSSA; ElementType = "i8" }
                    let gepText = render Quot.Gep.i32 gepParams
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                    let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
                    let loadParams: LoadParams = { Result = loadSSA; Pointer = gepSSA; Type = elemType }
                    let loadText = render Quot.Core.load loadParams
                    let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType returnType) zipper'''
                    zipper4, TRValue (loadSSA, elemType)
                | PtrSet, [(ptrSSA, _); (idxSSA, _); (valSSA, _)] ->
                    let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepParams: GepParams = { Result = gepSSA; Base = ptrSSA; Offset = idxSSA; ElementType = "i8" }
                    let gepText = render Quot.Gep.i32 gepParams
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                    let storeParams: StoreParams = { Value = valSSA; Pointer = gepSSA; Type = "i8" }
                    let storeText = render Quot.Core.store storeParams
                    let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
                    zipper''', TRVoid
                | PtrStackAlloc, [(countSSA, _)] ->
                    let elemType =
                        match returnType with
                        | NativeType.TNativePtr elemTy -> Serialize.mlirType (mapType elemTy)
                        | _ -> "i8"
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let countSSA64, zipper'' = MLIRZipper.yieldSSA zipper'
                    let extParams: ConversionParams = { Result = countSSA64; Operand = countSSA; FromType = "i32"; ToType = "i64" }
                    let extText = render Quot.Conversion.extSI extParams
                    let zipper''' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper''
                    let allocaParams: AllocaParams = { Result = ssaName; Count = countSSA64; ElementType = elemType }
                    let allocaText = render Quot.Core.alloca allocaParams
                    let zipper4 = MLIRZipper.witnessOpWithResult allocaText ssaName Pointer zipper'''
                    zipper4, TRValue (ssaName, "!llvm.ptr")
                | PtrCopy, [(destSSA, _); (srcSSA, _); (countSSA, _)] ->
                    let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                    let extParams: ConversionParams = { Result = countSSA64; Operand = countSSA; FromType = "i32"; ToType = "i64" }
                    let extText = render Quot.Conversion.extSI extParams
                    let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                    let memcpyParams: Quot.Intrinsic.MemCopyParams = { Dest = destSSA; Src = srcSSA; Len = countSSA64 }
                    let memcpyText = render Quot.Intrinsic.memcpy memcpyParams
                    let zipper''' = MLIRZipper.witnessVoidOp memcpyText zipper''
                    zipper''', TRVoid
                | PtrFill, [(destSSA, _); (valueSSA, _); (countSSA, _)] ->
                    let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                    let extParams: ConversionParams = { Result = countSSA64; Operand = countSSA; FromType = "i32"; ToType = "i64" }
                    let extText = render Quot.Conversion.extSI extParams
                    let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                    let memsetParams: Quot.Intrinsic.MemSetParams = { Dest = destSSA; Value = valueSSA; Len = countSSA64 }
                    let memsetText = render Quot.Intrinsic.memset memsetParams
                    let zipper''' = MLIRZipper.witnessVoidOp memsetText zipper''
                    zipper''', TRVoid
                | PtrAdd, [(ptrSSA, _); (offsetSSA, _)] ->
                    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepParams: GepParams = { Result = resultSSA; Base = ptrSSA; Offset = offsetSSA; ElementType = "i8" }
                    let gepText = render Quot.Gep.i32 gepParams
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText resultSSA Pointer zipper'
                    zipper'', TRValue (resultSSA, "!llvm.ptr")
                | _, _ ->
                    zipper, TRError (sprintf "NativePtr operation arity mismatch: %A" op)

            // Sys intrinsics - direct syscall dispatch
            | SysOp opName, argSSAs ->
                let argSSAsWithTypes =
                    argSSAs |> List.map (fun (ssa, tyStr) -> (ssa, Serialize.deserializeType tyStr))
                witnessPlatformBinding opName argSSAsWithTypes returnType zipper

            // Console intrinsics - higher-level I/O that lowers to Sys calls
            | ConsoleOp "write", [(strSSA, _)] ->
                // Console.write: extract pointer/length from fat string, call Sys.write(1, ptr, len)
                let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtrParams : Quot.Aggregate.ExtractParams = { Result = ptrSSA; Aggregate = strSSA; Index = 0; AggType = NativeStrTypeStr }
                let extractPtr = render Quot.Aggregate.extractValue extractPtrParams
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr ptrSSA Pointer zipper1

                let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLenParams : Quot.Aggregate.ExtractParams = { Result = lenSSA; Aggregate = strSSA; Index = 1; AggType = NativeStrTypeStr }
                let extractLen = render Quot.Aggregate.extractValue extractLenParams
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen lenSSA (Integer I64) zipper3

                // fd = 1 (stdout)
                let fdSSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let fdParams : ConstantParams = { Result = fdSSA; Value = "1"; Type = "i32" }
                let fdText = render Quot.Constant.intConst fdParams
                let zipper6 = MLIRZipper.witnessOpWithResult fdText fdSSA (Integer I32) zipper5

                // Call Sys.write via platform binding
                let argSSAsWithTypes = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                witnessPlatformBinding "Sys.write" argSSAsWithTypes returnType zipper6

            | ConsoleOp "writeln", [(strSSA, _)] ->
                // Console.writeln: write the string, then write newline
                let ptrSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtrParams : Quot.Aggregate.ExtractParams = { Result = ptrSSA; Aggregate = strSSA; Index = 0; AggType = NativeStrTypeStr }
                let extractPtr = render Quot.Aggregate.extractValue extractPtrParams
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr ptrSSA Pointer zipper1

                let lenSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLenParams : Quot.Aggregate.ExtractParams = { Result = lenSSA; Aggregate = strSSA; Index = 1; AggType = NativeStrTypeStr }
                let extractLen = render Quot.Aggregate.extractValue extractLenParams
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen lenSSA (Integer I64) zipper3

                // fd = 1 (stdout)
                let fdSSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let fdParams : ConstantParams = { Result = fdSSA; Value = "1"; Type = "i32" }
                let fdText = render Quot.Constant.intConst fdParams
                let zipper6 = MLIRZipper.witnessOpWithResult fdText fdSSA (Integer I32) zipper5

                // Write the string
                let argSSAsWithTypes = [(fdSSA, Integer I32); (ptrSSA, Pointer); (lenSSA, Integer I64)]
                // Sys.write returns bytes written (i32 after platform binding truncation)
                let zipper7, _ = witnessPlatformBinding "Sys.write" argSSAsWithTypes Types.int32Type zipper6

                // Write newline: allocate newline char on stack, write it
                let nlSSA, zipper8 = MLIRZipper.yieldSSA zipper7
                let nlParams : ConstantParams = { Result = nlSSA; Value = "10"; Type = "i8" }  // '\n' = 10
                let nlText = render Quot.Constant.intConst nlParams
                let zipper9 = MLIRZipper.witnessOpWithResult nlText nlSSA (Integer I8) zipper8

                let oneSSA, zipper10 = MLIRZipper.yieldSSA zipper9
                let oneParams : ConstantParams = { Result = oneSSA; Value = "1"; Type = "i64" }
                let oneText = render Quot.Constant.intConst oneParams
                let zipper11 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) zipper10

                let nlBufSSA, zipper12 = MLIRZipper.yieldSSA zipper11
                let allocaParams : AllocaParams = { Result = nlBufSSA; Count = oneSSA; ElementType = "i8" }
                let allocaText = render Quot.Core.alloca allocaParams
                let zipper13 = MLIRZipper.witnessOpWithResult allocaText nlBufSSA Pointer zipper12

                let storeParams : StoreParams = { Value = nlSSA; Pointer = nlBufSSA; Type = "i8" }
                let storeText = render Quot.Core.store storeParams
                let zipper14 = MLIRZipper.witnessVoidOp storeText zipper13

                // Write the newline
                let nlArgSSAs = [(fdSSA, Integer I32); (nlBufSSA, Pointer); (oneSSA, Integer I64)]
                witnessPlatformBinding "Sys.write" nlArgSSAs returnType zipper14

            | ConsoleOp "readln", ([] | [_]) ->  // Takes unit arg (or elided)
                // Console.readln: read a line from stdin into a buffer, return as string
                // For now, allocate a fixed buffer and read into it
                let bufSizeSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let bufSizeParams : ConstantParams = { Result = bufSizeSSA; Value = "256"; Type = "i64" }
                let bufSizeText = render Quot.Constant.intConst bufSizeParams
                let zipper2 = MLIRZipper.witnessOpWithResult bufSizeText bufSizeSSA (Integer I64) zipper1

                let bufSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let allocaParams : AllocaParams = { Result = bufSSA; Count = bufSizeSSA; ElementType = "i8" }
                let allocaText = render Quot.Core.alloca allocaParams
                let zipper4 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer zipper3

                // fd = 0 (stdin)
                let fdSSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let fdParams : ConstantParams = { Result = fdSSA; Value = "0"; Type = "i32" }
                let fdText = render Quot.Constant.intConst fdParams
                let zipper6 = MLIRZipper.witnessOpWithResult fdText fdSSA (Integer I32) zipper5

                // Call Sys.read
                let argSSAsWithTypes = [(fdSSA, Integer I32); (bufSSA, Pointer); (bufSizeSSA, Integer I64)]
                // Sys.read returns bytes read (i32 after platform binding truncation)
                let zipper7, readResult = witnessPlatformBinding "Sys.read" argSSAsWithTypes Types.int32Type zipper6

                // Get bytes read (strip newline)
                // Platform binding returns i32, extend to i64 for length arithmetic
                let bytesReadSSA32 = match readResult with TRValue (ssa, _) -> ssa | _ -> "%err"
                let bytesReadSSA, zipper8 = MLIRZipper.yieldSSA zipper7
                let extParams : ConversionParams = { Result = bytesReadSSA; Operand = bytesReadSSA32; FromType = "i32"; ToType = "i64" }
                let extText = render Quot.Conversion.extSI extParams
                let zipper9 = MLIRZipper.witnessOpWithResult extText bytesReadSSA (Integer I64) zipper8

                let oneSSA, zipper10 = MLIRZipper.yieldSSA zipper9
                let oneParams : ConstantParams = { Result = oneSSA; Value = "1"; Type = "i64" }
                let oneText = render Quot.Constant.intConst oneParams
                let zipper11 = MLIRZipper.witnessOpWithResult oneText oneSSA (Integer I64) zipper10

                let lenSSA, zipper12 = MLIRZipper.yieldSSA zipper11
                let subParams : BinaryOpParams = { Result = lenSSA; Lhs = bytesReadSSA; Rhs = oneSSA; Type = "i64" }
                let subText = render Quot.IntBinary.subI subParams
                let zipper13 = MLIRZipper.witnessOpWithResult subText lenSSA (Integer I64) zipper12

                // Build fat string struct
                let undefSSA, zipper14 = MLIRZipper.yieldSSA zipper13
                let undefText = render Quot.Aggregate.undef {| Result = undefSSA; Type = NativeStrTypeStr |}
                let zipper15 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper14

                let withPtrSSA, zipper16 = MLIRZipper.yieldSSA zipper15
                let insertPtrParams : Quot.Aggregate.InsertParams = { Result = withPtrSSA; Value = bufSSA; Aggregate = undefSSA; Index = 0; AggType = NativeStrTypeStr }
                let insertPtrText = render Quot.Aggregate.insertValue insertPtrParams
                let zipper17 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper16

                let fatPtrSSA, zipper18 = MLIRZipper.yieldSSA zipper17
                let insertLenParams : Quot.Aggregate.InsertParams = { Result = fatPtrSSA; Value = lenSSA; Aggregate = withPtrSSA; Index = 1; AggType = NativeStrTypeStr }
                let insertLenText = render Quot.Aggregate.insertValue insertLenParams
                let zipper19 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper18

                zipper19, TRValue (fatPtrSSA, NativeStrTypeStr)

            | NativeStrOp "fromPointer", [(ptrSSA, _); (lenSSA, _)] ->
                let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let undefText = render Quot.Aggregate.undef {| Result = undefSSA; Type = NativeStrTypeStr |}
                let zipper2 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper1

                let withPtrSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let insertPtrParams : Quot.Aggregate.InsertParams = { Result = withPtrSSA; Value = ptrSSA; Aggregate = undefSSA; Index = 0; AggType = NativeStrTypeStr }
                let insertPtrText = render Quot.Aggregate.insertValue insertPtrParams
                let zipper4 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper3

                let lenSSA64, zipper5 = MLIRZipper.yieldSSA zipper4
                let extParams : ConversionParams = { Result = lenSSA64; Operand = lenSSA; FromType = "i32"; ToType = "i64" }
                let extText = render Quot.Conversion.extSI extParams
                let zipper6 = MLIRZipper.witnessOpWithResult extText lenSSA64 (Integer I64) zipper5

                let fatPtrSSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let insertLenParams : Quot.Aggregate.InsertParams = { Result = fatPtrSSA; Value = lenSSA64; Aggregate = withPtrSSA; Index = 1; AggType = NativeStrTypeStr }
                let insertLenText = render Quot.Aggregate.insertValue insertLenParams
                let zipper8 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper7

                zipper8, TRValue (fatPtrSSA, NativeStrTypeStr)

            | NativeDefaultOp "zeroed", [] ->
                let zeroSSA, zipper' = MLIRZipper.yieldSSA zipper
                let mlirRetType = mapType returnType
                let mlirTypeStr = Serialize.mlirType mlirRetType
                let zeroText =
                    match mlirRetType with
                    | Integer _ ->
                        let constParams : ConstantParams = { Result = zeroSSA; Value = "0"; Type = mlirTypeStr }
                        render Quot.Constant.intConst constParams
                    | Float F32 ->
                        let floatParams : ConstantParams = { Result = zeroSSA; Value = "0.0"; Type = "f32" }
                        render Quot.Constant.floatConst floatParams
                    | Float F64 ->
                        let floatParams : ConstantParams = { Result = zeroSSA; Value = "0.0"; Type = "f64" }
                        render Quot.Constant.floatConst floatParams
                    | Pointer ->
                        render LLVMTemplates.Quot.Global.zeroInit {| Result = zeroSSA; Type = "!llvm.ptr" |}
                    | Struct _ when mlirTypeStr = NativeStrTypeStr ->
                        render Quot.Aggregate.undef {| Result = zeroSSA; Type = NativeStrTypeStr |}
                    | Struct _ ->
                        render Quot.Aggregate.undef {| Result = zeroSSA; Type = mlirTypeStr |}
                    | _ ->
                        render Quot.Aggregate.undef {| Result = zeroSSA; Type = mlirTypeStr |}
                let zipper'' = MLIRZipper.witnessOpWithResult zeroText zeroSSA mlirRetType zipper'
                zipper'', TRValue (zeroSSA, mlirTypeStr)

            | StringOp "concat2", [(str1SSA, _); (str2SSA, _)] ->
                // Extract ptr and len from first string
                let ptr1SSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtr1Params : Quot.Aggregate.ExtractParams = { Result = ptr1SSA; Aggregate = str1SSA; Index = 0; AggType = NativeStrTypeStr }
                let extractPtr1 = render Quot.Aggregate.extractValue extractPtr1Params
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr1 ptr1SSA Pointer zipper1

                let len1SSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLen1Params : Quot.Aggregate.ExtractParams = { Result = len1SSA; Aggregate = str1SSA; Index = 1; AggType = NativeStrTypeStr }
                let extractLen1 = render Quot.Aggregate.extractValue extractLen1Params
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen1 len1SSA (Integer I64) zipper3

                // Extract ptr and len from second string
                let ptr2SSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let extractPtr2Params : Quot.Aggregate.ExtractParams = { Result = ptr2SSA; Aggregate = str2SSA; Index = 0; AggType = NativeStrTypeStr }
                let extractPtr2 = render Quot.Aggregate.extractValue extractPtr2Params
                let zipper6 = MLIRZipper.witnessOpWithResult extractPtr2 ptr2SSA Pointer zipper5

                let len2SSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let extractLen2Params : Quot.Aggregate.ExtractParams = { Result = len2SSA; Aggregate = str2SSA; Index = 1; AggType = NativeStrTypeStr }
                let extractLen2 = render Quot.Aggregate.extractValue extractLen2Params
                let zipper8 = MLIRZipper.witnessOpWithResult extractLen2 len2SSA (Integer I64) zipper7

                // Compute total length
                let totalLenSSA, zipper9 = MLIRZipper.yieldSSA zipper8
                let addParams : BinaryOpParams = { Result = totalLenSSA; Lhs = len1SSA; Rhs = len2SSA; Type = "i64" }
                let addLenText = render Quot.IntBinary.addI addParams
                let zipper10 = MLIRZipper.witnessOpWithResult addLenText totalLenSSA (Integer I64) zipper9

                // Allocate buffer for concatenated string
                let bufSSA, zipper11 = MLIRZipper.yieldSSA zipper10
                let allocaParams : AllocaParams = { Result = bufSSA; Count = totalLenSSA; ElementType = "i8" }
                let allocaText = render Quot.Core.alloca allocaParams
                let zipper12 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer zipper11

                // Copy first string
                let memcpy1Params : Quot.Intrinsic.MemCopyParams = { Dest = bufSSA; Src = ptr1SSA; Len = len1SSA }
                let memcpy1 = render Quot.Intrinsic.memcpy memcpy1Params
                let zipper13 = MLIRZipper.witnessVoidOp memcpy1 zipper12

                // GEP to offset position for second string
                let offsetSSA, zipper14 = MLIRZipper.yieldSSA zipper13
                let gepParams : GepParams = { Result = offsetSSA; Base = bufSSA; Offset = len1SSA; ElementType = "i8" }
                let gepText = render Quot.Gep.i64 gepParams
                let zipper15 = MLIRZipper.witnessOpWithResult gepText offsetSSA Pointer zipper14

                // Copy second string
                let memcpy2Params : Quot.Intrinsic.MemCopyParams = { Dest = offsetSSA; Src = ptr2SSA; Len = len2SSA }
                let memcpy2 = render Quot.Intrinsic.memcpy memcpy2Params
                let zipper16 = MLIRZipper.witnessVoidOp memcpy2 zipper15

                // Build result fat string struct
                let undefSSA, zipper17 = MLIRZipper.yieldSSA zipper16
                let undefText = render Quot.Aggregate.undef {| Result = undefSSA; Type = NativeStrTypeStr |}
                let zipper18 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper17

                let withPtrSSA, zipper19 = MLIRZipper.yieldSSA zipper18
                let insertPtrParams : Quot.Aggregate.InsertParams = { Result = withPtrSSA; Value = bufSSA; Aggregate = undefSSA; Index = 0; AggType = NativeStrTypeStr }
                let insertPtrText = render Quot.Aggregate.insertValue insertPtrParams
                let zipper20 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper19

                let resultSSA, zipper21 = MLIRZipper.yieldSSA zipper20
                let insertLenParams : Quot.Aggregate.InsertParams = { Result = resultSSA; Value = totalLenSSA; Aggregate = withPtrSSA; Index = 1; AggType = NativeStrTypeStr }
                let insertLenText = render Quot.Aggregate.insertValue insertLenParams
                let zipper22 = MLIRZipper.witnessOpWithResult insertLenText resultSSA NativeStrType zipper21

                zipper22, TRValue (resultSSA, NativeStrTypeStr)

            | info, [] ->
                zipper, TRValue ("$intrinsic:" + info.FullName, "func")
            | info, _ ->
                zipper, TRError (sprintf "Unknown intrinsic: %s with %d args" info.FullName (List.length argSSAs))

        | SemanticKind.VarRef (name, defId) ->
            let argSSAsAndTypes =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)
            let argSSAs = argSSAsAndTypes |> List.map fst
            let argTypes = argSSAsAndTypes |> List.map (fun (_, ty) -> Serialize.deserializeType ty)

            if name = "op_PipeRight" || name = "op_PipeLeft" then
                match argSSAs with
                | [argSSA] ->
                    let (argType: string) = argSSAsAndTypes |> List.head |> snd
                    let marker = sprintf "$pipe:%s:%s" argSSA argType
                    zipper, TRValue (marker, "func")
                | _ ->
                    zipper, TRError (sprintf "Pipe operator '%s' expects 1 argument, got %d" name (List.length argSSAs))
            else

            match defId with
            | Some defNodeId ->
                match MLIRZipper.recallNodeSSA (string (NodeId.value defNodeId)) zipper with
                | Some (funcSSA, _funcType) ->
                    if funcSSA.StartsWith("@") then
                        let funcName = funcSSA.Substring(1)
                        
                        let expectedParams = MLIRZipper.lookupFuncParamCount funcName zipper
                        match expectedParams with
                        | Some paramCount when paramCount > List.length argSSAs ->
                            let argPairs = List.zip argSSAs (argTypes |> List.map Serialize.mlirType)
                            let argsEncoded = argPairs |> List.collect (fun (a, t) -> [a; t]) |> String.concat ":"
                            let marker = sprintf "$partial:%s:%s" funcName argsEncoded
                            zipper, TRValue (marker, "func")
                        | _ ->
                            let ssaName, zipper' = MLIRZipper.witnessCall funcName argSSAs argTypes (mapType returnType) zipper
                            let actualRetType =
                                match MLIRZipper.lookupFuncReturnType funcName zipper' with
                                | Some t -> t
                                | None -> Serialize.mlirType (mapType returnType)
                            zipper', TRValue (ssaName, actualRetType)
                    else
                        let ssaName, zipper' = MLIRZipper.witnessIndirectCall funcSSA argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | None ->
                    let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
            | None ->
                let argSSAsWithTypes = List.zip argSSAs (argTypes |> List.map Serialize.mlirType)
                
                match argSSAsWithTypes with
                | [(arg1SSA, arg1Type); (arg2SSA, arg2Type)] ->
                    match tryEmitPrimitiveBinaryOp name arg1SSA arg1Type arg2SSA arg2Type zipper with
                    | Some (resultSSA, resultType, zipper') ->
                        zipper', TRValue (resultSSA, resultType)
                    | None ->
                        let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | [(argSSA, argType)] ->
                    match tryEmitPrimitiveUnaryOp name argSSA argType zipper with
                    | Some (resultSSA, resultType, zipper') ->
                        zipper', TRValue (resultSSA, resultType)
                    | None ->
                        let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                | _ ->
                    let ssaName, zipper' = MLIRZipper.witnessCall name argSSAs argTypes (mapType returnType) zipper
                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))

        | SemanticKind.Lambda _ ->
            zipper, TRError "Lambda application not yet supported"

        | SemanticKind.Application (innerFuncId, innerArgIds) ->
            match MLIRZipper.recallNodeSSA (string (NodeId.value funcNodeId)) zipper with
            | Some (funcSSA, _funcType) ->
                let argSSAsAndTypes =
                    argNodeIds
                    |> List.choose (fun nodeId ->
                        MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)
                let argSSAs = argSSAsAndTypes |> List.map fst
                let argTypes = argSSAsAndTypes |> List.map (fun (_, ty) -> Serialize.deserializeType ty)

                if funcSSA.StartsWith("$pipe:") then
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 3 then
                        let pipedArgSSA = parts.[1]
                        let pipedArgType = parts.[2]
                        match argSSAs with
                        | [fSSA] ->
                            let retTypeStr = Serialize.mlirType (mapType returnType)
                            if retTypeStr = "i32" && pipedArgType = "i32" then
                                let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                let unitParams : ConstantParams = { Result = unitSSA; Value = "0"; Type = "i32" }
                                let unitText = render Quot.Constant.intConst unitParams
                                let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                zipper'', TRValue (unitSSA, "i32")
                            else
                                let pipedTypes = [Serialize.deserializeType pipedArgType]
                                if fSSA.StartsWith("@") then
                                    let funcName = fSSA.Substring(1)
                                    if funcName = "ignore" then
                                        let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                        let unitParams : ConstantParams = { Result = unitSSA; Value = "0"; Type = "i32" }
                                        let unitText = render Quot.Constant.intConst unitParams
                                        let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                        zipper'', TRValue (unitSSA, "i32")
                                    else
                                        let ssaName, zipper' = MLIRZipper.witnessCall funcName [pipedArgSSA] pipedTypes (mapType returnType) zipper
                                        zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                                else
                                    let ssaName, zipper' = MLIRZipper.witnessIndirectCall fSSA [pipedArgSSA] pipedTypes (mapType returnType) zipper
                                    zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
                        | _ ->
                            zipper, TRError (sprintf "Pipe application expected 1 function arg, got %d" (List.length argSSAs))
                    else
                        zipper, TRError (sprintf "Invalid pipe marker: %s" funcSSA)
                elif funcSSA.StartsWith("$partial:") then
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 2 then
                        let funcName = parts.[1]
                        let appliedArgs = 
                            parts 
                            |> Array.skip 2 
                            |> Array.chunkBySize 2 
                            |> Array.choose (function
                                | [| arg; ty |] -> Some (arg, ty)
                                | _ -> None)
                            |> Array.toList
                        let allArgSSAs = (appliedArgs |> List.map fst) @ argSSAs
                        let allArgTypes = (appliedArgs |> List.map (snd >> Serialize.deserializeType)) @ argTypes
                        
                        match MLIRZipper.lookupFuncParamCount funcName zipper with
                        | Some paramCount when paramCount > List.length allArgSSAs ->
                            let argPairs = List.zip allArgSSAs (allArgTypes |> List.map Serialize.mlirType)
                            let argsEncoded = argPairs |> List.collect (fun (a, t) -> [a; t]) |> String.concat ":"
                            let marker = sprintf "$partial:%s:%s" funcName argsEncoded
                            zipper, TRValue (marker, "func")
                        | _ ->
                            let ssaName, zipper' = MLIRZipper.witnessCall funcName allArgSSAs allArgTypes (mapType returnType) zipper
                            let actualRetType =
                                match MLIRZipper.lookupFuncReturnType funcName zipper' with
                                | Some t -> t
                                | None -> Serialize.mlirType (mapType returnType)
                            zipper', TRValue (ssaName, actualRetType)
                    else
                        zipper, TRError (sprintf "Invalid partial marker: %s" funcSSA)
                elif funcSSA.StartsWith("$platform:") then
                    ()
                    let parts = funcSSA.Split(':')
                    if parts.Length >= 2 then
                        let entryPoint = parts.[1]
                        let appliedArgs = 
                            parts 
                            |> Array.skip 2 
                            |> Array.chunkBySize 2 
                            |> Array.choose (function
                                | [| arg; ty |] -> Some (arg, Serialize.deserializeType ty)
                                | _ -> None)
                            |> Array.toList
                        let allArgSSAs = appliedArgs @ (List.zip argSSAs argTypes)
                        
                        let expectedParamCount = 
                            match entryPoint with
                            | "writeBytes" | "readBytes" -> 3
                            | "getCurrentTicks" -> 0
                            | "sleep" -> 1
                            | _ -> List.length allArgSSAs
                        
                        if List.length allArgSSAs < expectedParamCount then
                            let argsEncoded = allArgSSAs |> List.collect (fun (a, t) -> [a; Serialize.mlirType t]) |> String.concat ":"
                            let marker = sprintf "$platform:%s:%s" entryPoint argsEncoded
                            zipper, TRValue (marker, "func")
                        else
                            witnessPlatformBinding entryPoint allArgSSAs returnType zipper
                    else
                        zipper, TRError (sprintf "Invalid platform marker: %s" funcSSA)
                else

                let ssaName, zipper' = MLIRZipper.witnessIndirectCall funcSSA argSSAs argTypes (mapType returnType) zipper
                zipper', TRValue (ssaName, Serialize.mlirType (mapType returnType))
            | None ->
                zipper, TRError (sprintf "Curried function application not computed: %A" (innerFuncId, innerArgIds))

        | _ ->
            zipper, TRError (sprintf "Unexpected function node kind: %A" funcNode.Kind)

    | None ->
        zipper, TRError (sprintf "Function node not found: %d" (NodeId.value funcNodeId))
