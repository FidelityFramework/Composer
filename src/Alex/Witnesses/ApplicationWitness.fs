/// Application Witness - Witness function applications to MLIR
///
/// Observes application PSG nodes and generates corresponding MLIR calls.
/// Handles intrinsics, platform bindings, primitive ops, and curried calls.
/// Follows the codata/photographer principle: observe, don't compute.
module Alex.Witnesses.ApplicationWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Bindings.BindingTypes
open Alex.Patterns.SemanticPatterns

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// PRIMITIVE OPERATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Try to emit a binary primitive operation (arithmetic, comparison, bitwise)
let tryEmitPrimitiveBinaryOp (opName: string) (arg1SSA: string) (arg1Type: string) (arg2SSA: string) (arg2Type: string) (zipper: MLIRZipper) : (string * string * MLIRZipper) option =
    if arg1Type <> arg2Type then None
    elif not (isIntegerType arg1Type || isFloatType arg1Type) then None
    else
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let isInt = isIntegerType arg1Type
        
        // Use active patterns for type-safe operation classification
        let emitOp mlirOp resultType =
            let opText = sprintf "%s = %s %s, %s : %s" resultSSA mlirOp arg1SSA arg2SSA arg1Type
            let zipper'' = MLIRZipper.witnessOpWithResult opText resultSSA (Serialize.deserializeType resultType) zipper'
            Some (resultSSA, resultType, zipper'')
        
        let emitCmp mlirOp pred =
            let opText = sprintf "%s = %s %s, %s, %s : %s" resultSSA mlirOp pred arg1SSA arg2SSA arg1Type
            let zipper'' = MLIRZipper.witnessOpWithResult opText resultSSA (Integer I1) zipper'
            Some (resultSSA, "i1", zipper'')
        
        match opName, isInt with
        // Arithmetic operations via ArithBinaryOp pattern
        | ArithBinaryOp (mlirOp, _) -> emitOp mlirOp arg1Type
        // Comparison operations via CmpBinaryOp pattern
        | CmpBinaryOp (mlirOp, pred) -> emitCmp mlirOp pred
        // Bitwise operations (int only) via BitwiseBinaryOp pattern
        | _ when isInt ->
            match opName with
            | BitwiseBinaryOp mlirOp -> emitOp mlirOp arg1Type
            | _ -> None
        | _ -> None

/// Try to emit a unary primitive operation
let tryEmitPrimitiveUnaryOp (opName: string) (argSSA: string) (argType: string) (zipper: MLIRZipper) : (string * string * MLIRZipper) option =
    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
    
    // Use active pattern for type-safe unary operation classification
    match opName with
    | UnaryOp BoolNot when isBoolType argType ->
        let trueSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let trueOp = sprintf "%s = arith.constant true" trueSSA
        let zipper''' = MLIRZipper.witnessOpWithResult trueOp trueSSA (Integer I1) zipper''
        let notOp = sprintf "%s = arith.xori %s, %s : i1" resultSSA argSSA trueSSA
        let zipper4 = MLIRZipper.witnessOpWithResult notOp resultSSA (Integer I1) zipper'''
        Some (resultSSA, "i1", zipper4)
        
    | UnaryOp IntNegate when isIntegerType argType ->
        let zeroSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let zeroOp = sprintf "%s = arith.constant 0 : %s" zeroSSA argType
        let zipper''' = MLIRZipper.witnessOpWithResult zeroOp zeroSSA (Serialize.deserializeType argType) zipper''
        let negOp = sprintf "%s = arith.subi %s, %s : %s" resultSSA zeroSSA argSSA argType
        let zipper4 = MLIRZipper.witnessOpWithResult negOp resultSSA (Serialize.deserializeType argType) zipper'''
        Some (resultSSA, argType, zipper4)
        
    | UnaryOp BitwiseNot when isIntegerType argType ->
        let onesSSA, zipper'' = MLIRZipper.yieldSSA zipper'
        let onesOp = sprintf "%s = arith.constant -1 : %s" onesSSA argType
        let zipper''' = MLIRZipper.witnessOpWithResult onesOp onesSSA (Serialize.deserializeType argType) zipper''
        let notOp = sprintf "%s = arith.xori %s, %s : %s" resultSSA argSSA onesSSA argType
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

        | SemanticKind.Intrinsic intrinsicName ->
            let argSSAs =
                argNodeIds
                |> List.choose (fun nodeId ->
                    MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) zipper)

            match intrinsicName, argSSAs with
            // NativePtr operations - type-safe dispatch via NativePtrOpKind
            | NativePtrOp op, argSSAs ->
                match op, argSSAs with
                | PtrToNativeInt, [(argSSA, _)] ->
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let text = sprintf "%s = llvm.ptrtoint %s : !llvm.ptr to i64" ssaName argSSA
                    let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Integer I64) zipper'
                    zipper'', TRValue (ssaName, "i64")
                | PtrOfNativeInt, [(argSSA, _)] ->
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let text = sprintf "%s = llvm.inttoptr %s : i64 to !llvm.ptr" ssaName argSSA
                    let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
                    zipper'', TRValue (ssaName, "!llvm.ptr")
                | PtrToVoidPtr, [(argSSA, _)] ->
                    zipper, TRValue (argSSA, "!llvm.ptr")
                | PtrOfVoidPtr, [(argSSA, _)] ->
                    zipper, TRValue (argSSA, "!llvm.ptr")
                | PtrGet, [(ptrSSA, _); (idxSSA, _)] ->
                    let elemType = Serialize.mlirType (mapType returnType)
                    let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" gepSSA ptrSSA idxSSA
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                    let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
                    let loadText = sprintf "%s = llvm.load %s : !llvm.ptr -> %s" loadSSA gepSSA elemType
                    let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType returnType) zipper'''
                    zipper4, TRValue (loadSSA, elemType)
                | PtrSet, [(ptrSSA, _); (idxSSA, _); (valSSA, _)] ->
                    let gepSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" gepSSA ptrSSA idxSSA
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText gepSSA Pointer zipper'
                    let storeText = sprintf "llvm.store %s, %s : i8, !llvm.ptr" valSSA gepSSA
                    let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
                    zipper''', TRVoid
                | PtrStackAlloc, [(countSSA, _)] ->
                    let elemType =
                        match returnType with
                        | NativeType.TNativePtr elemTy -> Serialize.mlirType (mapType elemTy)
                        | _ -> "i8"
                    let ssaName, zipper' = MLIRZipper.yieldSSA zipper
                    let countSSA64, zipper'' = MLIRZipper.yieldSSA zipper'
                    let extText = sprintf "%s = arith.extsi %s : i32 to i64" countSSA64 countSSA
                    let zipper''' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper''
                    let allocaText = sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" ssaName countSSA64 elemType
                    let zipper4 = MLIRZipper.witnessOpWithResult allocaText ssaName Pointer zipper'''
                    zipper4, TRValue (ssaName, "!llvm.ptr")
                | PtrCopy, [(destSSA, _); (srcSSA, _); (countSSA, _)] ->
                    let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                    let extText = sprintf "%s = arith.extsi %s : i32 to i64" countSSA64 countSSA
                    let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                    let memcpyText = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" destSSA srcSSA countSSA64
                    let zipper''' = MLIRZipper.witnessVoidOp memcpyText zipper''
                    zipper''', TRVoid
                | PtrFill, [(destSSA, _); (valueSSA, _); (countSSA, _)] ->
                    let countSSA64, zipper' = MLIRZipper.yieldSSA zipper
                    let extText = sprintf "%s = arith.extsi %s : i32 to i64" countSSA64 countSSA
                    let zipper'' = MLIRZipper.witnessOpWithResult extText countSSA64 (Integer I64) zipper'
                    let memsetText = sprintf "\"llvm.intr.memset\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()" destSSA valueSSA countSSA64
                    let zipper''' = MLIRZipper.witnessVoidOp memsetText zipper''
                    zipper''', TRVoid
                | PtrAdd, [(ptrSSA, _); (offsetSSA, _)] ->
                    let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                    let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, i8" resultSSA ptrSSA offsetSSA
                    let zipper'' = MLIRZipper.witnessOpWithResult gepText resultSSA Pointer zipper'
                    zipper'', TRValue (resultSSA, "!llvm.ptr")
                | _, _ ->
                    zipper, TRError (sprintf "NativePtr operation arity mismatch: %A" op)

            // Platform intrinsics (Sys.*, Console.*, etc.)
            | (intrinsicName & PlatformIntrinsic _), argSSAs ->
                let argSSAsWithTypes =
                    argSSAs |> List.map (fun (ssa, tyStr) -> (ssa, Serialize.deserializeType tyStr))
                witnessPlatformBinding intrinsicName argSSAsWithTypes returnType zipper

            | "NativeStr.fromPointer", [(ptrSSA, _); (lenSSA, _)] ->
                let undefSSA, zipper1 = MLIRZipper.yieldSSA zipper
                let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
                let zipper2 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper1

                let withPtrSSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA NativeStrTypeStr
                let zipper4 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper3

                let lenSSA64, zipper5 = MLIRZipper.yieldSSA zipper4
                let extText = sprintf "%s = arith.extsi %s : i32 to i64" lenSSA64 lenSSA
                let zipper6 = MLIRZipper.witnessOpWithResult extText lenSSA64 (Integer I64) zipper5

                let fatPtrSSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" fatPtrSSA lenSSA64 withPtrSSA NativeStrTypeStr
                let zipper8 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper7

                zipper8, TRValue (fatPtrSSA, NativeStrTypeStr)

            | "NativeDefault.zeroed", [] ->
                let zeroSSA, zipper' = MLIRZipper.yieldSSA zipper
                let mlirRetType = mapType returnType
                let mlirTypeStr = Serialize.mlirType mlirRetType
                let zeroText =
                    match mlirRetType with
                    | Integer _ -> sprintf "%s = arith.constant 0 : %s" zeroSSA mlirTypeStr
                    | Float F32 -> sprintf "%s = arith.constant 0.0 : f32" zeroSSA
                    | Float F64 -> sprintf "%s = arith.constant 0.0 : f64" zeroSSA
                    | Pointer -> sprintf "%s = llvm.mlir.zero : !llvm.ptr" zeroSSA
                    | Struct _ when mlirTypeStr = NativeStrTypeStr ->
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA NativeStrTypeStr
                    | Struct _ ->
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA mlirTypeStr
                    | _ ->
                        sprintf "%s = llvm.mlir.undef : %s" zeroSSA mlirTypeStr
                let zipper'' = MLIRZipper.witnessOpWithResult zeroText zeroSSA mlirRetType zipper'
                zipper'', TRValue (zeroSSA, mlirTypeStr)

            | "String.concat2", [(str1SSA, _); (str2SSA, _)] ->
                let ptr1SSA, zipper1 = MLIRZipper.yieldSSA zipper
                let extractPtr1 = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptr1SSA str1SSA
                let zipper2 = MLIRZipper.witnessOpWithResult extractPtr1 ptr1SSA Pointer zipper1

                let len1SSA, zipper3 = MLIRZipper.yieldSSA zipper2
                let extractLen1 = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" len1SSA str1SSA
                let zipper4 = MLIRZipper.witnessOpWithResult extractLen1 len1SSA (Integer I64) zipper3

                let ptr2SSA, zipper5 = MLIRZipper.yieldSSA zipper4
                let extractPtr2 = sprintf "%s = llvm.extractvalue %s[0] : !llvm.struct<(ptr, i64)>" ptr2SSA str2SSA
                let zipper6 = MLIRZipper.witnessOpWithResult extractPtr2 ptr2SSA Pointer zipper5

                let len2SSA, zipper7 = MLIRZipper.yieldSSA zipper6
                let extractLen2 = sprintf "%s = llvm.extractvalue %s[1] : !llvm.struct<(ptr, i64)>" len2SSA str2SSA
                let zipper8 = MLIRZipper.witnessOpWithResult extractLen2 len2SSA (Integer I64) zipper7

                let totalLenSSA, zipper9 = MLIRZipper.yieldSSA zipper8
                let addLenText = sprintf "%s = arith.addi %s, %s : i64" totalLenSSA len1SSA len2SSA
                let zipper10 = MLIRZipper.witnessOpWithResult addLenText totalLenSSA (Integer I64) zipper9

                let bufSSA, zipper11 = MLIRZipper.yieldSSA zipper10
                let allocaText = sprintf "%s = llvm.alloca %s x i8 : (i64) -> !llvm.ptr" bufSSA totalLenSSA
                let zipper12 = MLIRZipper.witnessOpWithResult allocaText bufSSA Pointer zipper11

                let memcpy1 = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" bufSSA ptr1SSA len1SSA
                let zipper13 = MLIRZipper.witnessVoidOp memcpy1 zipper12

                let offsetSSA, zipper14 = MLIRZipper.yieldSSA zipper13
                let gepText = sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, i8" offsetSSA bufSSA len1SSA
                let zipper15 = MLIRZipper.witnessOpWithResult gepText offsetSSA Pointer zipper14

                let memcpy2 = sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()" offsetSSA ptr2SSA len2SSA
                let zipper16 = MLIRZipper.witnessVoidOp memcpy2 zipper15

                let undefSSA, zipper17 = MLIRZipper.yieldSSA zipper16
                let undefText = sprintf "%s = llvm.mlir.undef : !llvm.struct<(ptr, i64)>" undefSSA
                let zipper18 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper17

                let withPtrSSA, zipper19 = MLIRZipper.yieldSSA zipper18
                let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : !llvm.struct<(ptr, i64)>" withPtrSSA bufSSA undefSSA
                let zipper20 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper19

                let resultSSA, zipper21 = MLIRZipper.yieldSSA zipper20
                let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : !llvm.struct<(ptr, i64)>" resultSSA totalLenSSA withPtrSSA
                let zipper22 = MLIRZipper.witnessOpWithResult insertLenText resultSSA NativeStrType zipper21

                zipper22, TRValue (resultSSA, NativeStrTypeStr)

            | intrinsicName, [] ->
                zipper, TRValue ("$intrinsic:" + intrinsicName, "func")
            | _ ->
                zipper, TRError (sprintf "Unknown intrinsic: %s with %d args" intrinsicName (List.length argSSAs))

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
                                let unitText = sprintf "%s = arith.constant 0 : i32" unitSSA
                                let zipper'' = MLIRZipper.witnessOpWithResult unitText unitSSA (Integer I32) zipper'
                                zipper'', TRValue (unitSSA, "i32")
                            else
                                let pipedTypes = [Serialize.deserializeType pipedArgType]
                                if fSSA.StartsWith("@") then
                                    let funcName = fSSA.Substring(1)
                                    if funcName = "ignore" then
                                        let unitSSA, zipper' = MLIRZipper.yieldSSA zipper
                                        let unitText = sprintf "%s = arith.constant 0 : i32" unitSSA
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
