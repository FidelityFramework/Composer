/// MLIR-AIE target realization of the complete source-published kernel plan.
module BackEnd.AIE.KernelRealization

open System
open System.Text
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Core.Types.Pipeline

let private require condition reason = if not condition then invalidOp reason
let private scalar slot =
    match SettledScalar.tryType slot with
    | Some(TInt(IntWidth bits)) when bits>0 -> $"i{bits}"
    | _ -> invalidOp "AIE scalar construction has no integer realization for the published slot."
let private stepSite = function
    | KernelScalarStep.Parameter(site,_,_) | KernelScalarStep.Literal(site,_,_)
    | KernelScalarStep.Alias(site,_,_,_) -> site
    | KernelScalarStep.Operation operation -> operation.Site
let private stepCarrier = function
    | KernelScalarStep.Parameter(_,_,carrier) | KernelScalarStep.Literal(_,_,carrier)
    | KernelScalarStep.Alias(_,_,carrier,_) -> carrier
    | KernelScalarStep.Operation operation -> operation.Result
let private symbol (plan: KernelModuleWitness) = $"__clef_kernel_{NodeId.value plan.Site}"
let private transportType (transport: KernelTransport) =
    let representation=transport.Representation
    require (representation.Bits>0 && representation.Bits%8=0 && (representation.Family="int" || representation.Family="uint"))
            "AIE external transport requires a declared whole-byte integer encoding."
    $"i{representation.Bits}"

/// Physical scalar encoding only. Ordered source steps retain the complete
/// expression; no function-name classification or graph scan occurs here.
let scalarFunction (plan: KernelModuleWitness) : Result<string,string> =
    try
        let lines = ResizeArray<string>()
        let mutable values : Map<NodeId,string*string> = Map.empty
        let read id = values.TryFind id |> Option.defaultWith (fun () -> invalidOp $"Kernel scalar dependency {NodeId.value id} is not established in source order.")
        let emit value = lines.Add("    " + value)
        let adapt stem (meet: Meet option) (operand,sourceType) =
            match meet with
            | None -> operand,sourceType
            | Some meet ->
                require (sourceType=$"i{meet.From}") "AIE adaptation input differs from the source-published carrier."
                let opcode = match meet.Adapt with MeetKind.ExtendSigned -> "extsi" | MeetKind.ExtendUnsigned -> "extui" | MeetKind.Truncate -> "trunci" | _ -> invalidOp "Integer kernel has a noninteger adaptation."
                let destination = $"i{meet.To}"
                emit $"{stem} = arith.{opcode} {operand} : {sourceType} to {destination}"
                stem,destination
        let bind site value =
            require (not(values.ContainsKey site)) "Kernel scalar plan contains a duplicate value occurrence."
            values <- values.Add(site,value)
        let parameters = plan.Steps |> List.choose (function KernelScalarStep.Parameter(site,ordinal,carrier) -> Some(site,ordinal,carrier) | _ -> None) |> List.sortBy (fun (_,ordinal,_) -> ordinal)
        require (parameters |> List.map (fun (site,_,_) -> site) = (plan.Parameters |> List.map snd)) "Kernel formals differ from their ordered source declarations."
        let resultCarrier = plan.Steps |> List.tryFind (stepSite >> (=) plan.Result) |> Option.map stepCarrier |> Option.defaultWith (fun () -> invalidOp "Kernel result has no complete source step.")
        let resultType = scalar resultCarrier.Slot
        require (parameters |> List.map (fun (_,_,carrier) -> scalar carrier.Slot) = (plan.Ingress.Inputs |> List.map transportType))
                "The kernel formals differ from the exact ordered external transport carriers."
        require (resultType=transportType plan.Ingress.Output) "The kernel result differs from its declared output transport carrier."
        for step in plan.Steps do
            let site = stepSite step
            let name = sprintf "%%k%d" (NodeId.value site)
            match step with
            | KernelScalarStep.Parameter(_,ordinal,carrier) -> bind site (sprintf "%%arg%d" ordinal,scalar carrier.Slot)
            | KernelScalarStep.Literal(_,literal,carrier) ->
                let value =
                    match literal with
                    | KernelScalarLiteral.Integer value -> string value
                    | KernelScalarLiteral.Boolean value -> if value then "true" else "false"
                    | KernelScalarLiteral.Character value -> string(int value)
                let ty = scalar carrier.Slot
                emit $"{name} = arith.constant {value} : {ty}"
                bind site (name,ty)
            | KernelScalarStep.Alias(_,source,carrier,meet) ->
                let value = adapt name meet (read source)
                require (snd value=scalar carrier.Slot) "Kernel alias differs from its exact source result slot."
                bind site value
            | KernelScalarStep.Operation operation ->
                let ty = operation.OperationCarrier |> Option.map scalar |> Option.defaultWith (fun () -> invalidOp "Kernel operation lacks its published construction carrier.")
                let mutable adapted : Map<NodeId,string*string> = Map.empty
                let operands =
                    operation.Operands
                    |> List.mapi (fun ordinal operand ->
                        match adapted.TryFind operand.Actual with
                        | Some value -> value
                        | None ->
                            let value = adapt ($"{name}_in{ordinal}") operand.Adaptation (read operand.Actual)
                            require (snd value=ty) "Kernel operand differs from its source operation carrier."
                            adapted <- adapted.Add(operand.Actual,value)
                            value)
                    |> List.map fst
                let signed = match operation.Form with NumericOperationForm.Integer signed -> signed | NumericOperationForm.Boolean -> false | _ -> invalidOp "This AIE realization requires an integer or Boolean operation form."
                let suffix signedName unsignedName = if signed then signedName else unsignedName
                let raw = name + "_operation"
                let binary opcode =
                    match operands with
                    | [left;right] -> emit $"{raw} = arith.{opcode} {left}, {right} : {ty}"; raw,ty
                    | _ -> invalidOp "Kernel binary construction changed its ordered operand cardinality."
                let comparison predicate =
                    match operands with
                    | [left;right] -> emit $"{raw} = arith.cmpi {predicate}, {left}, {right} : {ty}"; raw,"i1"
                    | _ -> invalidOp "Kernel comparison changed its ordered operand cardinality."
                let computed =
                    match operation.Kind with
                    | NumericOperationKind.Add -> binary "addi" | NumericOperationKind.Subtract -> binary "subi"
                    | NumericOperationKind.Multiply -> binary "muli" | NumericOperationKind.Divide -> binary (suffix "divsi" "divui")
                    | NumericOperationKind.Remainder -> binary (suffix "remsi" "remui")
                    | NumericOperationKind.BitAnd -> binary "andi" | NumericOperationKind.BitOr -> binary "ori" | NumericOperationKind.BitXor -> binary "xori"
                    | NumericOperationKind.ShiftLeft -> binary "shli" | NumericOperationKind.ShiftRight -> binary (suffix "shrsi" "shrui")
                    | NumericOperationKind.Equal -> comparison "eq" | NumericOperationKind.NotEqual -> comparison "ne"
                    | NumericOperationKind.Less -> comparison (suffix "slt" "ult") | NumericOperationKind.LessOrEqual -> comparison (suffix "sle" "ule")
                    | NumericOperationKind.Greater -> comparison (suffix "sgt" "ugt") | NumericOperationKind.GreaterOrEqual -> comparison (suffix "sge" "uge")
                    | NumericOperationKind.Identity ->
                        match operands with [operand] -> operand,ty | _ -> invalidOp "Kernel identity changed its ordered operand cardinality."
                    | NumericOperationKind.Negate | NumericOperationKind.Complement | NumericOperationKind.LogicalNot ->
                        match operands with
                        | [operand] ->
                            let constant,opcode = match operation.Kind with NumericOperationKind.Negate -> "0","subi" | NumericOperationKind.LogicalNot -> "true","xori" | _ -> "-1","xori"
                            emit $"{name}_constant = arith.constant {constant} : {ty}"
                            emit $"{raw} = arith.{opcode} {name}_constant, {operand} : {ty}"
                            raw,ty
                        | _ -> invalidOp "Kernel unary construction changed its ordered operand cardinality."
                let value = adapt name operation.ResultAdaptation computed
                require (snd value=scalar operation.Result.Slot) "Kernel operation differs from its exact source result slot."
                bind site value
        let result,resultPhysical = read plan.Result
        require (resultPhysical=resultType) "Kernel returned value differs from its source result carrier."
        let args = parameters |> List.map (fun (_,ordinal,carrier) -> sprintf "%%arg%d: %s" ordinal (scalar carrier.Slot)) |> String.concat ", "
        Ok($"func.func private @{symbol plan}({args}) -> {resultType} {{\n" + String.concat "\n" lines + $"\n    func.return {result} : {resultType}\n  }}")
    with error -> Error("AIE scalar realization: " + error.Message)

let private plans (input: BackEndInput) = input.Operations |> List.choose (function MLIROp.SpatialModule(SpatialModuleWitness.Kernel plan) -> Some plan | _ -> None)

let validate (input: BackEndInput) (_context: BackEndContext) =
    let foreign = input.Operations |> List.exists (function MLIROp.SpatialModule(SpatialModuleWitness.Hardware _) | MLIROp.FuncOp(BoundaryFuncDecl _ | IntrinsicWriteDecl _) -> true | _ -> false)
    match plans input with
    | [plan] when not foreign ->
        // Verified against MLIR-AIE's documented npu2 topology: 8 columns,
        // shim row 0, memory row 1, compute rows 2 through 5.
        if plan.Target.Device<>"npu2" || plan.Target.Columns<>8 || plan.Target.ShimRow<>0 || plan.Target.ComputeRow<2 || plan.Target.ComputeRow>5 then
            Error "AIE realization requires the explicitly declared npu2 topology (8 columns, shim row 0, compute row 2..5)."
        elif plan.Target.Iterations<>1I then Error "The admitted AIE transaction performs exactly one declared host transfer; repeated submissions require a separate source progress contract."
        elif plan.Tiles.IsEmpty || plan.Tiles.Length>plan.Target.Columns then Error "AIE realization received an invalid published tile partition."
        else scalarFunction plan |> Result.map ignore
    | _ -> Error "AIE realization requires exactly one complete kernel plan and no foreign/hardware declarations."

let realize (context: BackEndContext) (input: BackEndInput) : Result<BackEndInput * KernelModuleWitness,string> =
    validate input context |> Result.bind (fun () ->
        let plan = (plans input).Head
        scalarFunction plan |> Result.map (fun compute ->
            let sb=StringBuilder()
            let emit (value:string)=sb.AppendLine(value) |> ignore
            let emitf format=Printf.kprintf emit format
            let parameters=plan.Ingress.Inputs |> List.map transportType
            let output=transportType plan.Ingress.Output
            let types=parameters @ [output]
            emit "aie.device(npu2) @main {"
            emit ("  " + compute)
            for tile in plan.Tiles do
                emitf "  %%shim_%d = aie.tile(%d, %d)" tile.Column tile.Column tile.ShimRow
                emitf "  %%tile_%d = aie.tile(%d, %d)" tile.Column tile.Column tile.ComputeRow
                for index,ty in List.indexed types do
                    let producer,consumer=if index<2 then sprintf "%%shim_%d" tile.Column,sprintf "%%tile_%d" tile.Column else sprintf "%%tile_%d" tile.Column,sprintf "%%shim_%d" tile.Column
                    emitf "  aie.objectfifo @stream_%d_%d(%s, {%s}, %d : i32) : !aie.objectfifo<memref<%dx%s>>" tile.Column index producer consumer plan.Target.FifoDepth tile.Elements ty
                emitf "  %%core_%d_%d = aie.core(%%tile_%d) {" tile.Column tile.ComputeRow tile.Column
                emit "    %c0 = arith.constant 0 : index"
                emit "    %c1 = arith.constant 1 : index"
                emitf "    %%extent = arith.constant %d : index" tile.Elements
                for index,ty in List.indexed types do
                    let direction=if index<2 then "Consume" else "Produce"
                    emitf "    %%slice_%d = aie.objectfifo.acquire @stream_%d_%d(%s, 1) : !aie.objectfifosubview<memref<%dx%s>>" index tile.Column index direction tile.Elements ty
                    emitf "    %%buffer_%d = aie.objectfifo.subview.access %%slice_%d[0] : !aie.objectfifosubview<memref<%dx%s>> -> memref<%dx%s>" index index tile.Elements ty tile.Elements ty
                emit "    scf.for %i = %c0 to %extent step %c1 {"
                for index,ty in List.indexed parameters do emitf "      %%in_%d = memref.load %%buffer_%d[%%i] : memref<%dx%s>" index index tile.Elements ty
                emitf "      %%result = func.call @%s(%%in_0, %%in_1) : (%s) -> %s" (symbol plan) (String.concat ", " parameters) output
                emitf "      memref.store %%result, %%buffer_2[%%i] : memref<%dx%s>" tile.Elements output
                emit "    }"
                for index,_ in List.indexed types do emitf "    aie.objectfifo.release @stream_%d_%d(%s, 1)" tile.Column index (if index<2 then "Consume" else "Produce")
                emit "    aie.end"
                emit "  }"
            let args=types |> List.mapi (fun index ty -> sprintf "%%host_%d: memref<%dx%s>" index plan.Elements ty) |> String.concat ", "
            emitf "  aie.runtime_sequence(%s) {" args
            for tile in plan.Tiles do
                for index,ty in List.indexed types do
                    emitf "    %%task_%d_%d = aiex.dma_configure_task_for @stream_%d_%d {" tile.Column index tile.Column index
                    emitf "      aie.dma_bd(%%host_%d : memref<%dx%s>, %d, %d, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = %d, stride = 1>]) {burst_length = 0 : i32}" index plan.Elements ty tile.Offset tile.Elements tile.Elements
                    emit "      aie.end"
                    emit (if index=2 then "    } {issue_token = true}" else "    }")
                    emitf "    aiex.dma_start_task(%%task_%d_%d)" tile.Column index
            for tile in plan.Tiles do emitf "    aiex.dma_await_task(%%task_%d_2)" tile.Column
            for tile in plan.Tiles do
                for index in [0;1] do emitf "    aiex.dma_free_task(%%task_%d_%d)" tile.Column index
            emit "  }"
            emit "}"
            let operations=input.Operations |> List.map (function MLIROp.SpatialModule(SpatialModuleWitness.Kernel _) -> MLIROp.RawMLIR(sb.ToString()) | operation -> operation)
            let text=match input.ModuleName with Some name -> moduleToString input.PointerBits name operations | None -> sprintf "module {\n%s\n}" (opsToString input.PointerBits operations "  ")
            {input with Operations=operations;Text=text},plan))
