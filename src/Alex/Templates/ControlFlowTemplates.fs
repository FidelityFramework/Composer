/// Control Flow Templates - SCF and CF dialect operations
///
/// Templates for structured control flow (SCF) and unstructured (CF) operations:
/// - scf.if / scf.else
/// - scf.while / scf.for
/// - scf.yield
/// - cf.br / cf.cond_br
/// - func.return
module Alex.Templates.ControlFlowTemplates

open Alex.Templates.TemplateTypes

// ═══════════════════════════════════════════════════════════════════════════
// SCF.IF - Structured conditional
// ═══════════════════════════════════════════════════════════════════════════

/// Parameters for scf.if operation
type IfParams = {
    Condition: string
    ResultTypes: string list
    HasElse: bool
}

/// Start of scf.if block
let ifStart = simple "scf" "control" (fun (p: IfParams) ->
    let resultTypes = 
        if List.isEmpty p.ResultTypes then ""
        else sprintf " -> (%s)" (String.concat ", " p.ResultTypes)
    sprintf "scf.if %s%s {" p.Condition resultTypes)

/// Start of else block
let elseStart = simple "scf" "control" (fun () ->
    "} else {")

/// End of if/else
let ifEnd = simple "scf" "control" (fun () ->
    "}")

// ═══════════════════════════════════════════════════════════════════════════
// SCF.WHILE - Structured while loop
// ═══════════════════════════════════════════════════════════════════════════

/// Parameters for scf.while operation
type WhileParams = {
    /// Initial values for iter_args
    InitArgs: (string * string) list  // (ssa, type)
    /// Names for block arguments in before region
    BeforeArgNames: string list
    /// Names for block arguments in after region
    AfterArgNames: string list
    /// Result types
    ResultTypes: string list
}

/// Start of scf.while
let whileStart = simple "scf" "control" (fun (p: WhileParams) ->
    let initArgsStr = 
        p.InitArgs 
        |> List.map (fun (ssa, ty) -> sprintf "%s = %s" ssa ty)
        |> String.concat ", "
    let resultTypes =
        if List.isEmpty p.ResultTypes then ""
        else sprintf " -> (%s)" (String.concat ", " p.ResultTypes)
    sprintf "scf.while (%s)%s {" initArgsStr resultTypes)

/// Condition yield in before region
let whileCondition = simple "scf" "control" (fun (condition: string, passedArgs: string list) ->
    let argsStr = String.concat ", " passedArgs
    sprintf "scf.condition(%s) %s" condition argsStr)

/// Start of do block (after region)
let whileDoStart = simple "scf" "control" (fun (argNames: (string * string) list) ->
    let argsStr = 
        argNames 
        |> List.map (fun (name, ty) -> sprintf "%s: %s" name ty)
        |> String.concat ", "
    sprintf "} do { ^bb0(%s):" argsStr)

/// End of while
let whileEnd = simple "scf" "control" (fun () ->
    "}")

// ═══════════════════════════════════════════════════════════════════════════
// SCF.FOR - Structured for loop
// ═══════════════════════════════════════════════════════════════════════════

/// Parameters for scf.for operation
type ForParams = {
    InductionVar: string
    LowerBound: string
    UpperBound: string
    Step: string
    IterArgs: (string * string * string) list  // (name, init, type)
    ResultTypes: string list
}

/// Start of scf.for
let forStart = simple "scf" "control" (fun (p: ForParams) ->
    let iterArgsStr =
        if List.isEmpty p.IterArgs then ""
        else
            let args = p.IterArgs |> List.map (fun (name, init, ty) -> sprintf "%s = %s" name init) |> String.concat ", "
            sprintf " iter_args(%s)" args
    let resultTypes =
        if List.isEmpty p.ResultTypes then ""
        else sprintf " -> (%s)" (String.concat ", " p.ResultTypes)
    sprintf "scf.for %s = %s to %s step %s%s%s {" 
        p.InductionVar p.LowerBound p.UpperBound p.Step iterArgsStr resultTypes)

/// End of for
let forEnd = simple "scf" "control" (fun () ->
    "}")

// ═══════════════════════════════════════════════════════════════════════════
// SCF.YIELD
// ═══════════════════════════════════════════════════════════════════════════

/// Yield from SCF region (if, while, for)
let yield_ = simple "scf" "control" (fun (values: (string * string) list) ->
    if List.isEmpty values then
        "scf.yield"
    else
        let valuesStr = values |> List.map (fun (ssa, ty) -> sprintf "%s : %s" ssa ty) |> String.concat ", "
        sprintf "scf.yield %s" valuesStr)

/// Yield with just SSA names (types inferred)
let yieldSimple = simple "scf" "control" (fun (values: string list) ->
    if List.isEmpty values then
        "scf.yield"
    else
        sprintf "scf.yield %s" (String.concat ", " values))

// ═══════════════════════════════════════════════════════════════════════════
// CF - Unstructured control flow
// ═══════════════════════════════════════════════════════════════════════════

/// Unconditional branch
let br = simple "cf" "branch" (fun (p: BranchParams) ->
    if List.isEmpty p.Args then
        sprintf "cf.br ^%s" p.Target
    else
        sprintf "cf.br ^%s(%s)" p.Target (String.concat ", " p.Args))

/// Conditional branch
let condBr = simple "cf" "branch" (fun (p: CondBranchParams) ->
    let trueArgs = if List.isEmpty p.TrueArgs then "" else sprintf "(%s)" (String.concat ", " p.TrueArgs)
    let falseArgs = if List.isEmpty p.FalseArgs then "" else sprintf "(%s)" (String.concat ", " p.FalseArgs)
    sprintf "cf.cond_br %s, ^%s%s, ^%s%s" p.Condition p.TrueTarget trueArgs p.FalseTarget falseArgs)

// ═══════════════════════════════════════════════════════════════════════════
// FUNC.RETURN
// ═══════════════════════════════════════════════════════════════════════════

/// Return from function with value
let returnVal = simple "func" "terminator" (fun (value: string, ty: string) ->
    sprintf "func.return %s : %s" value ty)

/// Return from function without value (void)
let returnVoid = simple "func" "terminator" (fun () ->
    "func.return")

// ═══════════════════════════════════════════════════════════════════════════
// BLOCK LABELS
// ═══════════════════════════════════════════════════════════════════════════

/// Block label with arguments
let blockLabel = simple "cf" "label" (fun (name: string, args: (string * string) list) ->
    if List.isEmpty args then
        sprintf "^%s:" name
    else
        let argsStr = args |> List.map (fun (n, t) -> sprintf "%s: %s" n t) |> String.concat ", "
        sprintf "^%s(%s):" name argsStr)

// ═══════════════════════════════════════════════════════════════════════════
// FUNC.FUNC - Function definitions
// ═══════════════════════════════════════════════════════════════════════════

/// Parameters for function definition
type FuncParams = {
    Name: string
    Args: (string * string) list  // (name, type)
    ReturnType: string option
    IsPublic: bool
}

/// Function definition start
let funcStart = simple "func" "definition" (fun (p: FuncParams) ->
    let visibility = if p.IsPublic then "public " else "private "
    let argsStr = p.Args |> List.map (fun (n, t) -> sprintf "%s: %s" n t) |> String.concat ", "
    let retStr = match p.ReturnType with Some t -> sprintf " -> %s" t | None -> ""
    sprintf "func.func %s@%s(%s)%s {" visibility p.Name argsStr retStr)

/// Function definition end
let funcEnd = simple "func" "definition" (fun () ->
    "}")

// ═══════════════════════════════════════════════════════════════════════════
// FUNC.CALL
// ═══════════════════════════════════════════════════════════════════════════

/// Function call with result
let call = simple "func" "call" (fun (p: CallParams) ->
    let argsStr = p.Args |> List.map fst |> String.concat ", "
    let argTypesStr = p.Args |> List.map snd |> String.concat ", "
    match p.Result, p.ReturnType with
    | Some result, Some retType ->
        sprintf "%s = func.call @%s(%s) : (%s) -> %s" result p.Callee argsStr argTypesStr retType
    | None, None ->
        sprintf "func.call @%s(%s) : (%s) -> ()" p.Callee argsStr argTypesStr
    | _ ->
        sprintf "func.call @%s(%s) : (%s) -> ()" p.Callee argsStr argTypesStr)

/// Void function call
let callVoid = simple "func" "call" (fun (callee: string, args: (string * string) list) ->
    let argsStr = args |> List.map fst |> String.concat ", "
    let argTypesStr = args |> List.map snd |> String.concat ", "
    sprintf "func.call @%s(%s) : (%s) -> ()" callee argsStr argTypesStr)


// ═══════════════════════════════════════════════════════════════════════════
// QUOTATION-BASED TEMPLATES (Phase 5)
// ═══════════════════════════════════════════════════════════════════════════

/// Quotation-based templates for inspectability and multi-target generation
module Quot =
    open Microsoft.FSharp.Quotations
    
    // ───────────────────────────────────────────────────────────────────────
    // SCF Structured Control Flow
    // ───────────────────────────────────────────────────────────────────────
    
    module SCF =
        /// Parameters for scf.if header
        type IfHeaderParams = {
            Condition: string
            ResultTypes: string  // comma-separated or empty
        }
        
        /// Parameters for scf.while header
        type WhileHeaderParams = {
            IterArgs: string  // formatted iter_args
            ResultTypes: string  // comma-separated or empty
        }
        
        /// Parameters for scf.for header
        type ForHeaderParams = {
            InductionVar: string
            LowerBound: string
            UpperBound: string
            Step: string
            IterArgs: string
            ResultTypes: string
        }
        
        /// scf.if with optional result types
        let ifHeader : MLIRTemplate<IfHeaderParams> = {
            Quotation = <@ fun p -> 
                if p.ResultTypes = "" then sprintf "scf.if %s {" p.Condition
                else sprintf "scf.if %s -> (%s) {" p.Condition p.ResultTypes @>
            Dialect = "scf"
            OpName = "if"
            IsTerminator = false
            Category = "control"
        }
        
        /// else block opener
        let elseBlock : MLIRTemplate<unit> = {
            Quotation = <@ fun () -> "} else {" @>
            Dialect = "scf"
            OpName = "else"
            IsTerminator = false
            Category = "control"
        }
        
        /// scf.while header
        let whileHeader : MLIRTemplate<WhileHeaderParams> = {
            Quotation = <@ fun p ->
                if p.ResultTypes = "" then sprintf "scf.while (%s) {" p.IterArgs
                else sprintf "scf.while (%s) -> (%s) {" p.IterArgs p.ResultTypes @>
            Dialect = "scf"
            OpName = "while"
            IsTerminator = false
            Category = "control"
        }
        
        /// scf.condition (while loop condition)
        let condition : MLIRTemplate<{| Condition: string; PassedArgs: string |}> = {
            Quotation = <@ fun p -> sprintf "scf.condition(%s) %s" p.Condition p.PassedArgs @>
            Dialect = "scf"
            OpName = "condition"
            IsTerminator = true  // terminates before region
            Category = "control"
        }
        
        /// scf.for header
        let forHeader : MLIRTemplate<ForHeaderParams> = {
            Quotation = <@ fun p ->
                let iterPart = if p.IterArgs = "" then "" else sprintf " iter_args(%s)" p.IterArgs
                let retPart = if p.ResultTypes = "" then "" else sprintf " -> (%s)" p.ResultTypes
                sprintf "scf.for %s = %s to %s step %s%s%s {" 
                    p.InductionVar p.LowerBound p.UpperBound p.Step iterPart retPart @>
            Dialect = "scf"
            OpName = "for"
            IsTerminator = false
            Category = "control"
        }
        
        /// scf.yield (terminates SCF regions)
        let yield_ : MLIRTemplate<{| Values: string |}> = {
            Quotation = <@ fun p -> 
                if p.Values = "" then "scf.yield"
                else sprintf "scf.yield %s" p.Values @>
            Dialect = "scf"
            OpName = "yield"
            IsTerminator = true
            Category = "control"
        }
        
        /// Region/block close
        let close : MLIRTemplate<unit> = {
            Quotation = <@ fun () -> "}" @>
            Dialect = "scf"
            OpName = "close"
            IsTerminator = false
            Category = "control"
        }
    
    // ───────────────────────────────────────────────────────────────────────
    // CF Unstructured Control Flow
    // ───────────────────────────────────────────────────────────────────────
    
    module CF =
        /// Unconditional branch
        let br : MLIRTemplate<BranchParams> = {
            Quotation = <@ fun p ->
                if List.isEmpty p.Args then sprintf "cf.br ^%s" p.Target
                else sprintf "cf.br ^%s(%s)" p.Target (String.concat ", " p.Args) @>
            Dialect = "cf"
            OpName = "br"
            IsTerminator = true
            Category = "branch"
        }
        
        /// Conditional branch
        let condBr : MLIRTemplate<CondBranchParams> = {
            Quotation = <@ fun p ->
                let trueArgs = if List.isEmpty p.TrueArgs then "" else sprintf "(%s)" (String.concat ", " p.TrueArgs)
                let falseArgs = if List.isEmpty p.FalseArgs then "" else sprintf "(%s)" (String.concat ", " p.FalseArgs)
                sprintf "cf.cond_br %s, ^%s%s, ^%s%s" p.Condition p.TrueTarget trueArgs p.FalseTarget falseArgs @>
            Dialect = "cf"
            OpName = "cond_br"
            IsTerminator = true
            Category = "branch"
        }
        
        /// Block label
        let blockLabel : MLIRTemplate<{| Name: string; Args: string |}> = {
            Quotation = <@ fun p ->
                if p.Args = "" then sprintf "^%s:" p.Name
                else sprintf "^%s(%s):" p.Name p.Args @>
            Dialect = "cf"
            OpName = "block"
            IsTerminator = false
            Category = "label"
        }
    
    // ───────────────────────────────────────────────────────────────────────
    // Func Dialect
    // ───────────────────────────────────────────────────────────────────────
    
    module Func =
        /// Parameters for function definition
        type FuncDefParams = {
            Name: string
            Args: string  // formatted (name: type, ...)
            ReturnType: string  // or empty
            Visibility: string  // "public" or "private"
        }
        
        /// Function definition header
        let funcDef : MLIRTemplate<FuncDefParams> = {
            Quotation = <@ fun p ->
                let retPart = if p.ReturnType = "" then "" else sprintf " -> %s" p.ReturnType
                sprintf "func.func %s @%s(%s)%s {" p.Visibility p.Name p.Args retPart @>
            Dialect = "func"
            OpName = "func"
            IsTerminator = false
            Category = "definition"
        }
        
        /// Return with value
        let retValue : MLIRTemplate<{| Value: string; Type: string |}> = {
            Quotation = <@ fun p -> sprintf "func.return %s : %s" p.Value p.Type @>
            Dialect = "func"
            OpName = "return"
            IsTerminator = true
            Category = "terminator"
        }
        
        /// Return void
        let retVoid : MLIRTemplate<unit> = {
            Quotation = <@ fun () -> "func.return" @>
            Dialect = "func"
            OpName = "return"
            IsTerminator = true
            Category = "terminator"
        }
        
        /// Function call
        let call : MLIRTemplate<CallParams> = {
            Quotation = <@ fun p ->
                let argsStr = p.Args |> List.map fst |> String.concat ", "
                let argTypesStr = p.Args |> List.map snd |> String.concat ", "
                match p.Result, p.ReturnType with
                | Some result, Some retType ->
                    sprintf "%s = func.call @%s(%s) : (%s) -> %s" result p.Callee argsStr argTypesStr retType
                | _ ->
                    sprintf "func.call @%s(%s) : (%s) -> ()" p.Callee argsStr argTypesStr @>
            Dialect = "func"
            OpName = "call"
            IsTerminator = false
            Category = "call"
        }
