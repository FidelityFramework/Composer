/// LLVM Templates - LLVM dialect operations
///
/// Templates for LLVM-specific operations:
/// - Global variables and constants
/// - Inline assembly
/// - Function declarations
/// - Calling conventions
/// - Syscall emission
module Alex.Templates.LLVMTemplates

open Alex.Templates.TemplateTypes

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL VARIABLES
// ═══════════════════════════════════════════════════════════════════════════

/// Global variable definition
let globalVar = simple "llvm" "global" (fun (name: string, ty: string, initializer: string, isConstant: bool) ->
    let constStr = if isConstant then "constant" else "global"
    sprintf "llvm.mlir.global %s @%s(%s) : %s" constStr name initializer ty)

/// Global string constant
let globalString = simple "llvm" "global" (fun (name: string, value: string) ->
    let escaped = value.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\0A")
    sprintf "llvm.mlir.global private constant @%s(\"%s\\00\") : !llvm.array<%d x i8>"
        name escaped (String.length value + 1))

/// Address of global: llvm.mlir.addressof
let addressOf = simple "llvm" "global" (fun (result: string, globalName: string) ->
    sprintf "%s = llvm.mlir.addressof @%s : !llvm.ptr" result globalName)

// ═══════════════════════════════════════════════════════════════════════════
// NULL AND ZERO
// ═══════════════════════════════════════════════════════════════════════════

/// Null pointer constant
let nullPtr = simple "llvm" "constant" (fun (result: string) ->
    sprintf "%s = llvm.mlir.null : !llvm.ptr" result)

/// Zero initializer
let zeroInit = simple "llvm" "constant" (fun (result: string, ty: string) ->
    sprintf "%s = llvm.mlir.zero : %s" result ty)

// ═══════════════════════════════════════════════════════════════════════════
// INLINE ASSEMBLY
// ═══════════════════════════════════════════════════════════════════════════

/// Parameters for inline assembly
type InlineAsmParams = {
    Result: string option
    ResultType: string option
    AsmString: string
    Constraints: string
    Operands: (string * string) list  // (ssa, type)
    HasSideEffects: bool
}

/// Inline assembly
let inlineAsm = simple "llvm" "asm" (fun (p: InlineAsmParams) ->
    let operandsStr = p.Operands |> List.map fst |> String.concat ", "
    let operandTypesStr = p.Operands |> List.map snd |> String.concat ", "
    let sideEffectsStr = if p.HasSideEffects then "has_side_effects, " else ""
    match p.Result, p.ResultType with
    | Some result, Some retType ->
        sprintf "%s = llvm.inline_asm %s\"%s\", \"%s\" %s : (%s) -> %s"
            result sideEffectsStr p.AsmString p.Constraints operandsStr operandTypesStr retType
    | _ ->
        sprintf "llvm.inline_asm %s\"%s\", \"%s\" %s : (%s) -> ()"
            sideEffectsStr p.AsmString p.Constraints operandsStr operandTypesStr)

// ═══════════════════════════════════════════════════════════════════════════
// SYSCALL (LINUX x86_64)
// ═══════════════════════════════════════════════════════════════════════════

/// Parameters for syscall
type SyscallParams = {
    Result: string
    SyscallNumber: int
    Args: string list  // Up to 6 arguments
}

/// Linux x86_64 syscall via inline assembly
let syscallLinuxX64 = simple "llvm" "syscall" (fun (p: SyscallParams) ->
    // Map syscall args to registers: rdi, rsi, rdx, r10, r8, r9
    let argRegs = ["rdi"; "rsi"; "rdx"; "r10"; "r8"; "r9"]
    let numArgs = min (List.length p.Args) 6
    
    // Build constraints: syscall number in rax, args in their registers
    let constraints = 
        let inputConstraints = 
            List.take numArgs argRegs 
            |> List.mapi (fun i reg -> sprintf "{%s}" reg)
        sprintf "={rax},{rax},%s" (String.concat "," inputConstraints)
    
    // Build operand list: syscall number, then args
    let operands = 
        [sprintf "%d : i64" p.SyscallNumber] @ 
        (p.Args |> List.take numArgs |> List.map (fun a -> sprintf "%s : i64" a))
    
    sprintf "%s = llvm.inline_asm \"syscall\", \"%s\" %s : (i64, %s) -> i64"
        p.Result constraints (String.concat ", " operands) 
        (String.replicate numArgs "i64, " |> fun s -> s.TrimEnd([|','; ' '|])))

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION DECLARATIONS (EXTERN)
// ═══════════════════════════════════════════════════════════════════════════

/// External function declaration
let externFunc = simple "llvm" "declaration" (fun (name: string, argTypes: string list, returnType: string option) ->
    let argsStr = String.concat ", " argTypes
    match returnType with
    | Some retType -> sprintf "llvm.func @%s(%s) -> %s" name argsStr retType
    | None -> sprintf "llvm.func @%s(%s)" name argsStr)

/// LLVM function call
let llvmCall = simple "llvm" "call" (fun (result: string option, callee: string, args: (string * string) list, returnType: string option) ->
    let argsStr = args |> List.map fst |> String.concat ", "
    let argTypesStr = args |> List.map snd |> String.concat ", "
    match result, returnType with
    | Some r, Some rt -> sprintf "%s = llvm.call @%s(%s) : (%s) -> %s" r callee argsStr argTypesStr rt
    | _ -> sprintf "llvm.call @%s(%s) : (%s) -> ()" callee argsStr argTypesStr)

// ═══════════════════════════════════════════════════════════════════════════
// LLVM INTRINSICS
// ═══════════════════════════════════════════════════════════════════════════

/// LLVM intrinsic call (e.g., llvm.abs, llvm.sqrt)
let intrinsic = simple "llvm" "intrinsic" (fun (result: string, intrinsicName: string, args: (string * string) list, returnType: string) ->
    let argsStr = args |> List.map fst |> String.concat ", "
    let argTypesStr = args |> List.map snd |> String.concat ", "
    sprintf "%s = \"llvm.intr.%s\"(%s) : (%s) -> %s" result intrinsicName argsStr argTypesStr returnType)

/// Unreachable instruction
let unreachable = simple "llvm" "terminator" (fun () ->
    "llvm.unreachable")

// ═══════════════════════════════════════════════════════════════════════════
// MODULE-LEVEL OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Module header with target triple
let moduleHeader = simple "llvm" "module" (fun (targetTriple: string) ->
    sprintf "module attributes {llvm.target_triple = \"%s\"} {" targetTriple)

/// Module end
let moduleEnd = simple "llvm" "module" (fun () ->
    "}")

// ═══════════════════════════════════════════════════════════════════════════
// COMMON TARGET TRIPLES
// ═══════════════════════════════════════════════════════════════════════════

/// Linux x86_64 target triple
let targetLinuxX64 = "x86_64-unknown-linux-gnu"

/// Linux ARM64 target triple
let targetLinuxARM64 = "aarch64-unknown-linux-gnu"

/// Windows x86_64 target triple
let targetWindowsX64 = "x86_64-pc-windows-msvc"

/// macOS x86_64 target triple
let targetMacOSX64 = "x86_64-apple-darwin"

/// macOS ARM64 target triple (Apple Silicon)
let targetMacOSARM64 = "arm64-apple-darwin"

// ═══════════════════════════════════════════════════════════════════════════
// LINUX SYSCALL NUMBERS
// ═══════════════════════════════════════════════════════════════════════════

/// Common Linux x86_64 syscall numbers
module LinuxX64Syscalls =
    let read = 0
    let write = 1
    let close = 3
    let exit = 60
    let exitGroup = 231
    let clockGettime = 228
    let nanosleep = 35
    let mmap = 9
    let munmap = 11
    let brk = 12


// ═══════════════════════════════════════════════════════════════════════════
// QUOTATION-BASED TEMPLATES (Phase 5)
// ═══════════════════════════════════════════════════════════════════════════

/// Quotation-based templates for inspectability and multi-target generation
module Quot =
    open Microsoft.FSharp.Quotations
    
    // ───────────────────────────────────────────────────────────────────────
    // Memory Operations
    // ───────────────────────────────────────────────────────────────────────
    
    module Memory =
        /// Allocate stack memory
        let alloca : MLIRTemplate<AllocaParams> = {
            Quotation = <@ fun p -> sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" p.Result p.Count p.ElementType @>
            Dialect = "llvm"
            OpName = "alloca"
            IsTerminator = false
            Category = "memory"
        }
        
        /// Load from pointer
        let load : MLIRTemplate<LoadParams> = {
            Quotation = <@ fun p -> sprintf "%s = llvm.load %s : !llvm.ptr -> %s" p.Result p.Pointer p.Type @>
            Dialect = "llvm"
            OpName = "load"
            IsTerminator = false
            Category = "memory"
        }
        
        /// Store to pointer
        let store : MLIRTemplate<StoreParams> = {
            Quotation = <@ fun p -> sprintf "llvm.store %s, %s : %s, !llvm.ptr" p.Value p.Pointer p.Type @>
            Dialect = "llvm"
            OpName = "store"
            IsTerminator = false
            Category = "memory"
        }
        
        /// Get element pointer (pointer arithmetic)
        let gep : MLIRTemplate<GepParams> = {
            Quotation = <@ fun p -> sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, %s" p.Result p.Base p.Offset p.ElementType @>
            Dialect = "llvm"
            OpName = "getelementptr"
            IsTerminator = false
            Category = "memory"
        }
        
        /// Addressof global
        let addressof : MLIRTemplate<{| Result: string; GlobalName: string |}> = {
            Quotation = <@ fun p -> sprintf "%s = llvm.mlir.addressof @%s : !llvm.ptr" p.Result p.GlobalName @>
            Dialect = "llvm"
            OpName = "mlir.addressof"
            IsTerminator = false
            Category = "memory"
        }
    
    // ───────────────────────────────────────────────────────────────────────
    // Struct Operations
    // ───────────────────────────────────────────────────────────────────────
    
    module Struct =
        /// Parameters for struct field extraction
        type ExtractValueParams = {
            Result: string
            Aggregate: string
            Index: int
            AggregateType: string
        }
        
        /// Parameters for struct field insertion
        type InsertValueParams = {
            Result: string
            Value: string
            Aggregate: string
            Index: int
            AggregateType: string
        }
        
        /// Extract value from aggregate (struct)
        let extractValue : MLIRTemplate<ExtractValueParams> = {
            Quotation = <@ fun p -> sprintf "%s = llvm.extractvalue %s[%d] : %s" p.Result p.Aggregate p.Index p.AggregateType @>
            Dialect = "llvm"
            OpName = "extractvalue"
            IsTerminator = false
            Category = "struct"
        }
        
        /// Insert value into aggregate (struct)
        let insertValue : MLIRTemplate<InsertValueParams> = {
            Quotation = <@ fun p -> sprintf "%s = llvm.insertvalue %s, %s[%d] : %s" p.Result p.Value p.Aggregate p.Index p.AggregateType @>
            Dialect = "llvm"
            OpName = "insertvalue"
            IsTerminator = false
            Category = "struct"
        }
        
        /// Undefined value (for struct construction)
        let undef : MLIRTemplate<{| Result: string; Type: string |}> = {
            Quotation = <@ fun p -> sprintf "%s = llvm.mlir.undef : %s" p.Result p.Type @>
            Dialect = "llvm"
            OpName = "mlir.undef"
            IsTerminator = false
            Category = "struct"
        }
    
    // ───────────────────────────────────────────────────────────────────────
    // Control Flow
    // ───────────────────────────────────────────────────────────────────────
    
    module Control =
        /// Return with value
        let retValue : MLIRTemplate<{| Value: string; Type: string |}> = {
            Quotation = <@ fun p -> sprintf "llvm.return %s : %s" p.Value p.Type @>
            Dialect = "llvm"
            OpName = "return"
            IsTerminator = true
            Category = "control"
        }
        
        /// Return void
        let retVoid : MLIRTemplate<unit> = {
            Quotation = <@ fun () -> "llvm.return" @>
            Dialect = "llvm"
            OpName = "return"
            IsTerminator = true
            Category = "control"
        }
        
        /// Unreachable terminator
        let unreachable : MLIRTemplate<unit> = {
            Quotation = <@ fun () -> "llvm.unreachable" @>
            Dialect = "llvm"
            OpName = "unreachable"
            IsTerminator = true
            Category = "control"
        }
    
    // ───────────────────────────────────────────────────────────────────────
    // Global Definitions
    // ───────────────────────────────────────────────────────────────────────
    
    module Global =
        /// Parameters for string constant
        type StringConstParams = {
            Name: string
            Value: string
            Length: int
        }
        
        /// Global string constant
        let stringConst : MLIRTemplate<StringConstParams> = {
            Quotation = <@ fun p -> sprintf "llvm.mlir.global internal constant @%s(\"%s\\00\") : !llvm.array<%d x i8>" p.Name p.Value p.Length @>
            Dialect = "llvm"
            OpName = "mlir.global"
            IsTerminator = false
            Category = "global"
        }
        
        /// Null pointer constant
        let nullPtr : MLIRTemplate<{| Result: string |}> = {
            Quotation = <@ fun p -> sprintf "%s = llvm.mlir.null : !llvm.ptr" p.Result @>
            Dialect = "llvm"
            OpName = "mlir.null"
            IsTerminator = false
            Category = "global"
        }
