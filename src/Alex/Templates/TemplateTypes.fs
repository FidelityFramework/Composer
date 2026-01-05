/// Template Types - Quotation-Based Template Infrastructure
///
/// This module provides the foundation for multi-target code generation.
/// Templates are defined using F# quotations, enabling:
/// - MLIR generation (primary target)
/// - Future: TableGen generation
/// - Future: C/C++ generation
///
/// Design principles:
/// - Templates are DATA (quotations), not CODE (functions)
/// - Quotations can be inspected for multi-target rendering
/// - Each template carries metadata (dialect, category)
/// - Rendering is a separate concern from definition
module Alex.Templates.TemplateTypes

open Microsoft.FSharp.Quotations

// ═══════════════════════════════════════════════════════════════════════════
// CORE TEMPLATE TYPE
// ═══════════════════════════════════════════════════════════════════════════

/// A template that can be inspected and rendered to multiple targets.
/// The quotation encodes the MLIR generation logic as data.
type Template<'Params> = {
    /// The quotation encoding the operation
    /// This can be inspected for multi-target generation
    Quotation: Expr<'Params -> string>
    
    /// MLIR dialect this template targets (e.g., "arith", "llvm", "scf")
    Dialect: string
    
    /// Category within the dialect (e.g., "binary", "compare", "memory")
    Category: string
    
    /// Human-readable description
    Description: string
}

/// An MLIR template with operation-level metadata for multi-target generation.
/// This is the primary template type for Phase 5+ quotation inspection.
type MLIRTemplate<'Params> = {
    /// The quotation encoding the operation - inspectable for multi-target
    Quotation: Expr<'Params -> string>
    
    /// MLIR dialect (e.g., "arith", "llvm", "scf", "cf")
    Dialect: string
    
    /// Operation name within the dialect (e.g., "addi", "load", "if")
    OpName: string
    
    /// Whether this operation is a block terminator
    IsTerminator: bool
    
    /// Category for grouping (e.g., "binary", "memory", "control")
    Category: string
}

/// A simple template that directly provides the MLIR string generator
/// (for cases where quotation inspection is not needed)
type SimpleTemplate<'Params> = {
    /// Direct function to generate MLIR
    Render: 'Params -> string
    
    /// MLIR dialect
    Dialect: string
    
    /// Category
    Category: string
}

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE CREATION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Create a template with full quotation
let template dialect category description (quotation: Expr<'Params -> string>) : Template<'Params> =
    {
        Quotation = quotation
        Dialect = dialect
        Category = category
        Description = description
    }

/// Create a simple template (no quotation inspection needed)
/// Note: Prefer MLIRTemplate for new code - SimpleTemplate exists for compatibility
let simple dialect category (render: 'Params -> string) : SimpleTemplate<'Params> =
    {
        Render = render
        Dialect = dialect
        Category = category
    }

// ═══════════════════════════════════════════════════════════════════════════
// RENDERING
// ═══════════════════════════════════════════════════════════════════════════

/// Render a quotation-based template to MLIR
/// For now, this evaluates the quotation; future versions may inspect it
let renderMLIR (template: Template<'Params>) (args: 'Params) : string =
    // Evaluate the quotation to get the function, then apply
    let func = Microsoft.FSharp.Linq.RuntimeHelpers.LeafExpressionConverter.EvaluateQuotation template.Quotation :?> ('Params -> string)
    func args

/// Render a simple template
let renderSimple (template: SimpleTemplate<'Params>) (args: 'Params) : string =
    template.Render args

/// Render an MLIRTemplate - evaluates quotation for MLIR text generation
/// Future: This function will be extended to support multi-target rendering
/// by analyzing the quotation structure instead of just evaluating it.
let render (template: MLIRTemplate<'Params>) (args: 'Params) : string =
    let func = Microsoft.FSharp.Linq.RuntimeHelpers.LeafExpressionConverter.EvaluateQuotation template.Quotation :?> ('Params -> string)
    func args

// ═══════════════════════════════════════════════════════════════════════════
// COMMON PARAMETER TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Parameters for binary operations (result, lhs, rhs, type)
type BinaryOpParams = {
    Result: string
    Lhs: string
    Rhs: string
    Type: string
}

/// Parameters for unary operations (result, operand, type)
type UnaryOpParams = {
    Result: string
    Operand: string
    Type: string
}

/// Parameters for comparison operations (result, lhs, rhs, type)
/// Result type is always i1 (bool)
type CompareParams = {
    Result: string
    Lhs: string
    Rhs: string
    Type: string
}

// ═══════════════════════════════════════════════════════════════════════════
// OPERATION KIND TYPES (shared between Templates and Patterns)
// ═══════════════════════════════════════════════════════════════════════════

/// Binary arithmetic operation kinds (for template dispatch)
type BinaryArithOp =
    | Add | Sub | Mul | Div | Mod
    | BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight

/// Comparison operation kinds
type CompareOp =
    | Lt | Le | Gt | Ge | Eq | Ne

/// Parameters for constant/literal operations
type ConstantParams = {
    Result: string
    Value: string
    Type: string
}

/// Parameters for load operations
type LoadParams = {
    Result: string
    Pointer: string
    Type: string
}

/// Parameters for store operations
type StoreParams = {
    Value: string
    Pointer: string
    Type: string
}

/// Parameters for GEP (pointer arithmetic) operations
type GepParams = {
    Result: string
    Base: string
    Offset: string
    ElementType: string
}

/// Parameters for type conversion operations
type ConversionParams = {
    Result: string
    Operand: string
    FromType: string
    ToType: string
}

/// Parameters for call operations
type CallParams = {
    Result: string option  // None for void calls
    Callee: string
    Args: (string * string) list  // (ssa, type) pairs
    ReturnType: string option
}

/// Parameters for alloca operations
type AllocaParams = {
    Result: string
    Count: string
    ElementType: string
}

/// Parameters for branch/jump operations
type BranchParams = {
    Target: string
    Args: string list
}

/// Parameters for conditional branch
type CondBranchParams = {
    Condition: string
    TrueTarget: string
    TrueArgs: string list
    FalseTarget: string
    FalseArgs: string list
}

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE RESULT TYPE - For witnessing
// ═══════════════════════════════════════════════════════════════════════════

/// Result of template rendering with metadata
type TemplateResult = {
    /// The generated MLIR text
    Text: string
    
    /// The result SSA name (if operation produces a value)
    ResultSSA: string option
    
    /// The result type (if operation produces a value)
    ResultType: string option
    
    /// Whether this is a terminator operation
    IsTerminator: bool
}

/// Create a result for value-producing operations
let valueResult text ssa ty = {
    Text = text
    ResultSSA = Some ssa
    ResultType = Some ty
    IsTerminator = false
}

/// Create a result for void operations
let voidResult text = {
    Text = text
    ResultSSA = None
    ResultType = None
    IsTerminator = false
}

/// Create a result for terminator operations
let terminatorResult text = {
    Text = text
    ResultSSA = None
    ResultType = None
    IsTerminator = true
}
