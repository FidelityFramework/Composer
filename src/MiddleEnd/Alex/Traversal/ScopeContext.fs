/// ScopeContext - Principled scope management for operation accumulation
///
/// This module implements explicit scope context threading (Nanopass-style) to
/// ensure operations accumulate in their correct scope without post-processing.
///
/// ARCHITECTURAL PRINCIPLE: Operations know their scope at creation time.
/// No subtraction, no reordering, no brittle list manipulation.
///
/// Inspired by:
/// - Nanopass Framework (Scheme): Explicit state threading through scope boundaries
/// - MLIR mlir-hs: Region builder monads with isolated scopes
/// - Composer Codata principle: Witnesses observe, patterns emit, elements are atomic
module Alex.Traversal.ScopeContext

open Alex.Dialects.Core.Types

/// Scope level determines which operations are valid at this level
type ScopeLevel =
    | ModuleLevel       // Top-level: FuncDef, GlobalString, FuncDecl
    | FunctionLevel     // Inside func.func: local operations (arith, memref, etc.)
    | BlockLevel        // Inside scf.if/while/for: control flow operations

/// Scope context tracks operations accumulated at a specific scope level
/// Forms a hierarchy: child scopes nest within parent scopes
type ScopeContext = {
    Level: ScopeLevel
    Operations: MLIROp list         // Accumulated ops at THIS level
    Parent: ScopeContext option     // Parent scope for nesting
}

module ScopeContext =
    /// Create root module-level scope
    ///
    /// This is the top of the scope hierarchy. All module-level operations
    /// (function definitions, global strings, external declarations) accumulate here.
    let root () : ScopeContext =
        { Level = ModuleLevel
          Operations = []
          Parent = None }

    /// Create child scope (for function bodies, control flow)
    ///
    /// Child scopes accumulate operations independently of their parent.
    /// When witnessing completes, operations are extracted and wrapped in a
    /// structural operation (FuncDef, SCFOp) which is added to the parent.
    ///
    /// This ensures operations never appear in multiple scopes.
    let createChild (parent: ScopeContext) (level: ScopeLevel) : ScopeContext =
        { Level = level
          Operations = []
          Parent = Some parent }

    /// Add single operation to current scope
    ///
    /// Operations are accumulated in reverse order (newest first) for efficiency.
    /// Call getOps to retrieve them in correct order.
    let addOp (op: MLIROp) (scope: ScopeContext) : ScopeContext =
        { scope with Operations = op :: scope.Operations }

    /// Add multiple operations to current scope
    ///
    /// Operations are added in the order they appear in the list.
    /// Input list is reversed before prepending to maintain correct order.
    let addOps (ops: MLIROp list) (scope: ScopeContext) : ScopeContext =
        { scope with Operations = (List.rev ops) @ scope.Operations }

    /// Get all operations from this scope in correct order
    ///
    /// Operations are accumulated newest-first (cons onto list head),
    /// so we reverse to get def-before-use order.
    let getOps (scope: ScopeContext) : MLIROp list =
        List.rev scope.Operations

    /// Get parent scope (for unwinding after witnessing body)
    ///
    /// Returns None for root scope.
    let getParent (scope: ScopeContext) : ScopeContext option =
        scope.Parent

    /// Check if scope is root (module-level)
    let isRoot (scope: ScopeContext) : bool =
        scope.Parent.IsNone && scope.Level = ModuleLevel

    /// Get scope depth (0 = root, 1 = function, 2 = nested block, etc.)
    let rec depth (scope: ScopeContext) : int =
        match scope.Parent with
        | None -> 0
        | Some parent -> 1 + depth parent
