/// WitnessRegistry - Global registry of all witness nanopasses
///
/// Each witness module exports a `nanopass` value. This registry collects all
/// witnesses into a single registry for parallel execution.
///
/// MIGRATION STATUS: Witnesses are being migrated incrementally to nanopass pattern.
/// As each witness is migrated, uncomment its registration below.
module Alex.Traversal.WitnessRegistry

// Suppress FS0040: Y-combinator uses delayed initialization for recursive scope witnesses
// This is safe - Lazy<_> ensures proper initialization order
#nowarn "40"

open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.TransferTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Core.Types.Dialects  // TargetPlatform

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS MODULE IMPORTS
// ═══════════════════════════════════════════════════════════════════════════

// Import all witness modules
// As witnesses are migrated to export `nanopass`, they're imported here

// Priority 1: Simple Witnesses
module LiteralWitness = Alex.Witnesses.LiteralWitness
module TypeAnnotationWitness = Alex.Witnesses.TypeAnnotationWitness
module PlatformWitness = Alex.Witnesses.PlatformWitness
module IntrinsicWitness = Alex.Witnesses.IntrinsicWitness

// Structural Witnesses (January 2026 - parallel fan-out, February 2026 - transparent witness pattern)
module StructuralWitness = Alex.Witnesses.StructuralWitness  // Transparent witness for ModuleDef, Sequential
module BindingWitness = Alex.Witnesses.BindingWitness
module VarRefWitness = Alex.Witnesses.VarRefWitness
module MutableAssignmentWitness = Alex.Witnesses.MutableAssignmentWitness
module MemoryIntrinsicWitness = Alex.Witnesses.MemoryIntrinsicWitness
module StringIntrinsicWitness = Alex.Witnesses.StringIntrinsicWitness
module ArithIntrinsicWitness = Alex.Witnesses.ArithIntrinsicWitness
module ApplicationWitness = Alex.Witnesses.ApplicationWitness  // Non-intrinsic apps only

// Priority 2: Collection Witnesses
module OptionWitness = Alex.Witnesses.OptionWitness
module ListWitness = Alex.Witnesses.ListWitness
module MapWitness = Alex.Witnesses.MapWitness
module SetWitness = Alex.Witnesses.SetWitness

// Priority 3: Control Flow (special - needs nanopass list for sub-graph traversal)
module ControlFlowWitness = Alex.Witnesses.ControlFlowWitness
module MatchWitness = Alex.Witnesses.MatchWitness

// Priority 4: Records, Memory, DU, Lambda & Hardware
module RecordWitness = Alex.Witnesses.RecordWitness
module MemoryWitness = Alex.Witnesses.MemoryWitness
module DUWitness = Alex.Witnesses.DUWitness
module LambdaWitness = Alex.Witnesses.LambdaWitness
module HardwareModuleWitness = Alex.Witnesses.HardwareModuleWitness

// Priority 5: Advanced Features
module LazyWitness = Alex.Witnesses.LazyWitness
module SeqWitness = Alex.Witnesses.SeqWitness

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL REGISTRY
// ═══════════════════════════════════════════════════════════════════════════

/// Global registry of all witness nanopasses
/// MIGRATION: Currently empty. As witnesses are migrated, register them here.
let mutable globalRegistry = NanopassRegistry.empty

/// Initialize the global registry
/// Called once at startup to populate the registry with all witness nanopasses
let initializeRegistry (targetPlatform: TargetPlatform) =
    // Conditional registration — target-gated witnesses are only registered
    // for platforms where they apply. This is the nanopass-level expression
    // of the Three-Category Model (fully shared, codata-dependent, target-gated).
    let conditionalRegister condition np reg =
        if condition then NanopassRegistry.register np reg else reg

    let isCPULike = targetPlatform <> FPGA  // CPU, MCU share memory/mutable semantics

    // First register leaf witnesses (literals, arithmetic, memory, etc.)
    // These witnesses don't need sub-graph traversal
    let leafRegistry =
        NanopassRegistry.empty
        // ─── Fully shared (all platforms) ───
        |> NanopassRegistry.register LiteralWitness.nanopass
        |> NanopassRegistry.register TypeAnnotationWitness.nanopass
        |> NanopassRegistry.register IntrinsicWitness.nanopass
        |> NanopassRegistry.register StructuralWitness.nanopass
        |> NanopassRegistry.register BindingWitness.nanopass
        |> NanopassRegistry.register VarRefWitness.nanopass

        // ─── Codata-dependent (all platforms, elision varies inside) ───
        |> NanopassRegistry.register ArithIntrinsicWitness.nanopass
        |> NanopassRegistry.register ApplicationWitness.nanopass

        // ─── Target-gated (CPU/MCU only — FPGA has no mutable state, heap, or syscalls) ───
        |> conditionalRegister isCPULike MutableAssignmentWitness.nanopass
        |> conditionalRegister isCPULike MemoryIntrinsicWitness.nanopass
        |> conditionalRegister isCPULike StringIntrinsicWitness.nanopass
        |> conditionalRegister isCPULike PlatformWitness.nanopass
        |> conditionalRegister isCPULike MemoryWitness.nanopass

        // ─── DU operations (all platforms — codata-dependent dispatch inside patterns) ───
        |> NanopassRegistry.register DUWitness.nanopass

        // ─── Records (all platforms, must be AFTER MemoryWitness — register prepends) ───
        |> NanopassRegistry.register RecordWitness.nanopass  // Prepends → runs BEFORE MemoryWitness

        // ─── Collections (CPU/MCU only for now — FPGA collection support is future) ───
        |> conditionalRegister isCPULike OptionWitness.nanopass
        |> conditionalRegister isCPULike ListWitness.nanopass
        |> conditionalRegister isCPULike MapWitness.nanopass
        |> conditionalRegister isCPULike SetWitness.nanopass

        // ─── Advanced features (CPU/MCU only) ───
        |> conditionalRegister isCPULike LazyWitness.nanopass
        |> conditionalRegister isCPULike SeqWitness.nanopass

    // ═══════════════════════════════════════════════════════════════════════════
    // Y-COMBINATOR FIXED POINT FOR RECURSIVE SCOPE WITNESSES
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // Problem: Scope witnesses (Lambda, ControlFlow) need to handle nested scopes
    // of their own kind (e.g., IfThenElse inside WhileLoop, nested lambdas).
    // This requires the combinator to include a reference to itself.
    //
    // Solution: Y-combinator via lazy recursive binding
    // Use Lazy<_> to ensure safe initialization - the combinator is computed once
    // on first access, after all nanopasses are defined. This breaks the initialization
    // cycle while maintaining referential transparency.
    //
    // This is purely functional - the lazy value acts as a safe fixed point.

    // Lazy combinator ensures initialization happens after nanopass list is built
    let rec lazyCombinator : Lazy<WitnessContext -> SemanticNode -> WitnessOutput> =
        lazy (
            fun ctx node ->
                let rec tryWitnesses witnesses =
                    match witnesses with
                    | [] -> WitnessOutput.skip
                    | nanopass :: rest ->
                        match nanopass.Witness ctx node with
                        | output when output.Result = TRSkip -> tryWitnesses rest
                        | output -> output
                tryWitnesses allNanopasses.Value
        )

    // Build nanopass list with thunks that access the lazy combinator
    and allNanopasses : Lazy<Nanopass list> =
        lazy (
            // FPGA-only: HardwareModuleWitness must run BEFORE BindingWitness
            // (both match Binding nodes — HW witness is more specific)
            let hwNanopass =
                if not isCPULike then
                    [ HardwareModuleWitness.createNanopass (fun () -> lazyCombinator.Value) ]
                else []

            // Insert HW witness before BindingWitness in the leaf list
            let leaves =
                leafRegistry.Nanopasses
                |> List.collect (fun np ->
                    if np.Name = "Binding" then hwNanopass @ [np]
                    else [np])

            // Append scope witnesses (receive combinator thunk for recursion)
            leaves @
            [
                LambdaWitness.createNanopass (fun () -> lazyCombinator.Value)
                ControlFlowWitness.createNanopass (fun () -> lazyCombinator.Value)
                MatchWitness.createNanopass (fun () -> lazyCombinator.Value)
            ]
        )

    let finalRegistry = { leafRegistry with Nanopasses = allNanopasses.Value }
    globalRegistry <- finalRegistry

// ═══════════════════════════════════════════════════════════════════════════
// MIGRATION NOTES
// ═══════════════════════════════════════════════════════════════════════════

/// MIGRATION CHECKLIST:
///
/// For each witness file:
/// 1. Add category-selective witness function (match node.Kind, return skip for others)
/// 2. Export `let nanopass : Nanopass = { Name = "..."; Witness = witness... }`
/// 3. Uncomment the module import above
/// 4. Uncomment the registry registration in initializeRegistry()
/// 5. Test in isolation
/// 6. Verify parallel = sequential output
///
/// See: docs/Witness_Migration_Guide.md for full migration process
