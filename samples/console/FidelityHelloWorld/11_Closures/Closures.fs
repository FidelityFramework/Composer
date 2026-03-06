/// Sample 11: Closures
/// Validates MLKit-style flat closure construction, capture, and invocation.
///
/// SECTION A — Immutable Captures
///   1. Single string capture (makeGreeter)
///   2. Single int capture (makeAdder)
///   3. Multi-value capture with comparisons (makeRangeChecker)
///   4. Heterogeneous capture: string + bool (makeFormatter)
///
/// SECTION B — Closure Composition
///   5. Zero-capture lambda via HOF (applyOp)
///   6. Closure as function argument (twice)
///   7. Nested closures: closure returning closure (makeScaledAdder)
///
/// Dependencies: arithmetic (05), comparisons + booleans (07), Format/Console
///
/// NOTE: This sample tests closure MECHANICS only — immutable captures,
/// invocation, HOF parameter passing, nested construction. Arena-based
/// mutable captures are deferred until 10b (DMM infrastructure) passes.
module ClosuresSample

open Console
open Format

// ═══════════════════════════════════════════════════════════
// SECTION A: Immutable Captures
// ═══════════════════════════════════════════════════════════

/// Create a greeting function that captures a name (immutable)
let makeGreeter (name: string) : (string -> string) =
    fun greeting -> $"{greeting}, {name}!"

/// Create an adder that captures a fixed offset (immutable)
let makeAdder (n: int) : (int -> int) =
    fun x -> x + n

/// Capture two values, use comparisons + boolean operators
let makeRangeChecker (lo: int) (hi: int) : (int -> bool) =
    fun x -> x >= lo && x <= hi

/// Heterogeneous capture: string + bool + comparison
let makeFormatter (prefix: string) (showSign: bool) : (int -> string) =
    fun value ->
        if showSign && value > 0 then
            $"{prefix}+{Format.int value}"
        else
            $"{prefix}{Format.int value}"

// ═══════════════════════════════════════════════════════════
// SECTION B: Closure Composition
// ═══════════════════════════════════════════════════════════

/// Zero-capture lambda — pure function value
/// Minimal closure struct: code_ptr only, no environment
let applyOp (f: int -> int -> int) (a: int) (b: int) : int =
    f a b

/// Apply a function twice — closure as function argument (HOF bridge)
/// Tests that closure struct flows correctly as parameter
let twice (f: int -> int) (x: int) : int =
    f (f x)

// TODO: Nested closures (makeScaledAdder) require partial application support.
// Curry flattening merges `fun m -> fun x -> ...` into a flat function,
// but `addScaled 3` expects a closure back. Deferred until PAP is implemented.

// ═══════════════════════════════════════════════════════════
// ENTRY POINT
// ═══════════════════════════════════════════════════════════

[<EntryPoint>]
let main _ =
    Console.writeln "=== Closures Test ==="

    // ─── Section A: Immutable Captures ───

    Console.writeln "--- Greeter ---"
    let greetAlice = makeGreeter "Alice"
    let greetBob = makeGreeter "Bob"
    Console.writeln (greetAlice "Hello")
    Console.writeln (greetAlice "Goodbye")
    Console.writeln (greetBob "Welcome")

    Console.writeln ""
    Console.writeln "--- Adder ---"
    let add10 = makeAdder 10
    let add100 = makeAdder 100
    Console.write "add10 5: "
    Console.writeln (Format.int (add10 5))
    Console.write "add10 20: "
    Console.writeln (Format.int (add10 20))
    Console.write "add100 5: "
    Console.writeln (Format.int (add100 5))

    Console.writeln ""
    Console.writeln "--- Range Checker ---"
    let inRange = makeRangeChecker 10 20
    Console.write "5 in range 10-20: "
    Console.writeln (if inRange 5 then "true" else "false")
    Console.write "15 in range 10-20: "
    Console.writeln (if inRange 15 then "true" else "false")
    Console.write "25 in range 10-20: "
    Console.writeln (if inRange 25 then "true" else "false")

    Console.writeln ""
    Console.writeln "--- Multi-Type Captures ---"
    let fmtPos = makeFormatter "val=" true
    let fmtPlain = makeFormatter "x=" false
    Console.writeln (fmtPos 42)
    Console.writeln (fmtPos (-7))
    Console.writeln (fmtPlain 42)

    // ─── Section B: Closure Composition ───

    Console.writeln ""
    Console.writeln "--- Zero-Capture Lambda ---"
    Console.write "add: "
    Console.writeln (Format.int (applyOp (fun a b -> a + b) 3 4))
    Console.write "mul: "
    Console.writeln (Format.int (applyOp (fun a b -> a * b) 3 4))

    Console.writeln ""
    Console.writeln "--- HOF Bridge (twice) ---"
    Console.write "twice add10 on 5: "
    Console.writeln (Format.int (twice add10 5))
    Console.write "twice (fun x -> x * 2) on 3: "
    Console.writeln (Format.int (twice (fun x -> x * 2) 3))

    0
