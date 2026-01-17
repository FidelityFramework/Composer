/// Sample 14: Lazy Values (PRD-14)
/// Tests deferred computation with flat closure captures
/// Key validation: Side effects verify lazy evaluation timing
module LazySample

/// Helper to demonstrate side effects in thunks
/// Returns unit after printing
let sideEffect msg =
    Console.writeln msg

/// Lazy value with side effect - verifies deferred evaluation
/// Using sequencing: sideEffect runs, then 42 is returned
let expensive = lazy (sideEffect "Computing expensive value..."; 42)

/// Lazy-returning function - captures function parameters
/// This tests that parameter NodeIds are properly bound for capture resolution
let lazyAdd a b = lazy (sideEffect "Adding captured values..."; a + b)

[<EntryPoint>]
let main _ =
    Console.writeln "=== Lazy Values Test (PRD-14) ==="

    // Test 1: Simple lazy with no captures, no side effects
    Console.writeln "--- No Captures (Simple) ---"
    let simple = lazy 42
    let v1 = Lazy.force simple
    Console.write "lazy 42 = "
    Console.writeln (Format.int v1)

    // Test 2: Lazy with side effect - first force
    // Should print "Computing expensive value..." then result
    Console.writeln "--- First Force (with side effect) ---"
    let v2 = Lazy.force expensive
    Console.write "Result: "
    Console.writeln (Format.int v2)

    // Test 3: Second force - NO memoization initially
    // Should print "Computing expensive value..." AGAIN (per PRD-14 Option C)
    Console.writeln "--- Second Force (no memoization) ---"
    let v3 = Lazy.force expensive
    Console.write "Result: "
    Console.writeln (Format.int v3)

    // Test 4: Lazy with local variable captures
    Console.writeln "--- Local Variable Captures ---"
    let x = 10
    let y = 20
    let sum = lazy (x + y)
    let v4 = Lazy.force sum
    Console.write "lazy (10 + 20) = "
    Console.writeln (Format.int v4)

    // Test 5: Captured multiplication
    Console.writeln "--- Captured Multiplication ---"
    let multiplier = 7
    let product = lazy (multiplier * 6)
    Console.write "lazy (7 * 6) = "
    Console.writeln (Format.int (Lazy.force product))

    // Test 6: Lazy-returning function with captures
    // This is the key test - parameters (a, b) must be captured
    Console.writeln "--- Lazy-Returning Function ---"
    let addResult = lazyAdd 15 25
    Console.write "lazyAdd 15 25: "
    Console.writeln (Format.int (Lazy.force addResult))

    // Test 7: Multiple lazy values from same function
    // Verifies each call creates independent lazy struct
    Console.writeln "--- Multiple Lazy from Function ---"
    let sum1 = lazyAdd 3 4
    let sum2 = lazyAdd 5 6
    Console.write "lazyAdd 3 4: "
    Console.writeln (Format.int (Lazy.force sum1))
    Console.write "lazyAdd 5 6: "
    Console.writeln (Format.int (Lazy.force sum2))

    0
