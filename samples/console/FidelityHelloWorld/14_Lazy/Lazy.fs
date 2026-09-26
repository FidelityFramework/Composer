/// Sample 14: Lazy Values (C-05)
/// Tests deferred computation, shared memoization and original capture cells.
/// Printed results demand each force; effects reveal its execution count.
module LazySample

open Console
open Format

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
    Console.writeln "=== Lazy Values Test (C-05) ==="

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

    // Test 3: The same instance returns its cached result without another effect.
    Console.writeln "--- Second Force (memoized) ---"
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
    // Verifies each call creates an independent memoized instance.
    Console.writeln "--- Multiple Lazy from Function ---"
    let sum1 = lazyAdd 3 4
    let sum2 = lazyAdd 5 6
    Console.write "lazyAdd 3 4: "
    Console.writeln (Format.int (Lazy.force sum1))
    Console.write "lazyAdd 5 6: "
    Console.writeln (Format.int (Lazy.force sum2))

    // Test 8: Aliasing a returned value preserves its existing cache.
    let sumAlias = sum1
    Console.write "Cached factory alias: "
    Console.writeln (Format.int (Lazy.force sumAlias))

    // Test 9: Captures retain the original cells. The first demanded force
    // sees the latest input; subsequent forces retain the cached result.
    Console.writeln "--- Shared Mutable Captures ---"
    let mutable input = 10
    let mutable executions = 0
    let memo = lazy (executions <- executions + 1; input)
    let memoAlias = memo
    input <- 20
    Console.write "First force via alias: "
    Console.writeln (Format.int (Lazy.force memoAlias))
    input <- 30
    Console.write "Cached force after mutation: "
    Console.writeln (Format.int (Lazy.force memo))
    Console.write "Thunk executions: "
    Console.writeln (Format.int executions)

    0
