/// Sample 14: Lazy Values (Thunks)
/// Foundation of the Lazy Stack - deferred computation with memoization
/// PRD-14: Lazy must work before Seq (PRD-15)
module LazySample

// ============================================================================
// PART 1: Basic Lazy Values - Simple Deferred Computation
// ============================================================================

/// Simplest possible lazy value - wraps a literal
let lazyFortyTwo = lazy 42

/// Lazy value with computation
let lazyComputed = lazy (10 + 32)

/// Lazy string
let lazyGreeting = lazy "Hello from lazy!"

// ============================================================================
// PART 2: Lazy Values with Side Effects (Demonstrates Memoization)
// ============================================================================

/// Counter to track how many times expensive computation runs
let mutable computationCounter = 0

/// Expensive computation that should only run once
let expensiveComputation () =
    computationCounter <- computationCounter + 1
    Console.writeln "  [Computing expensive value...]"
    // Simulate expensive work
    let mutable result = 0
    let mutable i = 0
    while i < 1000 do
        result <- result + i
        i <- i + 1
    result

/// Lazy value wrapping the expensive computation
let lazyExpensive = lazy (expensiveComputation ())

// ============================================================================
// PART 3: Lazy Values with Captures (Closure-like Behavior)
// ============================================================================

/// Create a lazy value that captures parameters
let lazyMultiply (x: int) (y: int) = lazy (x * y)

/// Create a lazy value that captures and transforms
let lazySquare (n: int) = lazy (n * n)

/// Lazy value that captures multiple values
let lazyTriangle (base_: int) (height: int) = lazy ((base_ * height) / 2)

// ============================================================================
// PART 4: Conditional Lazy Evaluation
// ============================================================================

/// Only compute the branch that's needed
let lazyConditional (condition: bool) (trueValue: int) (falseValue: int) =
    let lazyTrue = lazy (
        Console.writeln "  [Computing true branch...]"
        trueValue
    )
    let lazyFalse = lazy (
        Console.writeln "  [Computing false branch...]"
        falseValue
    )
    if condition then
        Lazy.force lazyTrue
    else
        Lazy.force lazyFalse

// ============================================================================
// PART 5: Chained Lazy Values
// ============================================================================

/// First lazy value in a chain
let lazyBase = lazy (
    Console.writeln "  [Computing base...]"
    10
)

/// Second lazy value depends on first being forced
let lazyDerived (baseValue: int) = lazy (
    Console.writeln "  [Computing derived...]"
    baseValue * 2
)

/// Demonstrates forcing a chain
let forceChain () =
    let base_ = Lazy.force lazyBase
    let derived = Lazy.force (lazyDerived base_)
    derived

// ============================================================================
// ENTRY POINT
// ============================================================================

[<EntryPoint>]
let main _ =
    Console.writeln "=== Sample 14: Lazy Values (Thunks) ==="
    Console.writeln ""

    // ----- Part 1: Basic Lazy Values -----
    Console.writeln "--- Part 1: Basic Lazy Values ---"

    Console.write "lazyFortyTwo: "
    Console.writeln (Format.int (Lazy.force lazyFortyTwo))

    Console.write "lazyComputed: "
    Console.writeln (Format.int (Lazy.force lazyComputed))

    Console.write "lazyGreeting: "
    Console.writeln (Lazy.force lazyGreeting)
    Console.writeln ""

    // ----- Part 2: Memoization Demo -----
    Console.writeln "--- Part 2: Memoization (Computed Once) ---"

    Console.write "computationCounter before: "
    Console.writeln (Format.int computationCounter)

    Console.writeln "First force:"
    let result1 = Lazy.force lazyExpensive
    Console.write "  Result: "
    Console.writeln (Format.int result1)
    Console.write "  Counter after first force: "
    Console.writeln (Format.int computationCounter)

    Console.writeln "Second force (should NOT recompute):"
    let result2 = Lazy.force lazyExpensive
    Console.write "  Result: "
    Console.writeln (Format.int result2)
    Console.write "  Counter after second force: "
    Console.writeln (Format.int computationCounter)

    Console.writeln "Third force (should NOT recompute):"
    let result3 = Lazy.force lazyExpensive
    Console.write "  Result: "
    Console.writeln (Format.int result3)
    Console.write "  Counter after third force: "
    Console.writeln (Format.int computationCounter)
    Console.writeln ""

    // ----- Part 3: Captures -----
    Console.writeln "--- Part 3: Lazy Values with Captures ---"

    let lazy7x8 = lazyMultiply 7 8
    Console.write "lazyMultiply 7 8: "
    Console.writeln (Format.int (Lazy.force lazy7x8))

    let lazy12Squared = lazySquare 12
    Console.write "lazySquare 12: "
    Console.writeln (Format.int (Lazy.force lazy12Squared))

    let lazyArea = lazyTriangle 10 6
    Console.write "lazyTriangle 10 6: "
    Console.writeln (Format.int (Lazy.force lazyArea))
    Console.writeln ""

    // ----- Part 4: Conditional Evaluation -----
    Console.writeln "--- Part 4: Conditional Lazy Evaluation ---"

    Console.writeln "Condition true (only true branch should compute):"
    let whenTrue = lazyConditional true 100 200
    Console.write "  Result: "
    Console.writeln (Format.int whenTrue)

    Console.writeln "Condition false (only false branch should compute):"
    let whenFalse = lazyConditional false 100 200
    Console.write "  Result: "
    Console.writeln (Format.int whenFalse)
    Console.writeln ""

    // ----- Part 5: Chained Evaluation -----
    Console.writeln "--- Part 5: Chained Lazy Values ---"
    Console.writeln "Forcing chain (base then derived):"
    let chainResult = forceChain ()
    Console.write "  Chain result: "
    Console.writeln (Format.int chainResult)
    Console.writeln ""

    // ----- Verification Summary -----
    Console.writeln "--- Verification Summary ---"
    Console.writeln "Key behaviors verified:"
    Console.writeln "  - Basic lazy creation and forcing"
    Console.writeln "  - Memoization (counter should be 1, not 3)"
    Console.writeln "  - Captured values in lazy expressions"
    Console.writeln "  - Conditional evaluation (only one branch computed)"
    Console.writeln "  - Chained lazy dependencies"
    Console.writeln ""
    Console.writeln "Expected values:"
    Console.writeln "  lazyFortyTwo: 42"
    Console.writeln "  lazyComputed: 42"
    Console.writeln "  lazyGreeting: Hello from lazy!"
    Console.writeln "  lazyExpensive: 499500 (sum 0..999)"
    Console.writeln "  computationCounter: 1 (memoized!)"
    Console.writeln "  lazyMultiply 7 8: 56"
    Console.writeln "  lazySquare 12: 144"
    Console.writeln "  lazyTriangle 10 6: 30"
    Console.writeln "  conditional true: 100"
    Console.writeln "  conditional false: 200"
    Console.writeln "  chain result: 20"

    0
