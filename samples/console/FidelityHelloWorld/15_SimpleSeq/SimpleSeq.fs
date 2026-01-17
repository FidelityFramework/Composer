/// Sample 15: Simple Sequence Expressions
/// Basic seq { } with yields, while loops, conditionals
/// PRD-15: Builds on Lazy (PRD-14) - sequences are resumable thunks
module SimpleSeqSample

// ============================================================================
// PART 1: Basic Sequences - Literal Yields
// ============================================================================

/// Simplest possible sequence - multiple literal yields
let threeNumbers = seq {
    yield 1
    yield 2
    yield 3
}

/// Single element sequence (edge case)
let singleElement = seq {
    yield 42
}

/// Empty sequence body with conditional that's always false (edge case)
let emptySeq = seq {
    if false then
        yield 0
}

// ============================================================================
// PART 2: Sequences with While Loops
// ============================================================================

/// Sequence with while loop and mutable state
let countUp (start: int) (stop: int) = seq {
    let mutable i = start
    while i <= stop do
        yield i
        i <- i + 1
}

/// Countdown sequence
let countDown (start: int) (stop: int) = seq {
    let mutable i = start
    while i >= stop do
        yield i
        i <- i - 1
}

// ============================================================================
// PART 3: Sequences with Conditional Yields
// ============================================================================

/// Sequence with conditional yields (filtering for evens)
let evenNumbersUpTo (max: int) = seq {
    let mutable n = 0
    while n <= max do
        if n % 2 = 0 then
            yield n
        n <- n + 1
}

/// Sequence with conditional yields (odd numbers)
let oddNumbersUpTo (max: int) = seq {
    let mutable n = 1
    while n <= max do
        yield n
        n <- n + 2
}

/// Sequence with nested conditionals (non-fizzbuzz numbers)
let nonFizzBuzzUpTo (max: int) = seq {
    let mutable n = 1
    while n <= max do
        // Yield only numbers that are NOT divisible by 3 or 5
        if n % 3 <> 0 then
            if n % 5 <> 0 then
                yield n
        n <- n + 1
}

// ============================================================================
// PART 4: Sequences with Captures (Closure-like behavior)
// ============================================================================

/// Sequence that captures a parameter and uses it in computation
let multiplesOf (factor: int) (count: int) = seq {
    let mutable i = 1
    while i <= count do
        yield factor * i
        i <- i + 1
}

/// Sequence that captures multiple values
let rangeWithStep (start: int) (stop: int) (step: int) = seq {
    let mutable current = start
    while current <= stop do
        yield current
        current <- current + step
}

// ============================================================================
// PART 5: Sequences with Computed Values
// ============================================================================

/// Sequence that yields computed values (squares)
let squares (count: int) = seq {
    let mutable i = 1
    while i <= count do
        yield i * i
        i <- i + 1
}

/// Sequence with accumulating computation (triangular numbers)
let triangularNumbers (count: int) = seq {
    let mutable sum = 0
    let mutable i = 1
    while i <= count do
        sum <- sum + i
        yield sum
        i <- i + 1
}

// ============================================================================
// PART 6: Classic Sequences
// ============================================================================

/// First N Fibonacci numbers
let fibonacci (count: int) = seq {
    let mutable a = 0
    let mutable b = 1
    let mutable i = 0
    while i < count do
        yield a
        let temp = a + b
        a <- b
        b <- temp
        i <- i + 1
}

/// Powers of 2
let powersOfTwo (count: int) = seq {
    let mutable power = 1
    let mutable i = 0
    while i < count do
        yield power
        power <- power * 2
        i <- i + 1
}

// ============================================================================
// ENTRY POINT
// ============================================================================

[<EntryPoint>]
let main _ =
    Console.writeln "=== Sample 15: Simple Sequences ==="
    Console.writeln ""

    // ----- Part 1: Basic Sequences -----
    Console.writeln "--- Part 1: Basic Sequences ---"

    Console.write "threeNumbers: "
    for x in threeNumbers do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "singleElement: "
    for x in singleElement do
        Console.write (Format.int x)
    Console.writeln ""

    Console.write "emptySeq (should be empty): "
    for x in emptySeq do
        Console.write (Format.int x)
    Console.writeln "(done)"
    Console.writeln ""

    // ----- Part 2: While Loop Sequences -----
    Console.writeln "--- Part 2: While Loop Sequences ---"

    Console.write "countUp 5 10: "
    for x in countUp 5 10 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "countDown 10 5: "
    for x in countDown 10 5 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 3: Conditional Sequences -----
    Console.writeln "--- Part 3: Conditional Yields ---"

    Console.write "evenNumbersUpTo 10: "
    for x in evenNumbersUpTo 10 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "oddNumbersUpTo 10: "
    for x in oddNumbersUpTo 10 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "nonFizzBuzzUpTo 15: "
    for x in nonFizzBuzzUpTo 15 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 4: Sequences with Captures -----
    Console.writeln "--- Part 4: Sequences with Captures ---"

    Console.write "multiplesOf 3 5: "
    for x in multiplesOf 3 5 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "rangeWithStep 0 20 4: "
    for x in rangeWithStep 0 20 4 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 5: Computed Sequences -----
    Console.writeln "--- Part 5: Computed Sequences ---"

    Console.write "squares 6: "
    for x in squares 6 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "triangularNumbers 6: "
    for x in triangularNumbers 6 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 6: Classic Sequences -----
    Console.writeln "--- Part 6: Classic Sequences ---"

    Console.write "fibonacci 10: "
    for x in fibonacci 10 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "powersOfTwo 8: "
    for x in powersOfTwo 8 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Verification Summary -----
    Console.writeln "--- Verification Summary ---"
    Console.writeln "Expected values:"
    Console.writeln "  threeNumbers: 1 2 3"
    Console.writeln "  singleElement: 42"
    Console.writeln "  emptySeq: (nothing)"
    Console.writeln "  countUp 5 10: 5 6 7 8 9 10"
    Console.writeln "  countDown 10 5: 10 9 8 7 6 5"
    Console.writeln "  evenNumbersUpTo 10: 0 2 4 6 8 10"
    Console.writeln "  oddNumbersUpTo 10: 1 3 5 7 9"
    Console.writeln "  nonFizzBuzzUpTo 15: 1 2 4 7 8 11 13 14"
    Console.writeln "  multiplesOf 3 5: 3 6 9 12 15"
    Console.writeln "  rangeWithStep 0 20 4: 0 4 8 12 16 20"
    Console.writeln "  squares 6: 1 4 9 16 25 36"
    Console.writeln "  triangularNumbers 6: 1 3 6 10 15 21"
    Console.writeln "  fibonacci 10: 0 1 1 2 3 5 8 13 21 34"
    Console.writeln "  powersOfTwo 8: 1 2 4 8 16 32 64 128"

    0
