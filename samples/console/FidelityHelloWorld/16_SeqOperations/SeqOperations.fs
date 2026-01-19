/// Sample 16: Sequence Operations
/// Comprehensive test coverage for Seq.map, Seq.filter, Seq.take, Seq.fold, Seq.collect
/// PRD-16: Builds on PRD-15 (SimpleSeq), PRD-14 (Lazy), PRD-12 (HOFs), PRD-11 (Closures)
///
/// KEY ARCHITECTURAL PRINCIPLE: Seq operations create COMPOSED FLAT CLOSURES
/// Each wrapper contains: {state, current, moveNext_ptr, inner_seq, closure}
/// Both inner_seq and closure are INLINED (copied by value, not by pointer)
module SeqOperationsSample

// ============================================================================
// PART 1: Source Sequences (from PRD-15)
// Foundation sequences used as inputs for transformation operations
// ============================================================================

/// Simple range sequence using PRD-15 while-based pattern
let range (start: int) (stop: int) = seq {
    let mutable i = start
    while i <= stop do
        yield i
        i <- i + 1
}

/// First N natural numbers (convenience wrapper)
let naturals (n: int) = range 1 n

// ============================================================================
// PART 2: Seq.map - Transform Each Element (No Captures)
// Tests basic transformation with inline computation
// Mapper closure: {code_ptr} (no captures)
// ============================================================================

/// Double every element
let doubled = Seq.map (fun x -> x * 2) (naturals 5)

/// Square every element
let squared = Seq.map (fun x -> x * x) (naturals 5)

/// Add constant to every element (constant is literal, not capture)
let addTen = Seq.map (fun x -> x + 10) (naturals 5)

// ============================================================================
// PART 3: Seq.filter - Select Matching Elements (No Captures)
// Tests predicate filtering with inline conditions
// Predicate closure: {code_ptr} (no captures)
// ============================================================================

/// Filter for even numbers
let evens = Seq.filter (fun x -> x % 2 = 0) (naturals 10)

/// Filter for odd numbers
let odds = Seq.filter (fun x -> x % 2 = 1) (naturals 10)

/// Filter for numbers greater than 5
let greaterThan5 = Seq.filter (fun x -> x > 5) (naturals 10)

// ============================================================================
// PART 4: Seq.take - Limit to First N Elements
// Tests count limiting with boundary conditions
// TakeSeq: {state, current, moveNext_ptr, inner_seq, remaining: int}
// ============================================================================

/// Take first 3 from a longer sequence
let firstThree = Seq.take 3 (naturals 100)

/// Take first 5 from exactly 5 (boundary condition)
let exactlyFive = Seq.take 5 (naturals 5)

/// Take more than available (graceful handling)
let takeMoreThanAvailable = Seq.take 10 (naturals 3)

// ============================================================================
// PART 5: Seq.fold - Reduce to Single Value (No Captures)
// Eager consumer - consumes sequence immediately, returns accumulated value
// NOT a wrapper sequence (doesn't produce seq<'T>)
// ============================================================================

/// Sum all elements: 1+2+3+4+5+6+7+8+9+10 = 55
let sum = Seq.fold (fun acc x -> acc + x) 0 (naturals 10)

/// Product of all elements: 1*2*3*4*5 = 120
let product = Seq.fold (fun acc x -> acc * x) 1 (naturals 5)

/// Find maximum (left fold with comparison)
let findMax = Seq.fold (fun acc x -> if x > acc then x else acc) 0 (naturals 10)

/// Count elements (ignore value, just increment)
let countElements = Seq.fold (fun acc _ -> acc + 1) 0 (naturals 10)

// ============================================================================
// PART 6: Composed Operations / Pipelines (No Captures)
// Tests that multiple operations compose correctly
// Creates nested struct: TakeSeq { MapSeq { FilterSeq { inner } } }
// ============================================================================

/// Filter then map: evens squared
/// Struct: MapSeq { FilterSeq { naturals } }
let evensSquared =
    naturals 10
    |> Seq.filter (fun x -> x % 2 = 0)
    |> Seq.map (fun x -> x * x)

/// Map then filter: squares > 10
/// Struct: FilterSeq { MapSeq { naturals } }
let squaresOver10 =
    naturals 10
    |> Seq.map (fun x -> x * x)
    |> Seq.filter (fun x -> x > 10)

/// Filter, map, take: first 3 even numbers doubled
/// Struct: TakeSeq { MapSeq { FilterSeq { naturals } } }
let first3EvensDoubled =
    naturals 100
    |> Seq.filter (fun x -> x % 2 = 0)
    |> Seq.map (fun x -> x * 2)
    |> Seq.take 3

/// Sum of squares of evens (pipeline ending in fold)
let sumEvenSquares =
    naturals 10
    |> Seq.filter (fun x -> x % 2 = 0)
    |> Seq.map (fun x -> x * x)
    |> Seq.fold (fun acc x -> acc + x) 0

// ============================================================================
// PART 7: Manual Consumption Functions (For Comparison)
// Verifies Seq.fold behaves equivalently to manual for-loop
// ============================================================================

/// Manually sum a sequence (no Seq.fold)
let manualSum (s: seq<int>) : int =
    let mutable total = 0
    for x in s do
        total <- total + x
    total

/// Manually count a sequence
let manualCount (s: seq<int>) : int =
    let mutable count = 0
    for _ in s do
        count <- count + 1
    count

// ============================================================================
// PART 8: Closures with Captures (CRITICAL for flat closure model)
// Tests that mapper/predicate closures correctly capture values
// Closure struct: {code_ptr, capture_0, capture_1, ...}
// ============================================================================

/// Seq.map with captured multiplier
/// Mapper closure: {code_ptr, factor: int}
let scale (factor: int) (xs: seq<int>) = 
    Seq.map (fun x -> x * factor) xs

let scaledBy3 = scale 3 (naturals 5)
let scaledBy7 = scale 7 (naturals 3)

/// Seq.filter with captured threshold
/// Predicate closure: {code_ptr, threshold: int}
let aboveThreshold (threshold: int) (xs: seq<int>) =
    Seq.filter (fun x -> x > threshold) xs

let above5 = aboveThreshold 5 (naturals 10)
let above0 = aboveThreshold 0 (naturals 5)

/// Seq.fold with captured offset added to each element
/// Folder closure: {code_ptr, offset: int}
let sumWithOffset (offset: int) (xs: seq<int>) =
    Seq.fold (fun acc x -> acc + x + offset) 0 xs

let sumPlus10Each = sumWithOffset 10 (naturals 5)

/// Multiple captures in mapper
/// Mapper closure: {code_ptr, lo: int, hi: int}
let rangeTransform (lo: int) (hi: int) (xs: seq<int>) =
    Seq.map (fun x -> x * lo + hi) xs

let transformed = rangeTransform 2 100 (naturals 3)

// ============================================================================
// PART 9: Seq.collect (flatMap)
// Maps each element to a sequence, then flattens
// CollectSeq: {state, current, moveNext_ptr, outer_seq, mapper, inner_seq_state}
// ============================================================================

/// Basic flatMap: expand each number to itself and its double
let expandDouble = Seq.collect (fun x -> seq { yield x; yield x * 2 }) (naturals 3)

/// Expand each to itself three times
let expandTriple = Seq.collect (fun x -> seq { yield x; yield x; yield x }) (naturals 2)

/// flatMap with captured multiplier
let expandWithFactor (factor: int) (xs: seq<int>) =
    Seq.collect (fun x -> seq { yield x; yield x * factor }) xs

let expandBy10 = expandWithFactor 10 (naturals 3)

/// flatMap producing variable-length sequences based on value
let repeatN (xs: seq<int>) =
    Seq.collect (fun x -> seq {
        let mutable i = 0
        while i < x do
            yield x
            i <- i + 1
    }) xs

let repeated = repeatN (range 1 4)

// ============================================================================
// PART 10: Composed Pipelines with Captures
// Tests that captures at different pipeline stages don't interfere
// Each wrapper has its own closure with its own captures
// ============================================================================

/// Pipeline where each operation captures a different value
let complexPipeline (threshold: int) (multiplier: int) (count: int) =
    naturals 20
    |> Seq.filter (fun x -> x > threshold)      // captures 'threshold'
    |> Seq.map (fun x -> x * multiplier)        // captures 'multiplier'
    |> Seq.take count                            // captures 'count' (in TakeSeq.remaining)

let pipelineResult = complexPipeline 5 2 5

/// Fold at the end with captures throughout
let sumFilteredScaled (minVal: int) (factor: int) (xs: seq<int>) =
    xs
    |> Seq.filter (fun x -> x >= minVal)        // captures 'minVal'
    |> Seq.map (fun x -> x * factor)            // captures 'factor'
    |> Seq.fold (fun acc x -> acc + x) 0

let totalFiltered = sumFilteredScaled 3 10 (naturals 5)

// ============================================================================
// PART 11: Edge Cases
// Boundary conditions that must be handled gracefully
// ============================================================================

/// Empty source sequence
let emptySource = seq { if false then yield 0 }

let mapEmpty = Seq.map (fun x -> x * 2) emptySource
let filterEmpty = Seq.filter (fun x -> x > 0) emptySource
let foldEmpty = Seq.fold (fun acc x -> acc + x) 42 emptySource

/// Single element sequence
let singleElement = seq { yield 99 }
let mapSingle = Seq.map (fun x -> x + 1) singleElement

/// Filter that removes all elements
let filterNone = Seq.filter (fun x -> x > 100) (naturals 10)

/// Take zero elements
let takeZero = Seq.take 0 (naturals 10)

/// Take from empty sequence
let takeFromEmpty = Seq.take 5 emptySource

// ============================================================================
// PART 12: Deep Composition (3+ Operations)
// Validates struct nesting works at depth 4-5
// ============================================================================

/// Four operations chained
let deepPipeline1 =
    naturals 50
    |> Seq.filter (fun x -> x % 2 = 0)    // evens: 2 4 6 8 ... 50
    |> Seq.map (fun x -> x / 2)           // halved: 1 2 3 4 ... 25
    |> Seq.filter (fun x -> x % 3 = 0)    // div by 3: 3 6 9 12 15 18 21 24
    |> Seq.take 5

/// Five operations ending with fold
let deepPipelineWithFold =
    naturals 100
    |> Seq.filter (fun x -> x % 5 = 0)    // 5 10 15 20 ... 100 (20 elements)
    |> Seq.map (fun x -> x * 2)           // 10 20 30 40 ... 200
    |> Seq.filter (fun x -> x > 50)       // 60 70 80 ... 200 (15 elements)
    |> Seq.take 5                          // 60 70 80 90 100
    |> Seq.fold (fun acc x -> acc + x) 0

// ============================================================================
// ENTRY POINT
// ============================================================================

[<EntryPoint>]
let main _ =
    Console.writeln "=== Sample 16: Sequence Operations ==="
    Console.writeln ""

    // ----- Part 2: Seq.map (No Captures) -----
    Console.writeln "--- Part 2: Seq.map ---"

    Console.write "naturals 5: "
    for x in naturals 5 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "doubled (x*2): "
    for x in doubled do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "squared (x*x): "
    for x in squared do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "addTen (x+10): "
    for x in addTen do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 3: Seq.filter (No Captures) -----
    Console.writeln "--- Part 3: Seq.filter ---"

    Console.write "evens from 1..10: "
    for x in evens do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "odds from 1..10: "
    for x in odds do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "greaterThan5 from 1..10: "
    for x in greaterThan5 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 4: Seq.take -----
    Console.writeln "--- Part 4: Seq.take ---"

    Console.write "firstThree from 1..100: "
    for x in firstThree do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "exactlyFive from 1..5: "
    for x in exactlyFive do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "takeMoreThanAvailable: "
    for x in takeMoreThanAvailable do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 5: Seq.fold (No Captures) -----
    Console.writeln "--- Part 5: Seq.fold ---"

    Console.write "sum of 1..10: "
    Console.writeln (Format.int sum)

    Console.write "product of 1..5: "
    Console.writeln (Format.int product)

    Console.write "max of 1..10: "
    Console.writeln (Format.int findMax)

    Console.write "count of 1..10: "
    Console.writeln (Format.int countElements)
    Console.writeln ""

    // ----- Part 6: Composed Operations (No Captures) -----
    Console.writeln "--- Part 6: Composed Operations ---"

    Console.write "evensSquared (filter even, map x*x): "
    for x in evensSquared do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "squaresOver10 (map x*x, filter >10): "
    for x in squaresOver10 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "first3EvensDoubled: "
    for x in first3EvensDoubled do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "sumEvenSquares: "
    Console.writeln (Format.int sumEvenSquares)
    Console.writeln ""

    // ----- Part 7: Manual vs Seq.fold -----
    Console.writeln "--- Part 7: Manual vs Seq.fold ---"

    Console.write "manualSum 1..10: "
    Console.writeln (Format.int (manualSum (naturals 10)))

    Console.write "manualCount 1..10: "
    Console.writeln (Format.int (manualCount (naturals 10)))
    Console.writeln ""

    // ----- Part 8: Closures with Captures -----
    Console.writeln "--- Part 8: Closures with Captures ---"

    Console.write "scaledBy3: "
    for x in scaledBy3 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "scaledBy7: "
    for x in scaledBy7 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "above5: "
    for x in above5 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "above0: "
    for x in above0 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "sumPlus10Each: "
    Console.writeln (Format.int sumPlus10Each)

    Console.write "transformed (2*x+100): "
    for x in transformed do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 9: Seq.collect -----
    Console.writeln "--- Part 9: Seq.collect ---"

    Console.write "expandDouble: "
    for x in expandDouble do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "expandTriple: "
    for x in expandTriple do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "expandBy10: "
    for x in expandBy10 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "repeated: "
    for x in repeated do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""
    Console.writeln ""

    // ----- Part 10: Composed with Captures -----
    Console.writeln "--- Part 10: Composed with Captures ---"

    Console.write "complexPipeline 5 2 5: "
    for x in pipelineResult do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "sumFilteredScaled 3 10: "
    Console.writeln (Format.int totalFiltered)
    Console.writeln ""

    // ----- Part 11: Edge Cases -----
    Console.writeln "--- Part 11: Edge Cases ---"

    Console.write "mapEmpty: "
    for x in mapEmpty do
        Console.write (Format.int x)
    Console.writeln "(done)"

    Console.write "filterEmpty: "
    for x in filterEmpty do
        Console.write (Format.int x)
    Console.writeln "(done)"

    Console.write "foldEmpty: "
    Console.writeln (Format.int foldEmpty)

    Console.write "mapSingle: "
    for x in mapSingle do
        Console.write (Format.int x)
    Console.writeln ""

    Console.write "filterNone: "
    for x in filterNone do
        Console.write (Format.int x)
    Console.writeln "(done)"

    Console.write "takeZero: "
    for x in takeZero do
        Console.write (Format.int x)
    Console.writeln "(done)"

    Console.write "takeFromEmpty: "
    for x in takeFromEmpty do
        Console.write (Format.int x)
    Console.writeln "(done)"
    Console.writeln ""

    // ----- Part 12: Deep Composition -----
    Console.writeln "--- Part 12: Deep Composition ---"

    Console.write "deepPipeline1: "
    for x in deepPipeline1 do
        Console.write (Format.int x)
        Console.write " "
    Console.writeln ""

    Console.write "deepPipelineWithFold: "
    Console.writeln (Format.int deepPipelineWithFold)
    Console.writeln ""

    // ----- Verification Summary -----
    Console.writeln "--- Verification Summary ---"
    Console.writeln "Expected values:"
    Console.writeln "  naturals 5: 1 2 3 4 5"
    Console.writeln "  doubled: 2 4 6 8 10"
    Console.writeln "  squared: 1 4 9 16 25"
    Console.writeln "  addTen: 11 12 13 14 15"
    Console.writeln "  evens: 2 4 6 8 10"
    Console.writeln "  odds: 1 3 5 7 9"
    Console.writeln "  greaterThan5: 6 7 8 9 10"
    Console.writeln "  firstThree: 1 2 3"
    Console.writeln "  exactlyFive: 1 2 3 4 5"
    Console.writeln "  takeMoreThanAvailable: 1 2 3"
    Console.writeln "  sum: 55"
    Console.writeln "  product: 120"
    Console.writeln "  max: 10"
    Console.writeln "  count: 10"
    Console.writeln "  evensSquared: 4 16 36 64 100"
    Console.writeln "  squaresOver10: 16 25 36 49 64 81 100"
    Console.writeln "  first3EvensDoubled: 4 8 12"
    Console.writeln "  sumEvenSquares: 220"
    Console.writeln "  manualSum: 55"
    Console.writeln "  manualCount: 10"
    Console.writeln "  scaledBy3: 3 6 9 12 15"
    Console.writeln "  scaledBy7: 7 14 21"
    Console.writeln "  above5: 6 7 8 9 10"
    Console.writeln "  above0: 1 2 3 4 5"
    Console.writeln "  sumPlus10Each: 65"
    Console.writeln "  transformed: 102 104 106"
    Console.writeln "  expandDouble: 1 2 2 4 3 6"
    Console.writeln "  expandTriple: 1 1 1 2 2 2"
    Console.writeln "  expandBy10: 1 10 2 20 3 30"
    Console.writeln "  repeated: 1 2 2 3 3 3 4 4 4 4"
    Console.writeln "  complexPipeline: 12 14 16 18 20"
    Console.writeln "  sumFilteredScaled: 120"
    Console.writeln "  foldEmpty: 42"
    Console.writeln "  mapSingle: 100"
    Console.writeln "  deepPipeline1: 3 6 9 12 15"
    Console.writeln "  deepPipelineWithFold: 400"

    0
