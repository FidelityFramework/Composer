/// Sample 16: Sequence Operations
/// Higher-order functions over sequences: map, filter, take, fold
/// PRD-16: Builds on SimpleSeq (PRD-15) and HOFs (PRD-12)
module SeqOperationsSample

// ============================================================================
// PART 1: Source Sequences (from Sample 15)
// ============================================================================

/// Simple range sequence
let range (start: int) (stop: int) = seq {
    let mutable i = start
    while i <= stop do
        yield i
        i <- i + 1
}

/// First N natural numbers
let naturals (n: int) = range 1 n

// ============================================================================
// PART 2: Seq.map - Transform Each Element
// ============================================================================

/// Double every element
let doubled = Seq.map (fun x -> x * 2) (naturals 5)

/// Square every element
let squared = Seq.map (fun x -> x * x) (naturals 5)

/// Add constant to every element
let addTen = Seq.map (fun x -> x + 10) (naturals 5)

// ============================================================================
// PART 3: Seq.filter - Select Matching Elements
// ============================================================================

/// Filter for even numbers
let evens = Seq.filter (fun x -> x % 2 = 0) (naturals 10)

/// Filter for odd numbers
let odds = Seq.filter (fun x -> x % 2 = 1) (naturals 10)

/// Filter for numbers greater than 5
let greaterThan5 = Seq.filter (fun x -> x > 5) (naturals 10)

// ============================================================================
// PART 4: Seq.take - Limit to First N Elements
// ============================================================================

/// Take first 3 from a longer sequence
let firstThree = Seq.take 3 (naturals 100)

/// Take first 5 from exactly 5 (boundary)
let exactlyFive = Seq.take 5 (naturals 5)

// ============================================================================
// PART 5: Seq.fold - Reduce to Single Value
// ============================================================================

/// Sum all elements
let sum = Seq.fold (fun acc x -> acc + x) 0 (naturals 10)

/// Product of all elements
let product = Seq.fold (fun acc x -> acc * x) 1 (naturals 5)

/// Find maximum (left fold with comparison)
let findMax = Seq.fold (fun acc x -> if x > acc then x else acc) 0 (naturals 10)

/// Count elements
let countElements = Seq.fold (fun acc _ -> acc + 1) 0 (naturals 10)

// ============================================================================
// PART 6: Composed Operations (Pipelines)
// ============================================================================

/// Filter then map: evens squared
let evensSquared =
    naturals 10
    |> Seq.filter (fun x -> x % 2 = 0)
    |> Seq.map (fun x -> x * x)

/// Map then filter: squares > 10
let squaresOver10 =
    naturals 10
    |> Seq.map (fun x -> x * x)
    |> Seq.filter (fun x -> x > 10)

/// Filter, map, take: first 3 even numbers doubled
let first3EvensDoubled =
    naturals 100
    |> Seq.filter (fun x -> x % 2 = 0)
    |> Seq.map (fun x -> x * 2)
    |> Seq.take 3

/// Sum of squares of evens
let sumEvenSquares =
    naturals 10
    |> Seq.filter (fun x -> x % 2 = 0)
    |> Seq.map (fun x -> x * x)
    |> Seq.fold (fun acc x -> acc + x) 0

// ============================================================================
// PART 7: Manual Consumption Functions (For Comparison)
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
// ENTRY POINT
// ============================================================================

[<EntryPoint>]
let main _ =
    Console.writeln "=== Sample 16: Sequence Operations ==="
    Console.writeln ""

    // ----- Part 2: Seq.map -----
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

    // ----- Part 3: Seq.filter -----
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
    Console.writeln ""

    // ----- Part 5: Seq.fold -----
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

    // ----- Part 6: Composed Operations -----
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
    Console.writeln "  sum: 55"
    Console.writeln "  product: 120"
    Console.writeln "  max: 10"
    Console.writeln "  count: 10"
    Console.writeln "  evensSquared: 4 16 36 64 100"
    Console.writeln "  squaresOver10: 16 25 36 49 64 81 100"
    Console.writeln "  first3EvensDoubled: 4 8 12"
    Console.writeln "  sumEvenSquares: 220 (4+16+36+64+100)"

    0
