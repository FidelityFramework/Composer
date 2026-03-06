/// Sample 10a: Imperative Control Flow
/// Validates mutable bindings, assignment (<-), and while loops.
///
/// These capabilities have witness implementations but have never been
/// validated end-to-end through the XParsec pipeline:
///   - Mutable bindings: BindingWitness.fs (pBuildMutableBinding)
///   - Assignment: MutableAssignmentWitness.fs (pStoreMutableVariable)
///   - While loops: ControlFlowWitness.fs (pBuildWhileLoop)
///
/// Dependencies: comparisons + booleans (validated by sample 07)
module ImperativeControl

open Console
open Format

[<EntryPoint>]
let main _ =
    Console.writeln "=== Imperative Control Test ==="

    // ─── Mutable Bindings + Assignment ───
    Console.writeln "--- Mutable Bindings ---"

    let mutable x = 0
    Console.write "initial: "
    Console.writeln (Format.int x)

    x <- x + 1
    Console.write "after increment: "
    Console.writeln (Format.int x)

    x <- x + 10
    Console.write "after add 10: "
    Console.writeln (Format.int x)

    // ─── While Loop ───
    Console.writeln ""
    Console.writeln "--- While Loop ---"

    // Sum 1 to 10
    let mutable sum = 0
    let mutable i = 1
    while i <= 10 do
        sum <- sum + i
        i <- i + 1
    Console.write "sum 1 to 10: "
    Console.writeln (Format.int sum)

    // Countdown
    Console.write "countdown: "
    let mutable n = 5
    while n >= 1 do
        Console.write (Format.int n)
        if n > 1 then
            Console.write " "
        n <- n - 1
    Console.writeln ""

    // ─── Nested While ───
    Console.writeln ""
    Console.writeln "--- Nested While ---"
    Console.writeln "multiplication table 3x3:"

    let mutable row = 1
    while row <= 3 do
        let mutable col = 1
        while col <= 3 do
            Console.write (Format.int (row * col))
            if col < 3 then
                Console.write " "
            col <- col + 1
        Console.writeln ""
        row <- row + 1

    0
