/// Sample 05: Add Numbers
/// Demonstrates SRTP arithmetic and numeric-to-string conversion
/// Tests: numeric input, arithmetic operators, string formatting
module AddNumbers

// Read two numbers, add them, display result
let add() =
    Console.write "Enter first number: "
    let aStr = Console.readln()
    Console.write "Enter second number: "
    let bStr = Console.readln()

    // For now, just echo back the strings since we need Int.parse and Int.toString
    // These will be added as FNCS intrinsics
    Console.write "You entered: "
    Console.write aStr
    Console.write " and "
    Console.writeln bStr

[<EntryPoint>]
let main argv =
    add()
    0
