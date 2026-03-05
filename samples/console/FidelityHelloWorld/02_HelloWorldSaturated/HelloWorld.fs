module Examples.HelloWorldSaturated

open Console

/// Demonstrates SATURATED function calls - all arguments provided at once.
/// Uses CCS Console intrinsics (Console.write, Console.readln, Console.writeln).
let hello() =
    Console.write "Enter your name: "
    let name = Console.readln()
    Console.writeln $"Hello, {name}!"

[<EntryPoint>]
let main argv =
    hello()
    0
