module Examples.HelloWorldFullCurried

/// Demonstrates FULL CURRYING patterns:
/// - Curried function with multiple parameters
/// - Partial application: binding first argument to create specialized function
/// - Applying remaining argument to complete the call

/// Curried greeting function - takes prefix then name
let greet prefix =
    fun name ->
        Console.writeln $"{prefix}, {name}!"

/// Partial application: bind "Hello" to create a specialized greeter
let helloGreeter = greet "Hello"

[<EntryPoint>]
let main argv =
    Console.write "Enter your name: "
    let name = Console.readln()

    // Apply remaining argument to the partially applied function
    helloGreeter name

    0
