/// Sample 06: Interactive Discriminated Union with Parsing
/// Demonstrates:
/// - User input via Console.readln (TWO inputs)
/// - String-to-number parsing via Parse intrinsics (int and float)
/// - Discriminated unions for runtime type representation
/// - Pattern matching for type-aware formatting
/// - String.contains for type detection
module AddNumbers

// Discriminated union with two cases
type Number =
    | IntVal of int
    | FloatVal of float

// Format a Number for display
let formatNumber (n: Number) : string =
    match n with
    | IntVal x -> Format.int x
    | FloatVal x -> Format.float x

// Parse string to Number - detect type by decimal point
let parseNumber (s: string) : Number =
    if String.contains s '.' then
        FloatVal (Parse.float s)
    else
        IntVal (Parse.int s)

// Add two Numbers - promotes to float if either is float
let addNumbers (a: Number) (b: Number) : Number =
    match a, b with
    | IntVal x, IntVal y -> IntVal (x + y)
    | IntVal x, FloatVal y -> FloatVal ((float x) + y)
    | FloatVal x, IntVal y -> FloatVal (x + (float y))
    | FloatVal x, FloatVal y -> FloatVal (x + y)

// Demo entry point
let runDemo () : int =
    Console.writeln "=== Interactive Number Addition ==="
    Console.writeln "Enter any numeric form: integers (42) or decimals (3.14)"
    Console.writeln ""

    // Get first number from user
    Console.write "Enter first number: "
    let input1 = Console.readln ()
    let n1 = parseNumber input1

    // Get second number from user
    Console.write "Enter second number: "
    let input2 = Console.readln ()
    let n2 = parseNumber input2

    // Add the numbers
    let result = addNumbers n1 n2

    // Display: "X + Y = Z"
    Console.write (formatNumber n1)
    Console.write " + "
    Console.write (formatNumber n2)
    Console.write " = "
    Console.writeln (formatNumber result)

    0

[<EntryPoint>]
let main argv =
    runDemo ()
