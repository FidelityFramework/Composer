/// Sample 13: Simple Recursion (without mutual recursion)
module RecursionSample

/// Classic factorial - not tail recursive
let rec factorial (n: int) : int =
    if n <= 1 then 1
    else n * factorial (n - 1)

/// Tail-recursive factorial with accumulator
let factorialTail (n: int) : int =
    let rec loop acc n =
        if n <= 1 then acc
        else loop (acc * n) (n - 1)
    loop 1 n

/// Count digits in a number
let rec countDigits (n: int) : int =
    if n < 10 then 1
    else 1 + countDigits (n / 10)

/// GCD using Euclidean algorithm - naturally tail recursive
let rec gcd (a: int) (b: int) : int =
    if b = 0 then a
    else gcd b (a % b)

/// Power function - a^n
let rec power (a: int) (n: int) : int =
    if n = 0 then 1
    elif n = 1 then a
    else a * power a (n - 1)

[<EntryPoint>]
let main _ =
    Console.writeln "=== Simple Recursion Test ==="

    // Factorial
    Console.writeln "--- Factorial ---"
    Console.write "factorial 5: "
    Console.writeln (Format.int (factorial 5))  // 120
    Console.write "factorialTail 5: "
    Console.writeln (Format.int (factorialTail 5))  // 120
    Console.write "factorial 10: "
    Console.writeln (Format.int (factorial 10))  // 3628800

    // Count digits
    Console.writeln ""
    Console.writeln "--- Count Digits ---"
    Console.write "digits in 12345: "
    Console.writeln (Format.int (countDigits 12345))  // 5
    Console.write "digits in 7: "
    Console.writeln (Format.int (countDigits 7))  // 1

    // GCD
    Console.writeln ""
    Console.writeln "--- GCD ---"
    Console.write "gcd 48 18: "
    Console.writeln (Format.int (gcd 48 18))  // 6
    Console.write "gcd 100 35: "
    Console.writeln (Format.int (gcd 100 35))  // 5

    // Power
    Console.writeln ""
    Console.writeln "--- Power ---"
    Console.write "2^10: "
    Console.writeln (Format.int (power 2 10))  // 1024
    Console.write "3^5: "
    Console.writeln (Format.int (power 3 5))  // 243

    0
