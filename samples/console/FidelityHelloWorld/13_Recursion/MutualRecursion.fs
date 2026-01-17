/// Minimal Mutual Recursion Test
module MutualRecursionSample

/// Mutual recursion example: even/odd determination
let rec isEven (n: int) : bool =
    if n = 0 then true
    else isOdd (n - 1)
and isOdd (n: int) : bool =
    if n = 0 then false
    else isEven (n - 1)

[<EntryPoint>]
let main _ =
    Console.writeln "=== Mutual Recursion Test ==="

    Console.write "isEven 10: "
    Console.writeln (if isEven 10 then "true" else "false")
    Console.write "isOdd 10: "
    Console.writeln (if isOdd 10 then "true" else "false")
    Console.write "isEven 7: "
    Console.writeln (if isEven 7 then "true" else "false")
    Console.write "isOdd 7: "
    Console.writeln (if isOdd 7 then "true" else "false")

    0
