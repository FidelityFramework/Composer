/// Sample 20: the array surface — Array.length, literals, indexers, blit, for loops,
/// arrays of records, and array literals in argument position.
module Examples.ArraySurface

open Console

type Point = { X: int; Y: int }

let show (label: string) (n: int) : unit =
    Console.writeln (String.concat2 label (Format.int n))

/// Array.length on a parameter, used in a let and in an expression.
let count (xs: int array) : int =
    let n = Array.length xs
    n + 0

/// Indexer read in a loop and indexer write.
let sum (data: byte array) : int =
    let mutable acc = 0
    let mutable i = 0
    while i < Array.length data do
        let b = data.[i]
        acc <- acc + int b
        i <- i + 1
    acc

/// for loop with a computed bound, up and down.
let triangle (n: int) : int =
    let mutable acc = 0
    for i = 0 to n - 1 do
        acc <- acc + i
    acc

let countdown (n: int) : int =
    let mutable acc = 0
    for i = n downto 1 do
        acc <- acc + i
    acc

/// Array.blit: copy a slice between arrays.
let copySlice () : int =
    let src = Array.zeroCreate<byte> 4
    Array.set src 0 5uy
    Array.set src 1 6uy
    let dst = Array.zeroCreate<byte> 8
    Array.blit src 0 dst 2 2
    int (Array.get dst 3)

/// Array literal in argument position, joined with a separator.
let join (sep: string) (parts: string array) : string =
    let mutable acc = ""
    let mutable i = 0
    while i < Array.length parts do
        acc <- (if i = 0 then Array.get parts i else String.concat2 (String.concat2 acc sep) (Array.get parts i))
        i <- i + 1
    acc

[<EntryPoint>]
let main argv =
    Console.writeln "=== Array Surface Test ==="
    let xs = [| 1; 2; 3 |]
    show "count [|1;2;3|]: " (count xs)
    show "first: " (Array.get xs 0)
    let data = Array.zeroCreate<byte> 4
    data.[0] <- 7uy
    data.[1] <- 9uy
    show "sum bytes: " (sum data)
    show "d0: " (int data.[0])
    show "triangle 4: " (triangle 4)
    show "countdown 3: " (countdown 3)
    show "blit dst[3]: " (copySlice ())
    let points = [| { X = 1; Y = 2 }; { X = 3; Y = 4 } |]
    let p = Array.get points 1
    show "points[1].Y: " p.Y
    show "points length: " (Array.length points)
    Console.writeln (join ", " [| "a"; "b"; "c" |])
    let names : string array = [| "x"; "y" |]
    Console.writeln (Array.get names 1)
    0
