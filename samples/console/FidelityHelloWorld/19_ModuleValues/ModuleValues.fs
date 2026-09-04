/// Sample 19: module-level values across modules and functions.
module Examples.ModuleValues

open Console
open Examples.ModuleValuesLib

let show (label: string) (n: int) : unit =
    Console.writeln (String.concat2 label (Format.int n))

/// A module-level value in the entry module, referenced from another function.
let k : int = 42
let useK (x: int) : int = x + k

[<EntryPoint>]
let main argv =
    Console.writeln "=== Module Values Test ==="
    Console.writeln (String.concat2 "rodata name: " Spaces.rodata.Name)
    show "rodata capacity: " (int Spaces.rodata.Capacity)
    show "wordSize: " Spaces.wordSize
    Console.writeln (String.concat2 "first of all: " (Spaces.first Spaces.all))
    show "total capacity: " (int (Spaces.total Spaces.all))
    show "bump 1: " (Spaces.bump ())
    show "bump 2: " (Spaces.bump ())
    show "counter: " Spaces.counter
    show "k: " k
    show "useK 1: " (useK 1)
    0
