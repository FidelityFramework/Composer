/// Sample 18: let-polymorphism for top-level functions.
/// A module-level generic function is generalized to a type scheme and compiled once per
/// distinct instantiation (monomorphization), so one definition serves several types.
module Examples.Generalization

open Console

type Pair = { A: int; B: int }

module Gen =
    /// Generic pass-through: instantiated at int, string and a record type below.
    let firstOf (items: 'a array) (fallback: 'a) : 'a =
        if Array.length items > 0 then Array.get items 0 else fallback

    /// Generic higher-order combinator: instantiated at int and int64 below.
    let applyTwice (f: 'a -> 'a) (x: 'a) : 'a =
        f (f x)

    /// Generic option construction: instantiated at int and string below.
    let wrap (v: 'a) : 'a option =
        Some v

    let incr (n: int) : int = n + 1
    let dbl (v: int64) : int64 = v * 2L

[<EntryPoint>]
let main argv =
    Console.writeln "=== Generalization Test ==="
    let a = Gen.firstOf [| 5; 6 |] 0
    let b = Gen.firstOf [| "x"; "y" |] "none"
    let c = Gen.firstOf [| { A = 1; B = 2 } |] { A = 0; B = 0 }
    let none = Gen.firstOf (Array.zeroCreate<int> 0) 42
    Console.writeln (String.concat2 "firstOf int: " (Format.int a))
    Console.writeln (String.concat2 "firstOf string: " b)
    Console.writeln (String.concat2 "firstOf record: " (Format.int (c.A + c.B)))
    Console.writeln (String.concat2 "firstOf empty: " (Format.int none))
    Console.writeln (String.concat2 "applyTwice incr 1: " (Format.int (Gen.applyTwice Gen.incr 1)))
    Console.writeln (String.concat2 "applyTwice dbl 3: " (Format.int (int (Gen.applyTwice Gen.dbl 3L))))
    let o = Gen.wrap 5
    let v = match o with | Some x -> x | None -> 0
    Console.writeln (String.concat2 "wrap int: " (Format.int v))
    let s = Gen.wrap "s"
    let t = match s with | Some x -> x | None -> "none"
    Console.writeln (String.concat2 "wrap string: " t)
    0
