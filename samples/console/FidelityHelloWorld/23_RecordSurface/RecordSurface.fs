/// Sample 23: the record surface — a record array reached through a let-bound field of
/// another record's element, a record literal whose labels several records share, a
/// module-level value whose initializer has local bindings and calls, and the
/// short-circuit evaluation of `||` and `&&`.
module Examples.RecordSurface

open Console

type Bit = { Name: string; Position: int }
type Fld = { Name: string; Offset: int; Bits: Bit array }

type N = { Name: string; Count: int }
type F = { Name: string; Count: int; Extra: int }

type Space = { Name: string; Capacity: int64 }
type Desc = { Id: string; Spaces: Space array }

let show (label: string) (n: int) : unit =
    Console.writeln (String.concat2 label (Format.int n))

let showBool (label: string) (b: bool) : unit =
    Console.writeln (String.concat2 label (if b then "true" else "false"))

module Layout =
    /// Sum of every bit position under the fields: the inner array is reached through a
    /// let-bound field of an element of the outer array.
    let sumPositions (fields: Fld array) : int =
        let n = Array.length fields
        let mutable total = 0
        let mutable i = 0
        while i < n do
            let f = Array.get fields i
            let bits = f.Bits
            let bn = Array.length bits
            let mutable b = 0
            while b < bn do
                let bf = Array.get bits b
                total <- total + bf.Position
                b <- b + 1
            i <- i + 1
        total

module Records =
    let mkN (n: string) (c: int) : N =
        let r : N = { Name = n; Count = c }
        r
    let mkF (n: string) (c: int) (e: int) : F = { Name = n; Count = c; Extra = e }

module Spaces =
    let create (name: string) (capacity: int64) : Space = { Name = name; Capacity = capacity }
    /// A module-level value whose initializer binds locals and calls functions.
    let linux : Desc =
        let a = create "stack" 8388608L
        let b = create "arena" 4096L
        let spaces = [| a; b |]
        { Id = "linux-x86_64"; Spaces = spaces }
    let total (d: Desc) : int64 =
        let mutable acc = 0L
        let mutable i = 0
        while i < Array.length d.Spaces do
            acc <- acc + (Array.get d.Spaces i).Capacity
            i <- i + 1
        acc

module Guards =
    /// The right operand runs only when the left one does not decide.
    let orDiv (a: int) (b: int) : bool = a = 0 || (10 / b) = 5
    let andDiv (a: int) (b: int) : bool = a = 1 && (10 / b) = 5

[<EntryPoint>]
let main argv =
    Console.writeln "=== Record Surface Test ==="
    let bits : Bit array = [| { Name = "a"; Position = 3 }; { Name = "b"; Position = 4 } |]
    let none : Bit array = Array.zeroCreate 0
    let fields : Fld array = [| { Name = "x"; Offset = 0; Bits = bits }; { Name = "y"; Offset = 4; Bits = none } |]
    show "sum positions: " (Layout.sumPositions fields)
    let n = Records.mkN "x" 3
    let f = Records.mkF "y" 4 5
    show "N count: " n.Count
    show "F extra: " f.Extra
    show "linux spaces: " (Array.length Spaces.linux.Spaces)
    show "linux total: " (int (Spaces.total Spaces.linux))
    showBool "orDiv 0 0: " (Guards.orDiv 0 0)
    showBool "andDiv 0 0: " (Guards.andDiv 0 0)
    showBool "orDiv 1 2: " (Guards.orDiv 1 2)
    showBool "andDiv 1 2: " (Guards.andDiv 1 2)
    0
