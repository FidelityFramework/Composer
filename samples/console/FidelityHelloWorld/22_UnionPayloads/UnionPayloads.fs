/// Sample 22: union payloads — a string payload constructed from a parameter and read in
/// another function's match arm, a record payload, a union that mentions itself, a record
/// and a union defined together in an `and` group, two option payload types matched by
/// pattern in one program, and a function whose name carries an apostrophe.
module Examples.UnionPayloads

open Console

type Spec = { Length: int }

type T =
    | P of string
    | F of Spec
    | Q of int

type Node =
    | Leaf of int
    | Branch of Node

type Field = { Name: string; Type: Shape }
and Shape =
    | Prim of string
    | Struct of Field array

let show (label: string) (n: int) : unit =
    Console.writeln (String.concat2 label (Format.int n))

module Types =
    let prim (k: string) : T = P k
    let fixedOf (n: int) : T = F { Length = n }
    let count (n: int) : T = Q n
    let size (t: T) : int =
        match t with
        | P k -> String.length k
        | F spec -> spec.Length
        | Q n -> n

module Nodes =
    let rec depth (n: Node) : int =
        match n with
        | Leaf v -> v
        | Branch inner -> 1 + depth inner
    let build (k: int) : Node =
        let mutable n = Leaf 7
        let mutable i = 0
        while i < k do
            n <- Branch n
            i <- i + 1
        n

module Shapes =
    let rec count (s: Shape) : int =
        match s with
        | Prim _ -> 1
        | Struct fields -> countFields fields
    and countFields (fields: Field array) : int =
        let n = Array.length fields
        let mutable i = 0
        let mutable acc = 0
        while i < n do
            let f = Array.get fields i
            acc <- acc + count f.Type
            i <- i + 1
        acc
    let field (name: string) (t: Shape) : Field = { Name = name; Type = t }

module Options =
    let halfOf (v: int) : int64 option = if v % 2 = 0 then Some (int64 (v / 2)) else None
    let scaled (v: float) : float option = if v > 0.0 then Some (v * 2.0) else None
    let half' (v: int) : int =
        match halfOf v with
        | Some h -> int h
        | None -> -1
    let scale' (v: float) : int =
        match scaled v with
        | Some s -> int s
        | None -> -1

[<EntryPoint>]
let main argv =
    Console.writeln "=== Union Payloads Test ==="
    show "size (P \"u8\"): " (Types.size (Types.prim "u8"))
    show "size (F 16): " (Types.size (Types.fixedOf 16))
    show "size (Q 3): " (Types.size (Types.count 3))
    show "depth (build 3): " (Nodes.depth (Nodes.build 3))
    let a = Shapes.field "a" (Prim "u8")
    let b = Shapes.field "b" (Prim "u16")
    let inner = Shapes.field "s" (Struct [| a; b |])
    show "count nested: " (Shapes.count (Struct [| inner; a |]))
    show "half' 10: " (Options.half' 10)
    show "half' 7: " (Options.half' 7)
    show "scale' 1.5: " (Options.scale' 1.5)
    show "scale' -1.0: " (Options.scale' (0.0 - 1.0))
    0
