/// Sample 21: records and tags — a mutable field assigned through a parameter, an
/// enumeration (nullary union) as a record field, three-arm matches in value position,
/// string-alias tags compared with = and <>, and a qualified module-value member access.
module Examples.RecordsAndTags

open Console

type Kind = string

[<RequireQualifiedAccess>]
module Kind =
    [<Literal>]
    let Stack: Kind = "stack"
    [<Literal>]
    let Arena: Kind = "arena"

type Space = { Name: string; Kind: Kind }

type FrameKind =
    | Tell
    | Ask
    | Reply

type Frame = { Kind: FrameKind; Correlation: uint32 }

type PairPayload = { A: int; B: int }

type Shape =
    | Leaf of int
    | Pair of PairPayload
    | Empty

/// A write cursor whose position is advanced through the parameter.
type Writer = { Data: byte array; mutable Position: int }

module Writer =
    let create (n: int) : Writer = { Data = Array.zeroCreate n; Position = 0 }
    let push (w: Writer) (b: byte) : unit =
        Array.set w.Data w.Position b
        w.Position <- w.Position + 1

module Frames =
    let mk (k: FrameKind) (c: uint32) : Frame = { Kind = k; Correlation = c }
    let isAsk (f: Frame) : bool =
        match f.Kind with
        | Ask -> true
        | _ -> false
    let index (k: FrameKind) : int =
        match k with
        | Tell -> 0
        | Ask -> 1
        | Reply -> 2

module Shapes =
    let weight (s: Shape) : int =
        match s with
        | Leaf n -> n
        | Pair p -> p.A + p.B
        | Empty -> 0
    let name (n: int) : string =
        match n with
        | 0 -> "zero"
        | 1 -> "one"
        | 2 -> "two"
        | _ -> "many"

module Config =
    let defaults : Space = { Name = "arena"; Kind = Kind.Arena }
    let isArena (s: Space) : bool = s.Kind = Kind.Arena
    let differs (a: string) (b: string) : bool = a <> b

let show (label: string) (n: int) : unit =
    Console.writeln (String.concat2 label (Format.int n))

let showBool (label: string) (b: bool) : unit =
    Console.writeln (String.concat2 label (if b then "true" else "false"))

[<EntryPoint>]
let main argv =
    Console.writeln "=== Records and Tags Test ==="
    let w = Writer.create 4
    Writer.push w 3uy
    Writer.push w 4uy
    show "writer position: " w.Position
    show "writer byte 1: " (int (Array.get w.Data 1))
    showBool "isAsk Ask: " (Frames.isAsk (Frames.mk Ask (uint32 7)))
    showBool "isAsk Reply: " (Frames.isAsk (Frames.mk Reply (uint32 9)))
    show "index Reply: " (Frames.index Reply)
    show "weight Leaf 7: " (Shapes.weight (Leaf 7))
    show "weight Pair 2 3: " (Shapes.weight (Pair { A = 2; B = 3 }))
    show "weight Empty: " (Shapes.weight Empty)
    Console.writeln (String.concat2 "name 2: " (Shapes.name 2))
    Console.writeln (String.concat2 "name 9: " (Shapes.name 9))
    showBool "defaults isArena: " (Config.isArena Config.defaults)
    showBool "stack isArena: " (Config.isArena { Name = "s"; Kind = Kind.Stack })
    showBool "differs x y: " (Config.differs "x" "y")
    showBool "differs x x: " (Config.differs "x" "x")
    Console.writeln (String.concat2 "defaults name: " Config.defaults.Name)
    0
