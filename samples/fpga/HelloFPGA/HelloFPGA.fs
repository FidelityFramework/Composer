/// HelloFPGA - Minimal FPGA target test
/// Pure combinational logic — validates CIRCT backend routing
module Examples.HelloFPGA

/// Simple adder — pure combinational logic suitable for hardware synthesis
let add (a: int) (b: int) : int =
    a + b

/// Multiplexer — select one of two values based on condition
let mux (sel: bool) (a: int) (b: int) : int =
    if sel then a else b

/// Top-level entry point (placeholder until HardwareModule attribute exists)
[<EntryPoint>]
let main (_argv: string array) =
    add 3 4
