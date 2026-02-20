/// HelloFPGA - Minimal FPGA target test
/// Validates CIRCT backend routing: combinational logic + Mealy machine extraction
module Examples.HelloFPGA

/// Simple adder — pure combinational logic suitable for hardware synthesis
let add (a: int) (b: int) : int =
    a + b

/// Multiplexer — select one of two values based on condition
let mux (sel: bool) (a: int) (b: int) : int =
    if sel then a else b

/// Hardware state type — each field becomes a seq.compreg register
type State = { Counter: int; Value: int }

/// Pure step function — combinational logic (next state from current state)
let step (s: State) : State =
    { Counter = s.Counter + 1; Value = s.Value }

/// Get the counter value from state
let getCounter (s: State) : int = s.Counter

/// Create a state from field values
let makeState (c: int) (v: int) : State = { Counter = c; Value = v }

/// Mealy machine design — compile-time hardware descriptor
/// InitialState → register reset values, Step → combinational logic block
type Design<'S> = {
    InitialState: 'S
    Step: 'S -> 'S
}

/// Hardware module binding — the FPGA top-level circuit
/// Generates hw.module with seq.compreg registers + hw.instance of step
[<HardwareModule>]
let counter : Design<State> = {
    InitialState = { Counter = 0; Value = 42 }
    Step = step
}

/// CPU entry point — validates mixed CPU/FPGA compilation
[<EntryPoint>]
let main (_argv: string array) =
    add 3 4
