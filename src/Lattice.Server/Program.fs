module Lattice.Server.Program

open System
open StreamJsonRpc

[<EntryPoint>]
let main argv =
    try
        let option name fallback =
            argv |> Array.tryFindIndex ((=) name)
            |> Option.map (fun i -> if i + 1 >= argv.Length then invalidArg name "Missing value" else argv[i + 1])
            |> Option.defaultValue fallback
        let project = option "--project" ""
        let solver = option "--solver" "cvc5"
        let output = Console.OpenStandardOutput()
        // Compiler tracing belongs on stderr; stdout is reserved for LSP framing.
        Console.SetOut(Console.Error)
        use formatter = new SystemTextJsonFormatter()
        use handler = new HeaderDelimitedMessageHandler(output, Console.OpenStandardInput(), formatter)
        use rpc = new JsonRpc(handler)
        use server = new Server(rpc, project, solver)
        rpc.AddLocalRpcTarget(server)
        rpc.CancelLocallyInvokedMethodsWhenConnectionIsClosed <- true
        rpc.StartListening()
        rpc.Completion.GetAwaiter().GetResult()
        0
    with ex ->
        Console.Error.WriteLine(ex.Message)
        1
