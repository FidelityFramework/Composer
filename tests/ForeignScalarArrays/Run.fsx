#load "Driver.fsx"

open Driver

let arguments = fsi.CommandLineArgs |> Array.skip 1 |> Array.filter ((<>) "--") |> Array.toList
let code =
    try
        match parseArguments arguments with
        | Ok None -> printfn "%s" usage; 0
        | Ok(Some options) -> run options |> fun pending -> pending.GetAwaiter().GetResult()
        | Error message -> eprintfn "%s" message; 2
    with ex -> eprintfn "%s" ex.Message; 1
exit code
