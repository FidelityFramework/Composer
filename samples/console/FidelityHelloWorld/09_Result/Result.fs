/// Sample 09: Result Type - Simplified
module ResultSample

[<EntryPoint>]
let main _ =
    Console.writeln "=== Result Type Test ==="

    // Test Ok case
    let okVal : Result<int, string> = Ok 42
    Console.write "Ok 42: "
    match okVal with
    | Ok x -> Console.writeln (Format.int x)
    | Error e -> Console.writeln e

    // Test Error case
    let errVal : Result<int, string> = Error "Something went wrong"
    Console.write "Error case: "
    match errVal with
    | Ok x -> Console.writeln (Format.int x)
    | Error e -> Console.writeln e

    Console.writeln "Done!"
    0
