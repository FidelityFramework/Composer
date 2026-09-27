module CLI.Commands.VerifyCommand

open System
open Argu

/// Command line arguments for the verify command
type VerifyArgs =
    | Binary of string
    | No_Heap
    | Max_Stack of int
    | Show_Symbol_Deps
with
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Binary _ -> "Path to binary to verify (required)"
            | No_Heap -> "Verify the binary uses zero heap allocations"
            | Max_Stack n -> "Verify the binary's maximum stack usage is below limit"
            | Show_Symbol_Deps -> "Show all external symbol dependencies"

/// Verifies a compiled binary meets the specified constraints
let verify (args: ParseResults<VerifyArgs>) =
    // Parse arguments
    let binaryPath = 
        match args.TryGetResult Binary with
        | Some path -> path
        | None -> 
            printfn "Error: Binary path is required"
            exit 1
    
    // No binary analysis exists behind these checks. Reporting a verdict without
    // evidence would be a fabricated pass, so every requested check is refused.
    let requested =
        [ if args.Contains No_Heap then yield "zero heap allocations"
          match args.TryGetResult Max_Stack with
          | Some limit -> yield sprintf "maximum stack usage below %d bytes" limit
          | None -> ()
          if args.Contains Show_Symbol_Deps then yield "external symbol dependencies" ]

    printfn "Error: composer verify has no binary analysis; it cannot establish %s for %s"
        (if requested.IsEmpty then "any property" else String.concat ", " requested) binaryPath
    1