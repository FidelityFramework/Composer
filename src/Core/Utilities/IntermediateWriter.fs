namespace Core.Utilities

open System.IO
open System.Text.Json
open System.Text.Json.Serialization

module IntermediateWriter =

    /// A scratch file for one compile, under a directory private to this process. A compile that
    /// keeps no intermediates still writes MLIR and IR to disk for the tools; a fixed name under the
    /// shared temp directory let two concurrent compiles link each other's IR without a word
    /// (found in CS-9's review). The process id makes the path this compile's own.
    let scratchPath (name: string) : string =
        let directory = Path.Combine(Path.GetTempPath(), $"composer-{System.Environment.ProcessId}")
        Directory.CreateDirectory directory |> ignore
        Path.Combine(directory, name)

    /// Simple file writer that takes a full path and content
    let writeFileToPath (filePath: string) (content: string) : unit =
        try
            // Ensure directory exists
            let directory = Path.GetDirectoryName(filePath)
            if not (Directory.Exists(directory)) then
                Directory.CreateDirectory(directory) |> ignore

            // Write the file
            File.WriteAllText(filePath, content)
            printfn "  Wrote %s (%d bytes)" (Path.GetFileName(filePath)) content.Length
        with ex ->
            printfn "  Warning: Could not write %s: %s" filePath ex.Message

    /// JSON options for consistent serialization (camelCase, no F# unions)
    let jsonOptions =
        let options = JsonSerializerOptions(WriteIndented = true)
        options.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
        options

    /// JSON options with F# type support (for F# union types, records, etc.)
    let jsonOptionsWithFSharpSupport =
        let options = JsonSerializerOptions(WriteIndented = true)
        options.Converters.Add(JsonFSharpConverter())
        options

    /// Write data as JSON to specified filename in output directory
    let writeJsonAsset (outputDir: string) (filename: string) (data: obj) : unit =
        let json = JsonSerializer.Serialize(data, jsonOptions)
        let outputPath = Path.Combine(outputDir, filename)
        writeFileToPath outputPath json