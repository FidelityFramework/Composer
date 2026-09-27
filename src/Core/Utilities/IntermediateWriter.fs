namespace Core.Utilities

open System.IO

module IntermediateWriter =

    /// A scratch file for one compile, under a directory private to this process. A compile that
    /// keeps no intermediates still writes MLIR and IR to disk for the tools; a fixed name under the
    /// shared temp directory let two concurrent compiles link each other's IR without a word
    /// (found in CS-9's review). The process id makes the path this compile's own.
    let scratchPath (name: string) : string =
        let directory = Path.Combine(Path.GetTempPath(), $"composer-{System.Environment.ProcessId}")
        Directory.CreateDirectory directory |> ignore
        Path.Combine(directory, name)
