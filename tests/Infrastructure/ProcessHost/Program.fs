module ProcessHost.Program

open System
open System.ComponentModel
open System.Diagnostics
open System.IO
open System.Runtime.InteropServices

module private Linux =
    [<DllImport("libc", EntryPoint = "setsid", SetLastError = true)>]
    extern int createSession()

let private error code message =
    eprintfn "ProcessHost: %s" message
    code

/// Atomic rename prevents the runner observing a partial identity or error.
let private writeAtomic (path: string) (contents: string) =
    let path = Path.GetFullPath path
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
    let temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp"
    try
        File.WriteAllText(temporary, contents)
        File.Move(temporary, path, true)
    finally
        if File.Exists temporary then File.Delete temporary

let private run readyPath command arguments = task {
    let errorPath = readyPath + ".error"
    try
        let readyPath = Path.GetFullPath readyPath
        Directory.CreateDirectory(Path.GetDirectoryName readyPath) |> ignore
        // Each invocation owns these paths. Leave the current evidence in place
        // for the runner to read and remove after its process/group cleanup.
        File.Delete readyPath
        File.Delete errorPath
        if OperatingSystem.IsLinux() then
            let session = Linux.createSession()
            if session < 0 then
                let detail = Win32Exception(Marshal.GetLastPInvokeError()).Message
                failwith ("setsid failed: " + detail)
            if session <> Environment.ProcessId then
                failwith "setsid returned an unexpected session identity"

        // On Linux this identifies a session/process group created before any
        // child is launched. Elsewhere it identifies only this host process.
        writeAtomic readyPath (string Environment.ProcessId + "\n")
        // Cwd and all three standard handles stay inherited. The outer runner
        // owns drainage and deadlines; this process introduces no extra pipes.
        let start = ProcessStartInfo(command, UseShellExecute = false)
        for argument in arguments do start.ArgumentList.Add argument
        use child = new Process(StartInfo = start)
        if not (child.Start()) then failwith ("could not start " + command)
        do! child.WaitForExitAsync()
        return child.ExitCode
    with ex ->
        try writeAtomic errorPath (ex.ToString() + "\n")
        with recordError -> eprintfn "ProcessHost: could not write failure sidecar: %s" recordError.Message
        return error 1 ex.Message
}

[<EntryPoint>]
let main argv =
    match Array.toList argv with
    | "--ready" :: readyPath :: "--" :: command :: arguments
        when not (String.IsNullOrWhiteSpace readyPath) && not (String.IsNullOrWhiteSpace command) ->
        (run readyPath command arguments).GetAwaiter().GetResult()
    | _ ->
        error 2 "Usage: ProcessHost --ready <path> -- <command> <args...>"
