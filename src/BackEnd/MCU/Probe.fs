/// Direct managed orchestration of the installed vendor SDK. No Python bridge,
/// C shim, generated command script, option-memory payload, or firmware update.
module BackEnd.MCU.Probe

open System
open System.IO
open System.Runtime.InteropServices
open System.Text
open System.Text.Json
open System.Threading
open Core.Types.Pipeline

[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type private Call0 = delegate of unit -> int
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type private Call1 = delegate of int -> int
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type private OpenCall = delegate of unit -> nativeint
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type private CommandCall = delegate of nativeint * nativeint * int -> int
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type private ReadCall = delegate of uint32 * uint32 * nativeint * uint32 -> int
[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type private DownloadCall = delegate of nativeint * uint32 -> int

let private export<'T when 'T :> Delegate> library name = Marshal.GetDelegateForFunctionPointer<'T>(NativeLibrary.GetExport(library, name))

type private Connection(target: EmbeddedTarget, programming: bool) =
    let library =
        Target.requireProbeSupport target.Profile
        NativeLibrary.Load(Tools.probeLibrary target.ProbeLibrary)
    let mutable opened = false
    let mutable disposed = false
    let call0 name = (export<Call0> library name).Invoke()
    let utf8 text action =
        let ptr = Marshal.StringToCoTaskMemUTF8 text
        try action ptr finally Marshal.FreeCoTaskMem ptr
    let command text =
        let output = Marshal.AllocHGlobal 1024
        try
            Marshal.WriteByte(output, 0uy)
            let code = utf8 text (fun input -> (export<CommandCall> library "JLINKARM_ExecCommand").Invoke(input, output, 1024))
            if code < 0 then failwithf "J-Link %s: %s" text (Marshal.PtrToStringUTF8 output)
        finally Marshal.FreeHGlobal output
    let read (address: uint32) (count: int) (width: int) =
        let buffer = Marshal.AllocHGlobal count
        try
            let got = (export<ReadCall> library "JLINKARM_ReadMemEx").Invoke(address, uint32 count, buffer, uint32 width)
            if got <> count then failwithf "J-Link short read at 0x%X: %d/%d" address got count
            let bytes = Array.zeroCreate<byte> count
            Marshal.Copy(buffer, bytes, 0, count)
            bytes
        finally Marshal.FreeHGlobal buffer
    let close () =
        if not disposed then
            if opened then call0 "JLINKARM_Close" |> ignore
            NativeLibrary.Free library
            disposed <- true
    do
        try
            command "SuppressGUI 1"
            command "DisableAutoUpdateFW"
            let error = (export<OpenCall> library "JLINKARM_Open").Invoke()
            if error <> 0n then failwith (Marshal.PtrToStringUTF8 error)
            opened <- true
            command ("device = " + if programming then target.Image.DebugDevice else "Cortex-M33")
            if (export<Call1> library "JLINKARM_TIF_Select").Invoke(1) < 0 then failwith "J-Link SWD unavailable"
            (export<Call1> library "JLINKARM_SetSpeed").Invoke(4000) |> ignore
            if call0 "JLINKARM_Connect" < 0 then failwith "J-Link SWD connection failed"
            let part = Encoding.ASCII.GetString(read (uint32 target.Image.PartNumberAddress) 16 4).Trim()
            if part <> target.Image.PartNumber then failwith ("Unexpected silicon part: " + part)
        with _ -> close (); reraise ()
    member _.Read(address, count, width) = read address count width
    member _.Options() = read (uint32 target.Image.PreservedOptionAddress) target.Image.PreservedOptionBytes 4
    member _.Snapshot() =
        {| utc = DateTimeOffset.UtcNow; part = target.Image.PartNumber; probe = uint32 (call0 "JLINKARM_GetSN")
           halted = call0 "JLINKARM_IsHalted" <> 0
           cpuid = BitConverter.ToUInt32(read 0xe000ed00u 4 4)
           vtor = BitConverter.ToUInt32(read 0xe000ed08u 4 4)
           preservedOptions = Convert.ToHexStringLower(read (uint32 target.Image.PreservedOptionAddress) target.Image.PreservedOptionBytes 4) |}
    member _.Reset() =
        if call0 "JLINKARM_Reset" < 0 then failwith "J-Link reset failed"
        call0 "JLINKARM_Go" |> ignore
    member _.Download(path) =
        if call0 "JLINKARM_Halt" < 0 then failwith "Cannot halt before code download"
        let result = utf8 path (fun ptr -> (export<DownloadCall> library "JLINK_DownloadFile").Invoke(ptr, uint32 (Layout.origin target.Flash)))
        if result < 0 then failwith "J-Link code-flash download failed"
    interface IDisposable with member _.Dispose() = close ()

let private readImage (probe: Connection) address size =
    [| for offset in 0 .. 65536 .. size - 1 do
           yield! probe.Read(address + uint32 offset, min 65536 (size - offset), 1) |]

let private validateRecovery (target: EmbeddedTarget) =
    use doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(target.RecoveryDirectory, "manifest.json")))
    let root = doc.RootElement
    if root.GetProperty("part").GetString() <> target.Image.PartNumber then failwith "Recovery image belongs to another part"
    let regions = root.GetProperty("regions").EnumerateArray() |> Seq.toArray
    for region in regions do
        let name = region.GetProperty("file").GetString()
        if Path.GetFileName name <> name then failwith "Invalid recovery filename"
        let bytes = File.ReadAllBytes(Path.Combine(target.RecoveryDirectory, name))
        if bytes.Length <> region.GetProperty("bytes").GetInt32() || Tools.sha256 bytes <> region.GetProperty("sha256").GetString() then
            failwith ("Invalid recovery hash/extent: " + name)
    if not (regions |> Array.exists (fun r -> r.GetProperty("file").GetString() = "code-flash.bin" && r.GetProperty("base").GetInt64() = Layout.origin target.Flash && r.GetProperty("bytes").GetInt64() = target.Flash.Capacity)) then
        failwith "Recovery manifest must contain the complete original code flash"

let private currentImage (target: EmbeddedTarget) elfPath =
    use doc = JsonDocument.Parse(File.ReadAllText(Path.ChangeExtension(elfPath, "build-evidence.json")))
    let e = doc.RootElement
    if e.GetProperty("orchestrator").GetString() <> "Composer" || e.GetProperty("platform").GetString() <> target.PlatformId then failwith "Build with Composer before using this artifact"
    let elf, binary = File.ReadAllBytes elfPath, File.ReadAllBytes(Path.ChangeExtension(elfPath, "bin"))
    if Tools.sha256 elf <> e.GetProperty("sha256").GetString() || Tools.sha256 binary <> e.GetProperty("binarySha256").GetString() || binary.Length <> e.GetProperty("binaryBytes").GetInt32() then
        failwith "Image changed since Composer verified it; rebuild before deployment"
    let table = File.ReadAllText(Path.ChangeExtension(elfPath, "symbols.txt")) |> Image.symbols
    Image.verify target elf binary table
    binary, table

let deploy (target: EmbeddedTarget) elfPath =
    Target.requireProbeSupport target.Profile
    validateRecovery target
    let binary, _ = currentImage target elfPath
    use probe = new Connection(target, true)
    let before = probe.Options()
    probe.Download(Path.ChangeExtension(elfPath, "bin"))
    if readImage probe (uint32 (Layout.origin target.Flash)) binary.Length <> binary then failwith "Code-flash readback differs; target left halted"
    if probe.Options() <> before then failwith "Preserved options changed; target left halted"
    probe.Reset()
    Tools.writeJson (Path.ChangeExtension(elfPath, "deployment.json"))
        {| binarySha256 = Tools.sha256 binary; readbackVerified = true; optionsUnchanged = true; board = probe.Snapshot() |}
    printfn "Composer deployed and readback-verified %d code bytes; reset and running" binary.Length

let device action seconds (target: EmbeddedTarget) elfPath =
    Target.requireProbeSupport target.Profile
    if not (List.contains action ["inspect";"watch";"capture";"reset";"restore"]) then failwith "Device action must be inspect, watch, capture, reset or restore"
    if action = "restore" then validateRecovery target
    let current = if action = "watch" then Some (currentImage target elfPath) else None
    use probe = new Connection(target, (action = "restore"))
    match action with
    | "capture" ->
        Directory.CreateDirectory target.RecoveryDirectory |> ignore
        let regions = [| "code-flash", Layout.origin target.Flash, int target.Flash.Capacity
                         "watchdog-options", target.Image.PreservedOptionAddress, target.Image.PreservedOptionBytes |]
        let captured = regions |> Array.map (fun (name, address, size) ->
            let bytes = readImage probe (uint32 address) size
            if readImage probe (uint32 address) size <> bytes then failwith ("Unstable recovery read: " + name)
            let path = Path.Combine(target.RecoveryDirectory, name + ".bin")
            if File.Exists path && File.ReadAllBytes path <> bytes then failwith "Refusing to overwrite an earlier recovery image; select a different embedded.recovery directory"
            path, bytes, {| file = name + ".bin"; ``base`` = address; bytes = size; sha256 = Tools.sha256 bytes |})
        for path, bytes, _ in captured do File.WriteAllBytes(path, bytes)
        Tools.writeJson (Path.Combine(target.RecoveryDirectory,"manifest.json"))
            {| part = target.Image.PartNumber; capturedUtc = DateTimeOffset.UtcNow; regions = captured |> Array.map (fun (_,_,row) -> row)
               dataFlash = "Not modified or backed up; erased reads are undefined (RA6M5 HW 50.16.2)." |}
        printfn "Original code flash and option snapshot captured and double-read-verified"
    | "restore" ->
        let path = Path.Combine(target.RecoveryDirectory,"code-flash.bin")
        let bytes, options = File.ReadAllBytes path, probe.Options()
        probe.Download path
        if readImage probe (uint32 (Layout.origin target.Flash)) bytes.Length <> bytes || probe.Options() <> options then failwith "Restore verification failed; target left halted"
        probe.Reset()
        printfn "Original code restored and readback-verified; option memory unchanged"
    | "reset" -> probe.Reset()
    | "watch" ->
        let binary, table = current.Value
        if readImage probe (uint32 (Layout.origin target.Flash)) binary.Length <> binary then failwith "Board differs from the current verified image"
        for _ in 1 .. max 1 (min 60 seconds) do
            let watches = target.WatchSymbols |> Map.toArray |> Array.map (fun (name, count) ->
                let address = Map.tryFind name table |> Option.defaultWith (fun () -> failwith ("Missing watch symbol: " + name))
                if int64 address < Layout.origin target.Ram || int64 address + int64 (4 * count) > Layout.origin target.Ram + target.Ram.Capacity then failwith "Watch must stay within declared SRAM"
                let bytes = probe.Read(address, count * 4, 4)
                name, [| for i in 0 .. count - 1 -> BitConverter.ToUInt32(bytes, 4*i) |]) |> dict
            printfn "%s" (JsonSerializer.Serialize {| board = probe.Snapshot(); watches = watches |})
            Thread.Sleep 1000
    | _ -> ()
    if action <> "watch" then printfn "%s" (JsonSerializer.Serialize(probe.Snapshot()))
