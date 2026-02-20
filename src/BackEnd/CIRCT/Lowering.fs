/// CIRCT Lowering - Optimize and export hw/comb/seq MLIR
///
/// Alex elides directly to hw/comb/seq dialects for FPGA targets.
/// The backend only needs optimization + Verilog export:
///   1. circt-opt --canonicalize --cse       : Optimize hardware IR
///   2. circt-opt --lower-seq-to-sv --lower-hw-to-sv --export-verilog : Export
///
/// This is the FPGA equivalent of BackEnd/LLVM/Lowering.fs.
module BackEnd.CIRCT.Lowering

open System.IO

let internal circtOptPath =
    let envPath = System.Environment.GetEnvironmentVariable("CIRCT_OPT_PATH")
    if not (System.String.IsNullOrEmpty(envPath)) then envPath
    else "circt-opt"

/// Run an external tool, returning Ok () or Error with stderr
let internal runTool (tool: string) (args: string) : Result<unit, string> =
    try
        let proc = new System.Diagnostics.Process()
        proc.StartInfo.FileName <- tool
        proc.StartInfo.Arguments <- args
        proc.StartInfo.UseShellExecute <- false
        proc.StartInfo.RedirectStandardOutput <- true
        proc.StartInfo.RedirectStandardError <- true
        proc.Start() |> ignore
        let _stdout = proc.StandardOutput.ReadToEnd()
        let stderr = proc.StandardError.ReadToEnd()
        proc.WaitForExit()

        if proc.ExitCode <> 0 then
            Error (sprintf "%s failed (exit %d): %s" tool proc.ExitCode stderr)
        else
            Ok ()
    with
    | :? System.ComponentModel.Win32Exception ->
        Error (sprintf "%s not found. Ensure it is on PATH or set CIRCT_OPT_PATH." tool)
    | ex ->
        Error (sprintf "%s failed: %s" tool ex.Message)

/// Step 1: Optimize hw/comb/seq MLIR
/// Alex elides directly to hw/comb/seq — map residual arith ops to comb, then optimize.
let optimizeHW (mlirPath: string) (optimizedPath: string) : Result<unit, string> =
    let args = sprintf "%s --map-arith-to-comb --canonicalize --cse -o %s" mlirPath optimizedPath
    runTool circtOptPath args

/// Step 2: Export hw/comb/seq to SystemVerilog via SV dialect
let exportToVerilog (hwPath: string) (svPath: string) : Result<unit, string> =
    let args =
        sprintf "%s --lower-seq-to-sv --lower-hw-to-sv --export-verilog -o %s"
            hwPath svPath
    runTool circtOptPath args
