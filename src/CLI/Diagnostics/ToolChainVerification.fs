module CLI.Diagnostics.ToolchainVerification

open System
open System.IO
open System.Runtime.InteropServices
open CLI.Diagnostics.Types

/// Represents the status of a toolchain element
type ComponentStatus =
    | Found of version: string
    | Missing of hint: string
    | Error of message: string

/// Represents a required toolchain element
type ToolchainComponent = {
    Name: string
    Description: string
    CheckCommand: string option
    CheckFiles: string list
    InstallHint: string
    Required: bool
}

/// Enhanced MSYS2 environment detection
module MSYS2Detection =
    
    /// Gets detailed MSYS2 environment information
    let getMSYS2EnvironmentInfo() : (string * string * string list) option =
        try
            let msystem = Environment.GetEnvironmentVariable("MSYSTEM")
            let msysRoot = Environment.GetEnvironmentVariable("MSYSTEM_PREFIX")
            let mingwPrefix = Environment.GetEnvironmentVariable("MINGW_PREFIX")
            let path = Environment.GetEnvironmentVariable("PATH")
            
            match msystem with
            | null | "" -> None
            | env ->
                let pathDirectories = 
                    if String.IsNullOrEmpty(path) then []
                    else path.Split(';') |> Array.toList |> List.filter (not << String.IsNullOrWhiteSpace)
                
                let rootPath = 
                    if not (String.IsNullOrEmpty(msysRoot)) then msysRoot
                    elif not (String.IsNullOrEmpty(mingwPrefix)) then mingwPrefix
                    else "/mingw64"
                
                Some (env, rootPath, pathDirectories)
        with _ -> None
    
    /// Validates MSYS2 environment for compilation
    let validateMSYS2Environment() : ComponentStatus =
        match getMSYS2EnvironmentInfo() with
        | Some ("MSYS", _, _) ->
            Missing "MSYS environment detected - use MINGW64 for native compilation"
        | Some ("MINGW64", rootPath, _) ->
            Found (sprintf "MINGW64 environment at %s" rootPath)
        | Some ("MINGW32", rootPath, _) ->
            Found (sprintf "MINGW32 environment at %s (consider MINGW64 for 64-bit targets)" rootPath)
        | Some (env, rootPath, _) ->
            Found (sprintf "%s environment at %s" env rootPath)
        | None ->
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                Missing "Not in MSYS2 environment - install and use MSYS2 MINGW64"
            else
                Found "Non-Windows environment"

/// Enhanced command availability checking with MSYS2 support
module CommandDetection =
    
    /// Comprehensive command search with multiple execution strategies
    let checkCommandAvailability (command: string) (args: string) : ComponentStatus =
        let commandVariants = 
            if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
                [command; command + ".exe"]
            else
                [command]
        
        let tryExecuteCommand (cmd: string) (arguments: string) : bool * string =
            try
                let processInfo = System.Diagnostics.ProcessStartInfo()
                processInfo.FileName <- cmd
                processInfo.Arguments <- arguments
                processInfo.UseShellExecute <- false
                processInfo.RedirectStandardOutput <- true
                processInfo.RedirectStandardError <- true
                processInfo.CreateNoWindow <- true
                processInfo.WindowStyle <- System.Diagnostics.ProcessWindowStyle.Hidden
                
                use proc = System.Diagnostics.Process.Start(processInfo)
                let output = proc.StandardOutput.ReadToEnd()
                let error = proc.StandardError.ReadToEnd()
                proc.WaitForExit(5000) |> ignore
                
                let combinedOutput = 
                    [output; error] 
                    |> List.filter (not << String.IsNullOrWhiteSpace)
                    |> String.concat " "
                
                (proc.ExitCode = 0, combinedOutput)
            with
            | ex -> (false, ex.Message)
        
        let tryFindInPath (cmd: string) : bool =
            try
                let pathDirs = 
                    Environment.GetEnvironmentVariable("PATH").Split(';')
                    |> Array.filter (not << String.IsNullOrWhiteSpace)
                
                pathDirs |> Array.exists (fun dir ->
                    let fullPath = Path.Combine(dir, cmd)
                    File.Exists(fullPath))
            with _ -> false
        
        let rec tryCommands variants =
            match variants with
            | [] -> Error "Command not found"
            | cmd :: rest ->
                let (success, output) = tryExecuteCommand cmd args
                if success then
                    let version = 
                        output.Split('\n')
                        |> Array.tryHead
                        |> Option.defaultValue output
                        |> fun s -> s.Trim()
                    Found version
                else
                    if tryFindInPath cmd then
                        Error (sprintf "Command found but failed to execute: %s" output)
                    else
                        tryCommands rest
        
        tryCommands commandVariants
    
    /// Specialized LLVM tool detection
    let checkLLVMTools() : (string * ComponentStatus) list =
        let llvmTools = [
            ("ld.lld", "LLVM ELF linker and LTO code generation")
            ("opt", "LLVM optimizer")
            ("llvm-config", "LLVM configuration tool")
        ]
        
        llvmTools |> List.map (fun (tool, description) ->
            let status = checkCommandAvailability tool "--version"
            (sprintf "%s (%s)" tool description, status))
    
/// Platform-specific toolchain requirements with enhanced detection
module PlatformRequirements =
    
    /// These are the executable tools used by the ELF backend on any host.
    /// Hosted runtime files are resolved separately from the selected target.
    let private elfTools () : ToolchainComponent list =
        [
            { Name = "LLVM bitcode preparation"; Description = "Target-aware LLVM IR verification and bitcode emission"
              CheckCommand = Some "opt --version"; CheckFiles = []
              InstallHint = "Install LLVM tools (llvm package)"; Required = true }
            { Name = "LLD ELF linker"; Description = "LLVM code generation, linking and ELF layout"
              CheckCommand = Some "ld.lld --version"; CheckFiles = []
              InstallHint = "Install LLD (lld package) matching the LLVM tool version"; Required = true }
        ]

    let getWindowsToolchainRequirements() = elfTools ()
    let getLinuxToolchainRequirements() = elfTools ()
    let getMacOSToolchainRequirements() = elfTools ()

/// element checking with enhanced error reporting
module ComponentChecking =
    
    /// Checks if required files exist with detailed reporting
    let checkFiles (files: string list) : ComponentStatus =
        if files.IsEmpty then
            Found "No files to check"
        else
            let missing = files |> List.filter (not << File.Exists)
            let existing = files |> List.filter File.Exists
            
            if missing.IsEmpty then
                Found (sprintf "All %d files present" files.Length)
            else
                let missingList = String.concat ", " missing
                let existingList = String.concat ", " existing
                Missing (sprintf "Missing files: %s (found: %s)" missingList existingList)
    
    /// Enhanced element checking with fallback strategies
    let checkComponent (element: ToolchainComponent) : ComponentStatus =
        match element.CheckCommand with
        | Some cmd ->
            let parts = cmd.Split(' ', 2)
            let executable = parts.[0]
            let args = if parts.Length > 1 then parts.[1] else ""
            CommandDetection.checkCommandAvailability executable args
        | None ->
            if element.CheckFiles.IsEmpty then
                Error "No check method specified"
            else
                checkFiles element.CheckFiles
    
    /// Special check for MSYS2 environment
    let checkMSYS2Environment() : ComponentStatus =
        MSYS2Detection.validateMSYS2Environment()

/// Main toolchain verification with comprehensive reporting
let verifyToolchain (verbose: bool) : DiagnosticResult<unit> =
    printfn "Verifying Composer toolchain requirements..."
    printfn "=========================================="
    
    // Enhanced platform detection and environment analysis
    let platform = 
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            match MSYS2Detection.getMSYS2EnvironmentInfo() with
            | Some (env, rootPath, pathDirs) -> 
                sprintf "Windows (MSYS2 %s at %s)" env rootPath
            | None -> 
                "Windows (native - consider using MSYS2)"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
            "Linux"
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
            "macOS"
        else
            "Unknown platform"
    
    printfn "Platform: %s" platform
    printfn "Runtime: %s" (RuntimeInformation.FrameworkDescription)
    printfn ""
    
    // Check MSYS2 environment first on Windows
    if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
        match ComponentChecking.checkMSYS2Environment() with
        | Found msg ->
            printfn "✓ MSYS2 Environment: %s" msg
        | Missing hint ->
            printfn "⚠ MSYS2 Environment: %s" hint
            printfn "  For optimal compatibility, use MSYS2 MINGW64 terminal"
        | Error msg ->
            printfn "✗ MSYS2 Environment: %s" msg
        printfn ""
    
    // Get platform-specific requirements
    let requirements = 
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            PlatformRequirements.getWindowsToolchainRequirements()
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then
            PlatformRequirements.getLinuxToolchainRequirements()
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then
            PlatformRequirements.getMacOSToolchainRequirements()
        else
            []
    
    // Check each element with enhanced reporting
    let results = 
        requirements 
        |> List.map (fun element ->
            let status = ComponentChecking.checkComponent element
            (element, status))
    
    // Display results with detailed information
    let mutable hasErrors = false
    let mutable hasWarnings = false
    
    results |> List.iter (fun (element, status) ->
        let statusSymbol, statusText, isError, isWarning = 
            match status with
            | Found version -> 
                ("✓", version, false, false)
            | Missing hint -> 
                ("✗", sprintf "Missing - %s" hint, element.Required, not element.Required)
            | Error msg -> 
                ("!", sprintf "Error - %s" msg, element.Required, not element.Required)
        
        if isError then hasErrors <- true
        if isWarning then hasWarnings <- true
        
        printfn "%s %s: %s" statusSymbol element.Name statusText
        
        if verbose || isError || isWarning then
            printfn "  %s" element.Description
            
        if isError then
            printfn "  Installation: %s" element.InstallHint
            printfn ""
        elif isWarning then
            printfn "  Optional: %s" element.InstallHint
            printfn ""
    )
    
    // Additional LLVM and compiler analysis
    if verbose then
        printfn ""
        printfn "Additional Tool Analysis:"
        printfn "========================="
        
        let llvmTools = CommandDetection.checkLLVMTools()
        llvmTools |> List.iter (fun (name, status) ->
            match status with
            | Found version -> printfn "✓ %s: %s" name version
            | Missing hint -> printfn "✗ %s: %s" name hint
            | Error msg -> printfn "! %s: %s" name msg)
        
    
    printfn ""
    printfn "=========================================="
    
    if hasErrors then
        printfn "ERROR: Missing required components!"
        printfn "Please install the missing components before using Composer."
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            printfn ""
            printfn "Quick setup for Windows:"
            printfn "1. Install MSYS2 from https://www.msys2.org/"
            printfn "2. Open 'MSYS2 MINGW64' terminal"
            printfn "3. Run: pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-llvm"
            printfn "4. Run: composer doctor"
        Failure [InternalError("toolchain", "Missing required toolchain components", None)]
    elif hasWarnings then
        printfn "All required components found!"
        printfn "Some optional components are missing but compilation should work."
        Success ()
    else
        printfn "All components found!"
        Success ()

/// Check the tools Composer actually executes; target runtime inputs are checked at linking.
let quickVerifyToolchain() : bool =
    ["opt"; "ld.lld"] |> List.forall (fun tool ->
        match CommandDetection.checkCommandAvailability tool "--version" with
        | Found _ -> true
        | _ -> false)

let suggestToolchainFixes (error: string) : unit =
    if error.Contains("runtime") || error.Contains("library") || error.Contains("crt") then
        printfn "Check target startup objects, --sysroot, --link-library-path and --dynamic-linker."
    elif error.Contains("entry") then
        printfn "Check that the deployment's startup object defines _start and that its linker script is supplied."
    else
        printfn "Check matching LLVM opt and ld.lld installations; run composer doctor --verbose."
