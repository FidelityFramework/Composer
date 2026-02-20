/// ArtifactVerification — Closed-loop verification that generated artifacts are consistent
///
/// After producing HDL (.sv) and constraints (.xdc), verify that:
///   1. Every constrained port in the XDC exists as a port in the HDL module
///   2. Every I/O port in the HDL top-level module has a constraint in the XDC
///
/// Mismatches surface as Result.Error diagnostics BEFORE the user runs
/// synthesis tooling (Vivado, Yosys, etc.) where errors are harder to trace.
///
/// Uses XParsec char parsers for extraction — same combinator framework
/// that powers the rest of the compiler pipeline.
module BackEnd.ArtifactVerification

open System.IO
open XParsec
open XParsec.Parsers
open XParsec.CharParsers

// Type alias for readability
type private R = Reader<char, unit, ReadableString, ReadableStringSlice>

// ═══════════════════════════════════════════════════════════════════════════
// Parser primitives
// ═══════════════════════════════════════════════════════════════════════════

/// Verilog identifier: [a-zA-Z_][a-zA-Z0-9_]*
let private pIdent (reader: R) =
    many1Chars2
        (satisfyL (fun c -> System.Char.IsLetter c || c = '_') "identifier start")
        (satisfyL (fun c -> System.Char.IsLetterOrDigit c || c = '_') "identifier char")
        reader

/// Parse a port direction keyword (input | output | inout)
let private pDirection (reader: R) =
    let pos = reader.Position
    match pIdent reader with
    | Ok { Parsed = "input" }  -> preturn "input" reader
    | Ok { Parsed = "output" } -> preturn "output" reader
    | Ok { Parsed = "inout" }  -> preturn "inout" reader
    | _ ->
        reader.Position <- pos
        fail (Message "expected port direction") reader

/// Skip a bit-width specifier like [7:0] or [63:0]
let private skipWidth (reader: R) =
    match reader.Peek() with
    | ValueSome '[' ->
        let mutable depth = 0
        let mutable cont = true
        while cont && not reader.AtEnd do
            match reader.Peek() with
            | ValueSome '[' -> depth <- depth + 1; reader.Skip()
            | ValueSome ']' -> depth <- depth - 1; reader.Skip(); if depth = 0 then cont <- false
            | ValueSome _ -> reader.Skip()
            | ValueNone -> cont <- false
        preturn () reader
    | _ -> preturn () reader

/// Skip to end of current line (past comma, comments), consuming the newline
let private skipToNewline (reader: R) =
    let mutable cont = true
    while cont && not reader.AtEnd do
        match reader.Peek() with
        | ValueSome '\n' -> reader.Skip(); cont <- false
        | ValueSome _ -> reader.Skip()
        | ValueNone -> cont <- false

// ═══════════════════════════════════════════════════════════════════════════
// SV module name extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Find all "module <Name>(" declarations in SV text; the last is the top-level.
/// CIRCT emits modules bottom-up: leaf modules first, top-level last.
let findTopModuleName (svText: string) : Result<string, string> =
    let reader = Reader.ofString svText ()
    let mutable lastModule = None

    while not reader.AtEnd do
        let pos = reader.Position
        // Try to match "module " at current position
        match pstring "module " reader with
        | Ok _ ->
            let _ = spaces reader
            match pIdent reader with
            | Ok { Parsed = name } ->
                // Verify it's followed by "(" (port list)
                let _ = spaces reader
                match reader.Peek() with
                | ValueSome '(' -> lastModule <- Some name
                | _ -> ()
            | Error _ -> ()
        | Error _ ->
            reader.Position <- pos
        // Advance past remainder of current line
        skipToNewline reader

    match lastModule with
    | Some name -> Ok name
    | None -> Error "No module declaration found in SystemVerilog output"

// ═══════════════════════════════════════════════════════════════════════════
// SV port extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Extract port names from a CIRCT-generated SystemVerilog module declaration.
/// Format:
///   module Name(
///     input  portA,   // comment
///            portB,   // comment
///     output portC    // comment
///   );
let extractSVModulePorts (svText: string) (moduleName: string) : Result<Set<string>, string> =
    let modulePrefix = sprintf "module %s(" moduleName
    match svText.IndexOf(modulePrefix) with
    | -1 -> Error (sprintf "Module '%s' not found in SystemVerilog output" moduleName)
    | startIdx ->
        let afterOpen = startIdx + modulePrefix.Length
        match svText.IndexOf(");", afterOpen) with
        | -1 -> Error (sprintf "Module '%s' port list not terminated with );" moduleName)
        | closeIdx ->
            let portListText = svText.Substring(afterOpen, closeIdx - afterOpen)
            let reader = Reader.ofString portListText ()
            let ports = ResizeArray<string>()

            while not reader.AtEnd do
                let _ = spaces reader
                if not reader.AtEnd then
                    // Try direction keyword
                    let pos = reader.Position
                    match pDirection reader with
                    | Ok _ -> let _ = spaces reader in ()
                    | Error _ -> reader.Position <- pos

                    // Skip optional width specifier
                    let _ = skipWidth reader
                    let _ = spaces reader

                    // Extract port name
                    if not reader.AtEnd then
                        let pos2 = reader.Position
                        match pIdent reader with
                        | Ok { Parsed = name } -> ports.Add(name)
                        | Error _ ->
                            reader.Position <- pos2
                            if not reader.AtEnd then reader.Skip()

                    // Skip remainder of line
                    skipToNewline reader

            if ports.Count = 0 then
                Error (sprintf "No ports found in module '%s' declaration" moduleName)
            else
                Ok (Set.ofSeq ports)

// ═══════════════════════════════════════════════════════════════════════════
// XDC port extraction
// ═══════════════════════════════════════════════════════════════════════════

/// Parse "get_ports {identifier}" and return the port name
let private pGetPorts (reader: R) =
    match pstring "get_ports" reader with
    | Ok _ ->
        let _ = spaces reader
        match pchar '{' reader with
        | Ok _ ->
            match pIdent reader with
            | Ok { Parsed = name } ->
                match pchar '}' reader with
                | Ok _ -> preturn name reader
                | Error e -> Error e
            | Error e -> Error e
        | Error e -> Error e
    | Error e -> Error e

/// Extract all port names from constraint file via [get_ports {name}]
let extractConstraintPorts (constraintText: string) : Set<string> =
    let reader = Reader.ofString constraintText ()
    let ports = ResizeArray<string>()

    while not reader.AtEnd do
        let pos = reader.Position
        match pGetPorts reader with
        | Ok { Parsed = name } ->
            if not (ports.Contains(name)) then
                ports.Add(name)
        | Error _ ->
            reader.Position <- pos
            if not reader.AtEnd then reader.Skip()

    Set.ofSeq ports

// ═══════════════════════════════════════════════════════════════════════════
// Closed-loop verification
// ═══════════════════════════════════════════════════════════════════════════

/// Ports expected in HDL but not in constraints (e.g., compiler-generated reset)
let private internalPorts = Set.ofList [ "rst" ]

/// Verify that HDL port names and constraint port names are consistent.
/// Finds the top-level module automatically (last module in CIRCT output).
/// Returns Ok with summary on success, Error with diagnostic on mismatch.
let verifyArtifacts (svPath: string) (xdcPath: string) : Result<string, string> =
    let svText = File.ReadAllText(svPath)
    let xdcText = File.ReadAllText(xdcPath)

    findTopModuleName svText
    |> Result.bind (fun topModule ->
        extractSVModulePorts svText topModule
        |> Result.bind (fun svPorts ->
            let xdcPorts = extractConstraintPorts xdcText

            // Ports in XDC but not in SV — constraint references a non-existent port
            let xdcOnly = Set.difference xdcPorts svPorts

            // Ports in SV but not in XDC, excluding known internal ports
            let svOnly = Set.difference (Set.difference svPorts xdcPorts) internalPorts

            if Set.isEmpty xdcOnly && Set.isEmpty svOnly then
                Ok (sprintf "Verified: %d HDL ports match %d constraints (module %s)"
                        (Set.count svPorts) (Set.count xdcPorts) topModule)
            else
                let sb = System.Text.StringBuilder()
                sb.AppendLine("Artifact verification failed: HDL/constraint port mismatch") |> ignore
                sb.AppendLine(sprintf "  HDL module: %s (%s)" topModule svPath) |> ignore
                sb.AppendLine(sprintf "  Constraints: %s" xdcPath) |> ignore

                if not (Set.isEmpty xdcOnly) then
                    sb.AppendLine() |> ignore
                    sb.AppendLine("  Ports constrained but missing from HDL module:") |> ignore
                    for p in xdcOnly do
                        sb.AppendLine(sprintf "    - %s" p) |> ignore
                    sb.AppendLine("  (Constraint will be ignored by synthesis — coeffect bug)") |> ignore

                if not (Set.isEmpty svOnly) then
                    sb.AppendLine() |> ignore
                    sb.AppendLine("  HDL ports without constraints (will be left unconnected):") |> ignore
                    for p in svOnly do
                        sb.AppendLine(sprintf "    - %s" p) |> ignore
                    sb.AppendLine("  (Add [<Pin>] attributes or check PlatformPinResolution)") |> ignore

                Error (sb.ToString().TrimEnd())))
