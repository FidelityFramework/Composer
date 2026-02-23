/// CLI Output — Colored diagnostic formatting for terminal output.
///
/// Uses Thuja's Color/Style vocabulary so the palette carries forward
/// when Composer gains TUI mode. For now, bridges Thuja types to ANSI
/// SGR escape sequences for direct console output.
///
/// Design: structural markers (severity, code, location) are colorized;
/// message body stays in default terminal color for readability.
module CLI.Output

open System
open Thuja
open Thuja.Styles

// ═══════════════════════════════════════════════════════════════════════════
// ANSI Bridge — Thuja.Color → SGR escape sequences
// ═══════════════════════════════════════════════════════════════════════════

/// Color support: on by default, disabled by --no-color flag or NO_COLOR env var (https://no-color.org/)
let mutable private useColor =
    Environment.GetEnvironmentVariable("NO_COLOR") |> isNull

/// Disable colored output (called from CLI when --no-color is passed)
let disableColor () = useColor <- false

/// Convert a Thuja Color to its ANSI SGR foreground code
let private fg (color: Color) =
    match color with
    | Reset -> "\x1b[0m"
    | Ansi code -> sprintf "\x1b[38;5;%dm" code
    | Rgb (r, g, b) -> sprintf "\x1b[38;2;%d;%d;%dm" r g b

let private rst = "\x1b[0m"

/// Apply foreground color to a string segment
let private c (color: Color) (text: string) =
    if useColor then sprintf "%s%s%s" (fg color) text rst
    else text

/// Apply dim attribute
let private dim (text: string) =
    if useColor then sprintf "\x1b[2m%s%s" text rst
    else text

// ═══════════════════════════════════════════════════════════════════════════
// Palette — Thuja color definitions (shared with future TUI)
// ═══════════════════════════════════════════════════════════════════════════

let private errorColor = Color.Red
let private warningColor = Color.Yellow
let private infoColor = Color.Blue

// ═══════════════════════════════════════════════════════════════════════════
// Path Formatting
// ═══════════════════════════════════════════════════════════════════════════

/// Relativize a path against the project directory.
/// Project sources → relative path. Dependencies → last 3 segments.
let private relativizePath (projectDir: string option) (filePath: string) =
    match projectDir with
    | None -> filePath
    | Some dir ->
        let normDir = dir.TrimEnd('/').TrimEnd('\\').Replace('\\', '/')
        let normFile = filePath.Replace('\\', '/')
        if normFile.StartsWith(normDir, StringComparison.OrdinalIgnoreCase) then
            normFile.Substring(normDir.Length).TrimStart('/')
        else
            let parts = normFile.Split('/')
            if parts.Length >= 3 then
                String.Join("/", parts.[parts.Length - 3..])
            else
                normFile

// ═══════════════════════════════════════════════════════════════════════════
// Diagnostic Formatting
// ═══════════════════════════════════════════════════════════════════════════

open Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics

/// Format and emit a single diagnostic to stderr.
/// Colored: location, severity label, diagnostic code.
/// Plain: message body.
/// Dim: [unreachable] tag.
let emitDiagnostic (warnaserror: bool) (projectDir: string option) (diag: Diagnostic) =
    let effectiveSev = Diagnostic.effectiveSeverity diag

    // When warnaserror is set, elevate reachable warnings to errors in display
    let displaySev =
        if warnaserror && effectiveSev = NativeDiagnosticSeverity.Warning then
            NativeDiagnosticSeverity.Error
        else
            effectiveSev

    let sevLabel, sevColor =
        match displaySev with
        | NativeDiagnosticSeverity.Error   -> "error", errorColor
        | NativeDiagnosticSeverity.Warning -> "warning", warningColor
        | NativeDiagnosticSeverity.Info    -> "info", infoColor

    let path = relativizePath projectDir diag.Range.File
    let location = c sevColor (sprintf "%s:%d" path diag.Range.Start.Line)

    let reachTag =
        match diag.Reachability with
        | Unreachable -> " " + dim "[unreachable]"
        | _ -> ""

    eprintfn "%s: %s %s: %s%s" location (c sevColor sevLabel) (c sevColor diag.Code) diag.Message reachTag

/// Emit all diagnostics grouped by effective severity.
/// When warnaserror is set, reachable warnings are elevated to errors in both display and counts.
/// Returns (errors, warnings, infos) counts reflecting the elevation.
let emitAllDiagnostics (warnaserror: bool) (projectDir: string option) (diagnostics: Diagnostic list) =
    let errors = diagnostics |> List.filter (fun d -> Diagnostic.effectiveSeverity d = NativeDiagnosticSeverity.Error)
    let warnings = diagnostics |> List.filter (fun d -> Diagnostic.effectiveSeverity d = NativeDiagnosticSeverity.Warning)
    let infos = diagnostics |> List.filter (fun d -> Diagnostic.effectiveSeverity d = NativeDiagnosticSeverity.Info)

    // Emit in severity order: errors first, then warnings (elevated if warnaserror), then info
    for d in errors do emitDiagnostic warnaserror projectDir d
    for d in warnings do emitDiagnostic warnaserror projectDir d
    for d in infos do emitDiagnostic warnaserror projectDir d

    // Return counts reflecting elevation
    if warnaserror then
        (List.length errors + List.length warnings, 0, List.length infos)
    else
        (List.length errors, List.length warnings, List.length infos)

/// Emit a summary line after diagnostics.
let emitSummary (errors: int) (warnings: int) (infos: int) =
    if errors > 0 || warnings > 0 || infos > 0 then
        let plural n = if n > 1 then "s" else ""
        let parts =
            [ if errors > 0 then c errorColor (sprintf "%d error%s" errors (plural errors))
              if warnings > 0 then c warningColor (sprintf "%d warning%s" warnings (plural warnings))
              if infos > 0 then c infoColor (sprintf "%d info" infos) ]
        eprintfn ""
        eprintfn "  %s" (String.Join("  ", parts))
