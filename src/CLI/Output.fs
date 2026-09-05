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
/// The interim warnings `--warnaserror` does not promote (Dimensional_Range_Design.md, CS-12
/// ruling 5, step 5a): CCS8019, the width-spelling alias, while the corpus still carries the
/// spellings and the suffixes. This set is the promotion switch: step three of the ruling deletes
/// it with the alias, and a spelling is CCS8706, a suffix CCS8018, errors in their own right.
let private interimWarnings : Set<string> = set [ "CCS8019" ]

/// Whether `--warnaserror` promotes this warning: every warning but the interim ones.
let private promoted (warnaserror: bool) (diag: Diagnostic) : bool =
    warnaserror && not (Set.contains diag.Code interimWarnings)

let emitDiagnostic (warnaserror: bool) (projectDir: string option) (diag: Diagnostic) =
    let effectiveSev = Diagnostic.effectiveSeverity diag

    // When warnaserror is set, elevate reachable warnings to errors in display
    let displaySev =
        if promoted warnaserror diag && effectiveSev = NativeDiagnosticSeverity.Warning then
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

    // Return counts reflecting elevation; an interim warning stays a warning under --warnaserror
    if warnaserror then
        let elevated, interim = warnings |> List.partition (promoted warnaserror)
        (List.length errors + List.length elevated, List.length interim, List.length infos)
    else
        (List.length errors, List.length warnings, List.length infos)

/// Emit the parse errors of a project: one error line per message, in the diagnostic line format,
/// so a file that failed to parse is reported and counted like any other error. The parser's messages
/// carry their own position text; they carry no CCS code yet, because the lexer and parser family
/// takes its CCS codes with the step-4 mapping table (Dimensional_Vetting_Plan.md D3), and no code is
/// minted here in the meantime. Returns the number of error lines emitted.
let emitParseErrors (projectDir: string option) (parseErrors: Map<string, string list>) : int =
    parseErrors
    |> Map.toList
    |> List.sumBy (fun (file, messages) ->
        // The message is already located ("file(line,col): text", NativeService.parseString);
        // print it with the project-relative path so the line reads like every other diagnostic.
        let path = relativizePath projectDir file
        messages
        |> List.iter (fun message ->
            // The message reads "CCS0NNN: file(line,col): text"; print it as "error CCS0NNN: ..."
            eprintfn "%s %s" (c errorColor "error") (message.Replace(file, path)))
        List.length messages)

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
