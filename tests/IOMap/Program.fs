module IOMapTests

open System
open System.IO
open System.Text.RegularExpressions
open System.Security.Cryptography
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution
open Clef.Compiler.PSGSaturation.SemanticGraph.MappedBindings

// Test projections of the CCS-checked declarations; production Clef is never
// compiled by the host F# compiler.
type PackagePin = { Number: int; Port: string; System: string; Analog: string }
type BoardTerminal = { Component: string; Contact: string; Net: string }
type TraceLink = { Reference: string }

let boardDeclarations project =
    let checkedProject = FrontEnd.ProjectLoader.load project |> function Ok p -> p | Error e -> failwith e
    if Clef.Compiler.Project.ProjectChecker.hasErrors checkedProject then
        failwith (String.concat "\n" (Clef.Compiler.Project.ProjectChecker.getErrorMessages checkedProject))
    let graph = checkedProject.CheckResult.Graph
    let records prefix name kind =
        let binding = graph.Nodes.Values |> Seq.filter (fun b ->
            let qualified = qualifiedBindingName graph b
            qualified = prefix + name || qualified = name) |> Seq.exactlyOne
        let value = List.exactlyOne binding.Children
        match valueOf graph value with
        | Some { Kind = SemanticKind.ArrayExpr elements } ->
            elements |> List.map (fun id ->
                match recordOf graph id with
                | Some(node, fields) when typeName node = Some kind -> fields
                | _ -> failwith ("Invalid checked declaration " + name)) |> List.toArray
        | _ -> failwith ("Expected checked declaration array " + name)
    let text fields name = field name fields |> Option.bind (stringOf graph) |> Option.defaultWith (fun () -> failwith ("Missing string " + name))
    let number fields name = field name fields |> Option.bind (int64Of graph) |> Option.map int |> Option.defaultWith (fun () -> failwith ("Missing integer " + name))
    let silicon = "Fidelity.Platform.Hardware.Silicon.MCU.Renesas.RA6M5.R7FA6M5BH3CFC."
    let product = "Fidelity.Platform.Hardware.Products.Renesas.EK_RA6M5."
    let pins = records silicon "PackagePins.pins" "PackagePin" |> Array.map (fun f -> { Number = number f "Number"; Port = text f "Port"; System = text f "System"; Analog = text f "Analog" })
    let terminals = records product "BoardNets.terminals" "BoardTerminal" |> Array.map (fun f -> { Component = text f "Component"; Contact = text f "Contact"; Net = text f "Net" })
    let links = records product "BoardNets.traceLinks" "TraceLink" |> Array.map (fun f -> { Reference = text f "Reference" })
    pins, terminals, links

[<EntryPoint>]
let main args =
    let path = Path.GetFullPath args.[0]
    let project = if args.Length > 1 then Path.GetFullPath args.[1] else Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../../../MCU/Renesas/EK-RA6M5/HelloBlinky/HelloBlinky.fidproj"))
    let pins, declaredTerminals, traceLinks = boardDeclarations project
    let bytes = File.ReadAllBytes path
    let hash = SHA256.HashData bytes |> Convert.ToHexStringLower
    if hash <> "cb8bfea9f92a1faaa1ff5fe9420cdc8e6fd356bd39d907e5c85dcf1d077922aa" then failwith "Review the new design revision before changing the pin map"
    let lines = File.ReadAllLines path
    let mutable net = ""
    let terminals = ResizeArray<BoardTerminal>()
    let mutable matchedPackagePins = 0
    for i in 0 .. lines.Length - 1 do
        if lines.[i] = "NET_NAME" then net <- lines.[i+1].Trim('\'')
        let node = Regex.Match(lines.[i], @"^NODE_NAME\s+(\S+)\s+(\S+)")
        if node.Success then
            let part, pin = node.Groups.[1].Value, node.Groups.[2].Value
            terminals.Add { Component = part; Contact = pin; Net = net }
            if part = "U1" then
                let declared = pins |> Array.find (fun p -> p.Number = int pin)
                let label = Regex.Match(lines.[i+2], "'([^']+)'" ).Groups.[1].Value
                if declared.Port <> "" && not (label.Split('/') |> Array.contains declared.Port) then failwithf "Package pin %s: datasheet %s, netlist %s" pin declared.Port label
                // The 2021 symbol groups these distinct 2026 datasheet supply
                // names under VSS_USBHS. Confirm their actual grounded net too.
                let usbGroundAlias = label = "VSS_USBHS" && net = "GROUND_POWER" && List.contains declared.System ["PVSS_USBHS";"VSS1_USBHS";"VSS2_USBHS"]
                if declared.Port = "" && declared.System <> "" && not usbGroundAlias && not (declared.System.Split('/') |> Array.exists (fun name -> label.Split('/') |> Array.contains name)) then
                    failwithf "System pin %s: datasheet %s, netlist %s" pin declared.System label
                matchedPackagePins <- matchedPackagePins + 1
    let expected, actual = Set.ofArray declaredTerminals, Set.ofSeq terminals
    if expected.Count <> declaredTerminals.Length || expected <> actual then failwith "Board terminal map differs from the pinned netlist"
    if matchedPackagePins <> 176 || (pins |> Array.map (fun p -> p.Number) |> Array.sort) <> [|1..176|] then failwith "Incomplete LQFP176 map"
    let contacts = declaredTerminals |> Array.filter (fun t -> Regex.IsMatch(t.Component,"^J[0-9]+$"))
    for header in ["J1";"J2";"J3";"J4"] do
        if (contacts |> Array.filter (fun t -> t.Component = header) |> Array.map (fun t -> int t.Contact) |> Array.sort) <> [|1..40|] then failwith ("Incomplete header " + header)
    for link in traceLinks do
        if (declaredTerminals |> Array.filter (fun t -> t.Component = link.Reference) |> Array.length) <> 2 then failwith ("Invalid trace link " + link.Reference)
    let contact name pin = contacts |> Array.find (fun t -> t.Component = name && t.Contact = pin)
    if (contact "J26" "8").Net <> "P311" then failwith "Pmod1 reset must follow schematic, not the manual typo"
    if (pins |> Array.find (fun p -> p.Port = "P506")).Analog <> "AN122" then failwith "P506 is AN122"
    printfn "%d terminals match the pinned netlist; %d connector contacts, 160 native-header contacts, 176 package pins and 38 trace links checked" actual.Count contacts.Length
    0
