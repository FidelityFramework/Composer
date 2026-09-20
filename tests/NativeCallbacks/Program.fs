module NativeCallbackTests

open System
open System.IO
open System.Security.Cryptography
open System.Text.Json

let cases = [
    "OptionDefaultWith", "OptionDefaultWith.clef", "option-default-with", ["scf.if";"func.call_indirect"]
    "OptionDefaults", "OptionDefaults.clef", "option-defaults", ["scf.if";"func.call_indirect"]
    "OptionPartials", "OptionPartials.clef", "option-partials", ["scf.if";"func.call_indirect"]
    "OptionFunctionPayloads", "OptionFunctionPayloads.clef", "option-function-payloads", ["scf.if";"func.call_indirect"]
    "OptionCallbacks", "OptionCallbacks.clef", "option-callbacks", ["scf.if";"func.call_indirect"]
    "OptionEvaluation", "OptionEvaluation.clef", "option-evaluation", ["scf.if";"func.call_indirect"]
    "GenericRecords", "GenericRecords.clef", "generic-records", ["func.call_indirect"]
    "NativeCallbacks", "Main.clef", "native-callbacks", ["func.constant @NativeCallbacks.add";"func.constant @NativeCallbacks.subtract";"func.call_indirect"]
    "CapturedBuffers", "CapturedBuffers.clef", "captured-buffers", ["func.call_indirect";"memref.dim"]
    "CapturedRecords", "CapturedRecords.clef", "captured-records", ["func.call_indirect";"memref.extract_aligned_pointer_as_index"]
    "FunctionSnapshots", "FunctionSnapshots.clef", "function-snapshots", ["func.call_indirect"]
    "FunctionFields", "FunctionFields.clef", "function-fields", ["func.call_indirect"]
    "ListenerEntry", "ListenerEntry.clef", "listener-entry", ["func.constant @__clef_callback_";" : (index, i32) -> ()";"func.call @ListenerEntry.onDone";"func.call_indirect"]
    "IgnoreValues", "IgnoreValues.clef", "ignore-values", ["func.call @IgnoreValues.numeric";"func.call @IgnoreValues.optional";"func.call @IgnoreValues.consumeUnit"]
]

[<EntryPoint>]
let main args =
    let source = __SOURCE_DIRECTORY__
    let root = Path.GetFullPath(Path.Combine(source,"../.."))
    let compiler = if args.Length > 0 then Path.GetFullPath args.[0] else Path.Combine(root,"src/bin/Debug/net10.0/Composer")
    let requested = args |> Array.skip (min 1 args.Length) |> Array.toList
    let selected =
        if requested.IsEmpty then cases
        else
            for name in requested do
                if not (cases |> List.exists (fun (candidate,_,_,_) -> candidate = name)) then
                    failwithf "Unknown native callback case: %s" name
            cases |> List.filter (fun (name,_,_,_) -> List.contains name requested)
    let platform = Path.GetFullPath(Path.Combine(root,"../Fidelity.Platform/Environments/Linux/x86_64"))
    let work = Path.Combine(Path.GetTempPath(),"composer-callbacks-fsharp-" + Guid.NewGuid().ToString("N"))
    for name, file, executable, operations in selected do
        let directory = Path.Combine(work,name)
        Directory.CreateDirectory directory |> ignore
        File.Copy(Path.Combine(source,file), Path.Combine(directory,file))
        let mutable project = File.ReadAllText(Path.Combine(source,name + ".fidproj"))
        for binding in ["Fidelity.Platform.CompilerSurface.fidproj";"Fidelity.Pthread.fidproj"] do
            project <- project.Replace("\"../../../Fidelity.Platform/Environments/Linux/x86_64/" + binding + "\"", JsonSerializer.Serialize(Path.Combine(platform,binding)))
        let fidproj = Path.Combine(directory,name + ".fidproj")
        File.WriteAllText(fidproj,project)
        Tests.Process.requireSuccess compiler ["compile";fidproj;"-k";"--no-color"] 600000 (Some (Path.Combine(directory,"compile.log"))) |> ignore
        let mlirPath = Path.Combine(directory,"targets/intermediates/10_output.mlir")
        let mlir = File.ReadAllText mlirPath
        for operation in operations do
            if not (mlir.Contains operation) then failwith ("Missing compiler-owned callback operation: " + operation)
        Tests.Process.requireSuccess "mlir-opt" [mlirPath;"--verify-each";"-o";Path.Combine(directory,"verified.mlir")] 60000 (Some (Path.Combine(directory,"verify.log"))) |> ignore
        Tests.Process.requireSuccess (Path.Combine(directory,"targets",executable)) [] 10000 (Some (Path.Combine(directory,"run.log"))) |> ignore
        printfn "PASS %s" name
    let compilerFiles =
        [compiler; Path.Combine(Path.GetDirectoryName compiler,"Composer.dll"); Path.Combine(Path.GetDirectoryName compiler,"Clef.Compiler.Service.dll")]
        |> List.map (fun path ->
            use stream = File.OpenRead path
            {| path = path; sha256 = Convert.ToHexString(SHA256.HashData stream) |})
    File.WriteAllText(Path.Combine(work,"evidence.json"),JsonSerializer.Serialize({| runner = "F# / .NET"; compiler = compilerFiles; cases = selected |> List.map (fun (name,_,_,_) -> name); passed = selected.Length; mlirVerifier = "mlir-opt --verify-each" |},JsonSerializerOptions(WriteIndented = true)))
    printfn "%d fresh hosted executables passed. Evidence: %s" selected.Length work
    0
