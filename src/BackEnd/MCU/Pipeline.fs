module BackEnd.MCU.Pipeline

open System.IO
open Core.Types.Pipeline
open Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig

/// Two sibling MCU image paths. Which one runs is decided by the target the
/// orchestrator resolved from the declared architecture, not by a flag: a
/// Cortex-M image executes from its declared flash behind an address-table vector,
/// an Xtensa image is ROM-loaded into SRAM behind a vector block of code.
let backend: BackEnd = {
    Name = "LLVM / MCU image (Cortex-M or Xtensa)"
    Compile = fun witnessed ctx ->
        let mlirText = witnessed.Text
        try
            let directory = ctx.IntermediatesDir |> Option.defaultWith (fun () ->
                let path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath ctx.OutputPath), "intermediates")
                Directory.CreateDirectory path |> ignore
                path)
            let mlir = Path.Combine(directory, artifactFilename ArtifactId.Mlir)
            let llvm = Path.Combine(directory, artifactFilename ArtifactId.Llvm)
            File.WriteAllText(mlir, mlirText)
            // The triple comes from the selected platform. An absent triple
            // must not silently become the Cortex-M profile.
            let triple =
                match ctx.EmbeddedTarget, ctx.XtensaTarget, ctx.TargetTripleOverride with
                | Some _, Some _, _ -> failwith "Exactly one MCU image target may be resolved"
                | _, _, Some declared -> declared
                | _, _, None -> failwith "The MCU image target requires a declared triple"
            BackEnd.LLVM.Lowering.lowerToLLVM mlir llvm triple ctx.TargetPointerBits
            |> Result.map (fun () ->
                if ctx.EmitIntermediateOnly then IntermediateOnly "LLVM IR"
                else
                    if ctx.DeploymentMode <> Core.Types.Dialects.DeploymentMode.Embedded then failwith "MCU image requires output_kind = embedded"
                    match ctx.EmbeddedTarget, ctx.XtensaTarget with
                    | Some target, None ->
                        let elf = Image.build llvm ctx target
                        if ctx.Deploy then Probe.deploy target elf
                        NativeBinary elf
                    | None, Some target ->
                        let elf = XtensaImage.build llvm ctx target
                        // No probe path exists for this target yet; download is
                        // an explicit follow-on rather than a silent no-op.
                        if ctx.Deploy then failwith "Xtensa image download is not implemented; flash the .bin with an external loader"
                        NativeBinary elf
                    | None, None -> failwith "Missing resolved MCU image declaration"
                    | Some _, Some _ -> failwith "Exactly one MCU image target may be resolved")
        with ex -> Error ex.Message
}
