module BackEnd.MCU.Pipeline

open System.IO
open Core.Types.Pipeline
open Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig

let backend: BackEnd = {
    Name = "LLVM / Cortex-M image"
    Compile = fun mlirText ctx ->
        try
            let directory = ctx.IntermediatesDir |> Option.defaultWith (fun () ->
                let path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath ctx.OutputPath), "intermediates")
                Directory.CreateDirectory path |> ignore
                path)
            let mlir = Path.Combine(directory, artifactFilename ArtifactId.Mlir)
            let llvm = Path.Combine(directory, artifactFilename ArtifactId.Llvm)
            File.WriteAllText(mlir, mlirText)
            BackEnd.LLVM.Lowering.lowerToLLVM mlir llvm "thumbv8m.main-none-eabi" ctx.TargetPointerBits
            |> Result.map (fun () ->
                if ctx.EmitIntermediateOnly then IntermediateOnly "LLVM IR"
                else
                    let target = ctx.EmbeddedTarget |> Option.defaultWith (fun () -> failwith "Missing resolved MCU image declaration")
                    if ctx.DeploymentMode <> Core.Types.Dialects.DeploymentMode.Embedded then failwith "MCU image requires output_kind = embedded"
                    let elf = Image.build llvm ctx target
                    if ctx.Deploy then Probe.deploy target elf
                    NativeBinary elf)
        with ex -> Error ex.Message
}
