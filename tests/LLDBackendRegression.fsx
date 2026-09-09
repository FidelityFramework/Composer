// Build Composer, then dotnet fsi tests/LLDBackendRegression.fsx.
// Linux x86_64 execution tests plus an ARM ELF cross-link; no C compiler installed/used by the test.
#I "../src/bin/Debug/net10.0"
#r "Fidelity.Data.dll"
#r "Clef.Compiler.Service.dll"
#r "Composer.dll"
open System
open System.IO
open System.Diagnostics
open System.Runtime.InteropServices
open Core.Types.Dialects
open Core.Types.Pipeline
open BackEnd.LLVM.Codegen

[<UnmanagedFunctionPointer(CallingConvention.Cdecl)>]
type Answer = delegate of unit -> int

let directory = Path.Combine(Path.GetTempPath(), "composer lld " + Guid.NewGuid().ToString("N"))
Directory.CreateDirectory directory |> ignore
let bin = Path.Combine(directory, "blocked compilers")
Directory.CreateDirectory bin |> ignore
let trap = Path.Combine(directory, "forbidden-tool-called")
for tool in ["clang"; "clang++"; "gcc"; "cc"; "llc"] do
    let path = Path.Combine(bin, tool)
    File.WriteAllText(path, "#!/bin/sh\ntouch '" + trap + "'\nexit 97\n")
    File.SetUnixFileMode(path, UnixFileMode.UserRead ||| UnixFileMode.UserWrite ||| UnixFileMode.UserExecute)
let previousPath = Environment.GetEnvironmentVariable "PATH"
Environment.SetEnvironmentVariable("PATH", bin + ":" + previousPath)
let invoke program =
    let info = ProcessStartInfo(program, UseShellExecute=false, RedirectStandardOutput=true, RedirectStandardError=true)
    use child = Process.Start info
    let stdout = child.StandardOutput.ReadToEndAsync()
    let stderr = child.StandardError.ReadToEndAsync()
    if not (child.WaitForExit 10000) then child.Kill(true); failwith "Executable timed out"
    child.ExitCode, stdout.Result, stderr.Result
let build name target mode (text: string) options =
    let source, output = Path.Combine(directory, name + ".ll"), Path.Combine(directory, name)
    File.WriteAllText(source, text)
    compileToNative source output target mode Set.empty options, output
let succeed = function Ok () -> () | Error message -> failwith message
let reject (result, _) = match result with Error _ -> () | Ok () -> failwith "Expected link refusal"
let host = "x86_64-unknown-linux-gnu"
let consoleIR = """
@text = private constant [6 x i8] c"hello\00"
declare i32 @puts(ptr)
define i32 @main(i32 %argc, ptr %argv) {
  %ignored = call i32 @puts(ptr @text)
  ret i32 0
}
"""
let standaloneIR = """
define void @_start() noreturn {
  call void asm sideeffect "syscall", "{rax},{rdi},~{rcx},~{r11},~{memory}"(i64 60, i64 42)
  unreachable
}
"""
try
    let result, program = build "host console" host Console consoleIR NativeLinkOptions.Empty
    succeed result
    if invoke program <> (0, "hello\n", "") then failwith "Hosted startup/runtime failed"
    let result, standalone = build "freestanding" host Freestanding standaloneIR NativeLinkOptions.Empty
    succeed result
    if invoke standalone <> (42, "", "") then failwith "Freestanding entry failed"
    let result, library = build "library.so" host Library "define i32 @answer() { ret i32 42 }" NativeLinkOptions.Empty
    succeed result
    let handle = NativeLibrary.Load library
    try
        let answer = Marshal.GetDelegateForFunctionPointer<Answer>(NativeLibrary.GetExport(handle, "answer"))
        if answer.Invoke() <> 42 then failwith "Shared library entry failed"
    finally NativeLibrary.Free handle
    let script = Path.Combine(directory, "board layout.ld")
    File.WriteAllText(script, "ENTRY(_start)\nSECTIONS { . = 0x08000000; .text : { *(.text*) } }\n")
    let result, arm = build "embedded.elf" "thumbv8m.main-none-eabi" Embedded "define void @_start() { ret void }"
                                { NativeLinkOptions.Empty with LinkerScript = Some script }
    succeed result
    let image = File.ReadAllBytes arm
    if image[4] <> 1uy || BitConverter.ToUInt16(image, 18) <> 40us then failwith "Cross-link is not ARM ELF32"
    let entry = BitConverter.ToUInt32(image, 24)
    if entry < 0x08000000u || entry >= 0x08001000u then failwith "Linker script placement was not applied"
    build "unresolved" host Console "declare i32 @absent()\ndefine i32 @main() { %v = call i32 @absent() ret i32 %v }" NativeLinkOptions.Empty |> reject
    build "missing entry" host Freestanding "define i32 @different() { ret i32 0 }" NativeLinkOptions.Empty |> reject
    build "missing runtime" "aarch64-unknown-linux-gnu" Console "define i32 @main() { ret i32 0 }" NativeLinkOptions.Empty |> reject
    build "missing script" host Embedded "define void @_start() { ret void }" {NativeLinkOptions.Empty with LinkerScript=Some(Path.Combine(directory,"missing.ld"))} |> reject
    build "unsupported image" "x86_64-pc-windows-gnu" Console "define i32 @main() { ret i32 0 }" NativeLinkOptions.Empty |> reject
    build "conflicting target" host Freestanding "target triple = \"aarch64-unknown-linux-gnu\"\ndefine void @_start() { ret void }" NativeLinkOptions.Empty |> reject
    if File.Exists trap then failwith "A forbidden compiler/codegen driver was invoked"
    printfn "PASS: hosted, shared, freestanding, scripted ARM cross-link, six refusal cases; no Clang/GCC/llc invocation. Artifacts: %s" directory
finally Environment.SetEnvironmentVariable("PATH", previousPath)
