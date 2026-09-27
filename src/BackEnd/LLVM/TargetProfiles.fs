/// Target ABI capabilities, independent of program startup and diagnostic effects.
module BackEnd.LLVM.TargetProfiles

open Core.Types.Pipeline

type LinuxAmd64C = private | LinuxAmd64C
type LinuxAmd64Syscalls = private | LinuxAmd64Syscalls

let linuxAmd64Syscalls (context: BackEndContext) (triple: string) =
    let linuxAbi =
        match triple.Split '-' with
        | [| "x86_64"; "linux"; environment |]
        | [| "x86_64"; _; "linux"; environment |] -> environment = "gnu" || environment = "musl"
        | _ -> false
    if context.PlatformOS = Some "linux"
       && context.TargetPointerBits = Some 64 && linuxAbi then Some LinuxAmd64Syscalls
    else None

let libcAmd64 (context: BackEndContext) (triple: string) =
    if (linuxAmd64Syscalls context triple).IsSome
       && context.RuntimeModel = Some Clef.Compiler.NativeTypedTree.NativeTypes.RuntimeModel.Libc then Some LinuxAmd64C
    else None
