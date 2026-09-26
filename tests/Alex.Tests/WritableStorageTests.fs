module Alex.Tests.WritableStorageTests

open System
open System.Buffers.Binary
open System.Diagnostics
open System.IO
open Xunit
open BAREWire.Platform
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Core.Types.Pipeline
open BackEnd.LLVM.StorageCommitment

let private succeed = function Ok value -> value | Error message -> failwith message
let private target = BackEnd.LLVM.Codegen.getDefaultTarget()

let private space: MemorySpace =
    { Name = "declared-data"; Kind = MemoryKind.Data; Capacity = 4096L
      Alignment = 4096; Granularity = 4096; Growth = Growth.Fixed
      Access = Access.ReadWrite; Base = None; Notes = ""; MapKind = ""; Since = ""; Until = "" }

let private entry id: ProgramStorageEntry =
    { Identity = ProgramStorageIdentity.BindingSlot(NodeId id)
      SourceType = Types.boolType; Shape = ProgramStorageShape.Scalar SettledSlot.Bool
      Bytes = 1; Alignment = 1; SpaceNode = NodeId 1; Space = space
      Participants = Set.ofList [NodeId 1; NodeId id] }

let private globals = ["owned_first", entry 101; "owned_second", entry 102]

let private llvm = """@owned_first = private global [1 x i1] undef
@owned_second = private global [1 x i1] undef
@runtime_payload = hidden global [64 x i8] zeroinitializer, section ".data", align 1
define i32 @main() {
 store i1 true, ptr @owned_first
 store i1 false, ptr @owned_second
 %first = load i1, ptr @owned_first
 %second = load i1, ptr @owned_second
 %not_second = xor i1 %second, true
 %both = and i1 %first, %not_second
 %exit = select i1 %both, i32 0, i32 23
 ret i32 %exit
}
"""

let private withImage held action =
    let directory = Path.Combine(Path.GetTempPath(), "composer writable " + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory directory |> ignore
    try
        let source, binary = Path.Combine(directory, "input.ll"), Path.Combine(directory, "program")
        File.WriteAllText(source, llvm)
        let result =
            BackEnd.LLVM.Codegen.compileToNativeWithStorage source binary target
                Core.Types.Dialects.Console Set.empty NativeLinkOptions.Empty None held
        action directory binary result
    finally
        Directory.Delete(directory, true)

[<Fact>]
let ``linked writable commitment preserves two actual identities and counts the complete mapped region`` () =
    withImage globals (fun directory binary result ->
        result |> succeed
        let plans = plan target globals |> succeed
        let commitment = verifyNative binary plans |> succeed |> Assert.Single
        Assert.Equal(".data", commitment.Section)
        Assert.Equal(4096L, commitment.AllocationSize)
        Assert.True(commitment.UsedSize >= 66L)
        Assert.Equal(2, commitment.Objects.Length)
        let first = commitment.Objects |> Array.find (fun value -> value.Name = "owned_first")
        let second = commitment.Objects |> Array.find (fun value -> value.Name = "owned_second")
        Assert.Equal(1L, first.Length)
        Assert.Equal(1L, second.Length)
        Assert.NotEqual(first.Address, second.Address)
        let region = (observeElf binary |> succeed)[".data"]
        Assert.Contains(region.Objects, fun value -> value.Name = "runtime_payload" && value.Length = 64L)
        Assert.True(File.Exists(Path.Combine(directory, "input.storage.json")))
        use child = Process.Start(ProcessStartInfo(binary, UseShellExecute = false))
        if not (child.WaitForExit 10000) then
            child.Kill(true)
            failwith "Writable identity executable did not terminate"
        Assert.Equal(0, child.ExitCode))

[<Fact>]
let ``individual and aggregate source payload fit cannot certify runtime contributions and padding`` () =
    withImage globals (fun _ binary result ->
        result |> succeed
        let tooSmall = globals |> List.map (fun (name, held) -> name, { held with Space = { space with Capacity = 2L; Alignment = 1; Granularity = 1 } })
        let plans = plan target tooSmall |> succeed
        Assert.Equal(2L, (Assert.Single plans).Reservation.PayloadSize)
        match verifyNative binary plans with
        | Error reason -> Assert.Contains("Writable region commitment failed", reason)
        | Ok _ -> failwith "Two source bytes hid the actual region's runtime objects and padding")

[<Fact>]
let ``actual target ABI extent must equal the settled source object`` () =
    let changed = globals |> List.map (fun (name, held) -> name, { held with Bytes = 2 })
    withImage changed (fun _ _ result ->
        match result with
        | Error reason -> Assert.Contains("Target ABI extent disagrees", reason)
        | Ok () -> failwith "An invented source extent acquired native storage authority")

[<Fact>]
let ``backend correspondence rejects foreign targets and ambiguous section ownership`` () =
    Assert.True((plan "x86_64-pc-windows-linux" globals).IsError)
    Assert.True((plan "x86_64-unknown-freebsd" globals).IsError)
    let sram = globals |> List.map (fun (name, held) -> name, { held with Space = { space with Kind = MemoryKind.Sram } })
    Assert.True((plan target sram).IsError)
    let split = globals |> List.mapi (fun index (name, held) -> name, { held with SpaceNode = NodeId(index + 1) })
    Assert.True((plan target split).IsError)

[<Fact>]
let ``observed section authority retracts when the actual load mapping loses write access`` () =
    withImage globals (fun _ binary result ->
        result |> succeed
        let region = (observeElf binary |> succeed)[".data"]
        let bytes = File.ReadAllBytes binary
        Assert.Equal(2uy, bytes[4])
        Assert.Equal(1uy, bytes[5])
        let read16 offset = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(offset, 2)) |> int
        let read64 offset = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(offset, 8))
        let table, size, count = int (read64 32), read16 54, read16 56
        let mutable changed = false
        for index in 0 .. count - 1 do
            let at = table + index * size
            let kind = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(at, 4))
            if kind = 1u && read64 (at + 16) <= uint64 region.Address && uint64 region.Address + uint64 region.Length <= read64 (at + 16) + read64 (at + 40) then
                let flags = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(at + 4, 4))
                BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(at + 4, 4), flags &&& ~~~2u)
                changed <- true
        Assert.True(changed)
        File.WriteAllBytes(binary, bytes)
        match observeElf binary with
        | Error reason -> Assert.Contains("load permissions disagree", reason)
        | Ok _ -> failwith "Section flags hid a read-only actual load mapping")

[<Fact>]
let ``a partial executable load overlap cannot hide behind a complete writable mapping`` () =
    withImage globals (fun _ binary result ->
        result |> succeed
        let region = (observeElf binary |> succeed)[".data"]
        let bytes = File.ReadAllBytes binary
        Assert.Equal(2uy, bytes[4])
        Assert.Equal(1uy, bytes[5])
        let read16 offset = BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(offset, 2)) |> int
        let read64 offset = BinaryPrimitives.ReadUInt64LittleEndian(bytes.AsSpan(offset, 8))
        let write64 offset value = BinaryPrimitives.WriteUInt64LittleEndian(bytes.AsSpan(offset, 8), value)
        let table, size, count = int (read64 32), read16 54, read16 56
        let spare =
            [0 .. count - 1] |> List.tryFind (fun index ->
                BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(table + index * size, 4)) <> 1u)
            |> Option.defaultWith (fun () -> failwith "Native fixture has no non-load header for the adversarial overlap")
        let at = table + spare * size
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(at, 4), 1u)
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(at + 4, 4), 7u)
        write64 (at + 16) (uint64 region.Address + 1UL)
        write64 (at + 40) 1UL
        File.WriteAllBytes(binary, bytes)
        match verifyNative binary (plan target globals |> succeed) with
        | Error reason -> Assert.Contains("wrong-permissions", reason)
        | Ok _ -> failwith "A partial executable load acquired writable-only authority")

[<Fact>]
let ``truncated linked artifact supplies no storage evidence`` () =
    withImage globals (fun _ binary result ->
        result |> succeed
        let bytes = File.ReadAllBytes binary
        File.WriteAllBytes(binary, bytes[0..47])
        Assert.True((verifyNative binary (plan target globals |> succeed)).IsError))
