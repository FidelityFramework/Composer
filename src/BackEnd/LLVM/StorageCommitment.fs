/// Linux ELF commitment of source-owned writable program objects. Source and
/// Alex supply identities/layout/spaces; this backend owns sections and target
/// ABI correspondence. Capacity is checked against actual linked regions.
module BackEnd.LLVM.StorageCommitment

open System
open System.IO
open System.Runtime.InteropServices
open System.Text
open BAREWire.Platform
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

type RegionPlan = {
    Section: string
    Reservation: WritableReservation
    Globals: (string * ProgramStorageEntry) list
}

let plan (triple: string) (globals: (string * ProgramStorageEntry) list) =
    try
        let target = triple.Split('-')
        if not globals.IsEmpty && (target.Length < 3 || target[2] <> "linux") then
            failwith "Writable storage requires an explicit section correspondence for the selected backend target"
        if globals |> List.map fst |> List.distinct |> List.length <> globals.Length then failwith "Writable symbols are not unique"
        if globals |> List.map (snd >> _.Identity) |> List.distinct |> List.length <> globals.Length then failwith "Writable source identities are not unique"
        globals |> List.groupBy (fun (_, entry) -> entry.SpaceNode)
        |> List.map (fun (_, members) ->
            let space = (snd members.Head).Space
            if members |> List.exists (fun (_, entry) -> entry.Space <> space) then failwith "One source space has contradictory declarations"
            // This is the Linux ELF profile's normative correspondence. SRAM
            // and other target spaces require their own backend realization.
            let section =
                if space.Kind = MemoryKind.Data then ".data"
                elif space.Kind = MemoryKind.Bss then ".bss"
                else failwithf "Linux ELF has no admitted program-storage section for declared kind '%s'" space.Kind
            let requests = members |> List.map (fun (symbol, entry) ->
                ({ Name = symbol; Length = int64 entry.Bytes; Alignment = entry.Alignment }: StorageRequest)) |> List.toArray
            match WritableStorage.reserve space requests with
            | Error findings -> failwithf "Writable reservation failed: %A" findings
            | Ok reservation -> { Section = section; Reservation = reservation; Globals = members })
        |> fun plans ->
            if plans |> List.map _.Section |> List.distinct |> List.length <> plans.Length then
                failwith "Distinct declared spaces cannot share one ELF section without an explicit joint ownership contract"
            Ok plans
    with error -> Error("Writable target mapping failed: " + error.Message)

module private LLVM =
    [<Literal>]
    let Library = "libLLVM.so"
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint LLVMContextCreate()
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMContextDispose(nativeint context)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern int LLVMCreateMemoryBufferWithContentsOfFile([<MarshalAs(UnmanagedType.LPUTF8Str)>] string path, nativeint& buffer, nativeint& error)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMDisposeMemoryBuffer(nativeint buffer)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMDisposeMessage(nativeint message)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern int LLVMParseBitcodeInContext2(nativeint context, nativeint buffer, nativeint& llvmModule)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMDisposeModule(nativeint llvmModule)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint LLVMGetDataLayoutStr(nativeint llvmModule)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint LLVMCreateTargetData([<MarshalAs(UnmanagedType.LPUTF8Str)>] string dataLayout)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMDisposeTargetData(nativeint targetData)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint LLVMGetNamedGlobal(nativeint llvmModule, [<MarshalAs(UnmanagedType.LPUTF8Str)>] string name)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint LLVMGlobalGetValueType(nativeint globalValue)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern uint64 LLVMABISizeOfType(nativeint targetData, nativeint valueType)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern uint32 LLVMABIAlignmentOfType(nativeint targetData, nativeint valueType)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern int LLVMIsGlobalConstant(nativeint globalValue)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern int LLVMIsDeclaration(nativeint globalValue)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMSetSection(nativeint globalValue, [<MarshalAs(UnmanagedType.LPUTF8Str)>] string section)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMSetAlignment(nativeint globalValue, uint32 alignment)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMSetLinkage(nativeint globalValue, int linkage)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMSetVisibility(nativeint globalValue, int visibility)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMSetUnnamedAddress(nativeint globalValue, int unnamedAddress)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern int LLVMWriteBitcodeToFile(nativeint llvmModule, [<MarshalAs(UnmanagedType.LPUTF8Str)>] string path)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern void LLVMGetVersion(uint32& major, uint32& minor, uint32& patch)
    [<DllImport(Library, CallingConvention = CallingConvention.Cdecl)>]
    extern int LLVMVerifyModule(nativeint llvmModule, int action, nativeint& error)

/// Work on the actual target-filled bitcode, using the installed LLVM C API.
/// No LLVM/MLIR text substitution supplies layout or symbol correspondence.
let realizeBitcode (path: string) (plans: RegionPlan list) =
    if plans.IsEmpty then Ok () else
    try
        let mutable major, minor, patch = 0u, 0u, 0u
        LLVM.LLVMGetVersion(&major, &minor, &patch)
        use version = new Diagnostics.Process()
        version.StartInfo <- Diagnostics.ProcessStartInfo("opt", "--version", UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true)
        if not (version.Start()) then failwith "Cannot observe the selected LLVM toolchain version"
        let output = version.StandardOutput.ReadToEndAsync()
        let errors = version.StandardError.ReadToEndAsync()
        version.WaitForExit()
        let output, errors = output.GetAwaiter().GetResult(), errors.GetAwaiter().GetResult()
        if version.ExitCode <> 0 || not (output.Contains(sprintf "version %d.%d.%d" major minor patch, StringComparison.Ordinal)) then
            failwithf "LLVM library %d.%d.%d and selected opt version disagree: %s%s" major minor patch output errors
        let context = LLVM.LLVMContextCreate()
        if context = 0n then failwith "LLVM context creation failed"
        let mutable buffer, llvmModule, targetData, error = 0n, 0n, 0n, 0n
        try
            if LLVM.LLVMCreateMemoryBufferWithContentsOfFile(path, &buffer, &error) <> 0 then
                failwith (Marshal.PtrToStringUTF8 error)
            if LLVM.LLVMParseBitcodeInContext2(context, buffer, &llvmModule) <> 0 then failwith "LLVM cannot parse the target bitcode"
            let layout = Marshal.PtrToStringUTF8(LLVM.LLVMGetDataLayoutStr llvmModule)
            if String.IsNullOrWhiteSpace layout then failwith "Target bitcode has no data layout"
            targetData <- LLVM.LLVMCreateTargetData layout
            for region in plans do
                for name, entry in region.Globals do
                    let globalValue = LLVM.LLVMGetNamedGlobal(llvmModule, name)
                    if globalValue = 0n || LLVM.LLVMIsDeclaration globalValue <> 0 then failwithf "Source writable allocation '%s' is absent from target bitcode" name
                    if LLVM.LLVMIsGlobalConstant globalValue <> 0 then failwithf "Source writable allocation '%s' became constant" name
                    let valueType = LLVM.LLVMGlobalGetValueType globalValue
                    if LLVM.LLVMABISizeOfType(targetData, valueType) <> uint64 entry.Bytes then failwithf "Target ABI extent disagrees with source storage '%s'" name
                    if LLVM.LLVMABIAlignmentOfType(targetData, valueType) > uint32 entry.Alignment then failwithf "Target ABI alignment exceeds source storage '%s'" name
                    LLVM.LLVMSetSection(globalValue, region.Section)
                    LLVM.LLVMSetAlignment(globalValue, uint32 entry.Alignment)
                    // Hidden link anchors preserve exact artifact identity;
                    // they are not exported source declarations or preemptible
                    // data. Distinct allocation addresses remain significant.
                    LLVM.LLVMSetLinkage(globalValue, 0)
                    LLVM.LLVMSetVisibility(globalValue, 1)
                    LLVM.LLVMSetUnnamedAddress(globalValue, 0)
            if error <> 0n then LLVM.LLVMDisposeMessage error; error <- 0n
            if LLVM.LLVMVerifyModule(llvmModule, 2, &error) <> 0 then failwith (Marshal.PtrToStringUTF8 error)
            if LLVM.LLVMWriteBitcodeToFile(llvmModule, path) <> 0 then failwith "LLVM cannot save committed target bitcode"
            Ok ()
        finally
            if error <> 0n then LLVM.LLVMDisposeMessage error
            if targetData <> 0n then LLVM.LLVMDisposeTargetData targetData
            if llvmModule <> 0n then LLVM.LLVMDisposeModule llvmModule
            if buffer <> 0n then LLVM.LLVMDisposeMemoryBuffer buffer
            LLVM.LLVMContextDispose context
    with error -> Error("Writable LLVM realization failed: " + error.Message)

/// Align the complete standard output sections, retaining all runtime/linker
/// contributions. A supplied linker script remains the owner's placement;
/// the same artifact observer checks it instead of silently replacing it.
let linkerScript path (plans: RegionPlan list) =
    if plans.IsEmpty then None else
    let sections = plans |> List.sortBy (fun plan -> if plan.Section = ".data" then 0 else 1) |> List.map (fun plan ->
        let space = plan.Reservation.Space
        let origin = space.Base |> Option.map (fun value -> sprintf "0x%X" value) |> Option.defaultValue (sprintf "ALIGN(%d)" space.Alignment)
        let storage = if plan.Section = ".bss" then " (NOLOAD)" else ""
        let common = if plan.Section = ".bss" then " *(COMMON)" else ""
        sprintf "  %s %s%s : { *(%s %s.*)%s }" plan.Section origin storage plan.Section plan.Section common)
    File.WriteAllText(path, "SECTIONS {\n" + String.concat "\n" sections + "\n}\nINSERT AFTER .rodata;\n")
    Some path

type private ElfSection = {
    Index: int
    NameOffset: uint64
    Kind: uint64
    Flags: uint64
    Address: int64
    Offset: int64
    Length: int64
    Link: int
    EntrySize: int64
}

/// Observe the final ELF bytes. Both 32/64-bit and both byte orders have exact
/// bounded reads; corrupt/unsupported tables fail instead of supplying defaults.
let observeElf (path: string) : Result<Map<string, WritableRegion>, string> =
    try
        let bytes = File.ReadAllBytes path
        if bytes.Length < 52 || bytes[0..3] <> [| 0x7Fuy; 0x45uy; 0x4Cuy; 0x46uy |] then failwith "Expected ELF image"
        let is64 = match bytes[4] with 1uy -> false | 2uy -> true | _ -> failwith "Unsupported ELF word class"
        let little = match bytes[5] with 1uy -> true | 2uy -> false | _ -> failwith "Unsupported ELF byte order"
        let unsigned (offset: int64) count =
            if offset < 0L || offset > int64 bytes.Length - int64 count then failwith "Truncated ELF table"
            let mutable value = 0UL
            for index in 0 .. count - 1 do
                let shift = (if little then index else count - index - 1) * 8
                value <- value ||| (uint64 bytes[int offset + index] <<< shift)
            value
        let exact value = if value > uint64 Int64.MaxValue then failwith "ELF value exceeds the exact storage observer extent" else int64 value
        let word offset = unsigned offset (if is64 then 8 else 4)
        let shoff = unsigned (if is64 then 40L else 32L) (if is64 then 8 else 4) |> exact
        let entrySize = unsigned (if is64 then 58L else 46L) 2 |> exact
        let rawCount = unsigned (if is64 then 60L else 48L) 2
        let rawNames = unsigned (if is64 then 62L else 50L) 2
        if entrySize < (if is64 then 64L else 40L) then failwith "ELF section entry is too short"
        let section index =
            if index < 0 || int64 index > (Int64.MaxValue - shoff) / entrySize then failwith "ELF section index overflows"
            let at = shoff + int64 index * entrySize
            if at < 0L || at > int64 bytes.Length - entrySize then failwith "Truncated ELF section table"
            { Index = index; NameOffset = unsigned at 4; Kind = unsigned (at + 4L) 4
              Flags = word (at + 8L)
              Address = word (at + (if is64 then 16L else 12L)) |> exact
              Offset = word (at + (if is64 then 24L else 16L)) |> exact
              Length = word (at + (if is64 then 32L else 20L)) |> exact
              Link = unsigned (at + (if is64 then 40L else 24L)) 4 |> int
              EntrySize = word (at + (if is64 then 56L else 36L)) |> exact }
        let first = section 0
        let count = if rawCount = 0UL then first.Length else exact rawCount
        let nameIndex = if rawNames = 65535UL then first.Link else int rawNames
        if count <= 0L || shoff < 0L || shoff > int64 bytes.Length || count > (int64 bytes.Length - shoff) / entrySize || count > int64 Int32.MaxValue then
            failwith "ELF section count exceeds the image"
        let sections = Array.init (int count) section
        let phoff = unsigned (if is64 then 32L else 28L) (if is64 then 8 else 4) |> exact
        let phsize = unsigned (if is64 then 54L else 42L) 2 |> exact
        let phcount = unsigned (if is64 then 56L else 44L) 2
        if phcount = 65535UL then failwith "Extended ELF program-header count requires an explicit observer implementation"
        if phcount > 0UL && (phsize < (if is64 then 56L else 32L) || phoff > int64 bytes.Length || int64 phcount > (int64 bytes.Length - phoff) / phsize) then
            failwith "Truncated ELF program-header table"
        let loads =
            [ for index in 0L .. int64 phcount - 1L do
                let at = phoff + index * phsize
                if unsigned at 4 = 1UL then
                    let flags = unsigned (at + (if is64 then 4L else 24L)) 4
                    let address = word (at + (if is64 then 16L else 8L)) |> exact
                    let extent = word (at + (if is64 then 40L else 20L)) |> exact
                    if extent > Int64.MaxValue - address then failwith "ELF mapped extent overflows"
                    yield address, address + extent, flags ]
        if nameIndex < 0 || nameIndex >= sections.Length then failwith "ELF section-name table is missing"
        let name (table: ElfSection) offset =
            let offset = exact offset
            if table.Kind <> 3UL || table.Offset < 0L || table.Length < 0L || table.Offset > int64 bytes.Length - table.Length || offset < 0L || offset >= table.Length then
                failwith "ELF string offset is outside its table"
            let start = table.Offset + offset
            let mutable ending = start
            while ending < table.Offset + table.Length && bytes[int ending] <> 0uy do ending <- ending + 1L
            if ending = table.Offset + table.Length then failwith "Unterminated ELF symbol name"
            Encoding.UTF8.GetString(bytes, int start, int (ending - start))
        let names = sections |> Array.map (fun row -> name sections[nameIndex] row.NameOffset)
        let objects = ResizeArray<int * WritableObject>()
        for symbols in sections do
            if symbols.Kind = 2UL || symbols.Kind = 11UL then
                let minimum = if is64 then 24L else 16L
                if symbols.EntrySize < minimum || symbols.Length % symbols.EntrySize <> 0L ||
                   symbols.Offset < 0L || symbols.Offset > int64 bytes.Length - symbols.Length ||
                   symbols.Link < 0 || symbols.Link >= sections.Length then failwith "Malformed ELF symbol table"
                let strings = sections[symbols.Link]
                for ordinal in 0L .. symbols.Length / symbols.EntrySize - 1L do
                    let at = symbols.Offset + ordinal * symbols.EntrySize
                    let symbolName = name strings (unsigned at 4)
                    let info = unsigned (at + (if is64 then 4L else 12L)) 1
                    let sectionIndex = unsigned (at + (if is64 then 6L else 14L)) 2 |> int
                    let address = word (at + (if is64 then 8L else 4L)) |> exact
                    let size = word (at + (if is64 then 16L else 8L)) |> exact
                    if info &&& 15UL = 1UL && size > 0L && sectionIndex > 0 && sectionIndex < sections.Length then
                        objects.Add(sectionIndex, { Name = symbolName; Address = address; Length = size })
        let rows =
            sections |> Array.choose (fun row ->
                if row.Flags &&& 2UL = 0UL then None else
                if row.Length > Int64.MaxValue - row.Address then failwith "ELF section extent overflows"
                let ending = row.Address + row.Length
                let mapped = loads |> List.filter (fun (start, finish, _) -> start <= row.Address && ending <= finish)
                let overlaps = loads |> List.filter (fun (start, finish, _) -> start < ending && row.Address < finish)
                if mapped.IsEmpty then failwithf "Allocated ELF section '%s' has no complete mapped load region" names[row.Index]
                let writable = row.Flags &&& 1UL <> 0UL
                let executable = row.Flags &&& 4UL <> 0UL
                if overlaps |> List.exists (fun (_, _, flags) -> flags &&& 4UL = 0UL || (writable && flags &&& 2UL = 0UL) || (executable && flags &&& 1UL = 0UL)) then
                    failwithf "ELF load permissions disagree with allocated section '%s'" names[row.Index]
                // A writable source object cannot acquire executable access
                // through a broader load mapping despite innocuous SHF flags.
                let access =
                    if executable || overlaps |> List.exists (fun (_, _, flags) -> flags &&& 1UL <> 0UL) then Access.ReadExecute
                    elif writable then Access.ReadWrite else Access.ReadOnly
                Some(names[row.Index],
                    { Section = names[row.Index]; Address = row.Address; Length = row.Length
                      Allocated = true; Access = access
                      Objects = objects |> Seq.filter (fst >> (=) row.Index) |> Seq.map snd |> Seq.distinct |> Seq.toArray }))
        if rows |> Array.map fst |> Array.distinct |> Array.length <> rows.Length then failwith "ELF allocated section names are not unique"
        Ok(Map.ofArray rows)
    with error -> Error("Writable ELF observation failed: " + error.Message)

let verifyNative path plans =
    if List.isEmpty plans then Ok [] else
    observeElf path |> Result.bind (fun regions ->
        let results = plans |> List.map (fun plan ->
            match regions.TryFind plan.Section with
            | None -> Error(sprintf "Missing mapped writable section %s" plan.Section)
            | Some region ->
                WritableStorage.commit plan.Reservation plan.Section region
                |> Result.mapError (fun findings -> sprintf "Writable region commitment failed: %A" findings))
        results |> List.fold (fun accumulated result ->
            match accumulated, result with
            | Ok values, Ok value -> Ok(value :: values)
            | Error reason, _ | _, Error reason -> Error reason) (Ok [])
        |> Result.map List.rev)
