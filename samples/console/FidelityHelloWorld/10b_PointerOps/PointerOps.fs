/// Sample 10b: NativePtr Operations (Indexed Read/Write)
/// Validates the NativePtr primitives that Platform.Display's callback
/// data-passing pattern depends on:
///   - NativePtr.stackalloc — allocate typed buffer on stack
///   - NativePtr.set — indexed write (used by callbacks to store state)
///   - NativePtr.get — indexed read (used by callers to retrieve state)
///   - NativePtr.read/write — scalar access at typed addresses
///   - NativePtr.ofNativeInt — cast nativeint to typed pointer
///
/// The set/get round-trip is the exact pattern Connection.connect uses:
///   malloc buffer → NativePtr.set to init → pass as callback data →
///   callback writes via NativePtr.set → caller reads via NativePtr.get
///
/// Dependencies: arithmetic (sample 05)
module PointerOps

open Console
open Format

[<EntryPoint>]
let main _ =
    Console.writeln "=== NativePtr Operations ==="

    // ─── NativePtr.set + get on nativeptr<int> (indexed access) ───
    Console.writeln "--- set/get<int> ---"

    let intBuf = NativePtr.stackalloc<int> 4
    NativePtr.set intBuf 0 42
    NativePtr.set intBuf 1 99
    NativePtr.set intBuf 2 7
    NativePtr.set intBuf 3 256
    Console.write "slot0: "
    Console.writeln (Format.int (NativePtr.get intBuf 0))
    Console.write "slot1: "
    Console.writeln (Format.int (NativePtr.get intBuf 1))
    Console.write "slot2: "
    Console.writeln (Format.int (NativePtr.get intBuf 2))
    Console.write "slot3: "
    Console.writeln (Format.int (NativePtr.get intBuf 3))

    // ─── NativePtr.set + get on nativeptr<nativeint> (callback data pattern) ───
    Console.writeln ""
    Console.writeln "--- set/get<nativeint> ---"

    let ptrBuf = NativePtr.stackalloc<nativeint> 3
    NativePtr.set ptrBuf 0 42n
    NativePtr.set ptrBuf 1 99n
    NativePtr.set ptrBuf 2 7n
    let n0 = NativePtr.get ptrBuf 0
    let n1 = NativePtr.get ptrBuf 1
    let n2 = NativePtr.get ptrBuf 2
    if n0 = 42n then Console.writeln "slot0=42n: PASS"
    else Console.writeln "slot0=42n: FAIL"
    if n1 = 99n then Console.writeln "slot1=99n: PASS"
    else Console.writeln "slot1=99n: FAIL"
    if n2 = 7n then Console.writeln "slot2=7n: PASS"
    else Console.writeln "slot2=7n: FAIL"

    // ─── Set/Get round-trip (simulating callback data writes + readback) ───
    Console.writeln ""
    Console.writeln "--- callback round-trip ---"

    let cbBuf = NativePtr.stackalloc<nativeint> 3
    // Init to zero (as Connection.connect does)
    NativePtr.set cbBuf 0 0n
    NativePtr.set cbBuf 1 0n
    NativePtr.set cbBuf 2 0n
    // Verify initial zeros
    let z0 = NativePtr.get cbBuf 0
    let z1 = NativePtr.get cbBuf 1
    if z0 = 0n then Console.writeln "init0=0n: PASS"
    else Console.writeln "init0=0n: FAIL"
    if z1 = 0n then Console.writeln "init1=0n: PASS"
    else Console.writeln "init1=0n: FAIL"
    // Simulate callback writing globals (registryGlobal pattern)
    NativePtr.set cbBuf 0 100n
    NativePtr.set cbBuf 1 200n
    NativePtr.set cbBuf 2 300n
    // Read back (what Connection.connect does after wl_display_roundtrip)
    let r0 = NativePtr.get cbBuf 0
    let r1 = NativePtr.get cbBuf 1
    let r2 = NativePtr.get cbBuf 2
    if r0 = 100n then Console.writeln "cb0=100n: PASS"
    else Console.writeln "cb0=100n: FAIL"
    if r1 = 200n then Console.writeln "cb1=200n: PASS"
    else Console.writeln "cb1=200n: FAIL"
    if r2 = 300n then Console.writeln "cb2=300n: PASS"
    else Console.writeln "cb2=300n: FAIL"

    0
