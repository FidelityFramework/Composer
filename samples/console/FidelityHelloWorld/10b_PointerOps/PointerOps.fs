/// Sample 10b: Arena Memory Management (DMM Infrastructure)
/// Validates the Deterministic Memory Management primitives that
/// closures with mutable captures depend on.
///
/// The Arena is the DMM mechanism for managing memory with known
/// lifetime boundaries. Memory is allocated from an arena and
/// lives as long as the arena's backing storage (stack frame).
///
/// Pattern (Phase 1 explicit):
///   1. NativePtr.stackalloc — allocate arena backing on stack
///   2. Arena.fromPointer — create arena from backing storage
///   3. Arena.alloc — allocate from arena
///   4. NativePtr.ofNativeInt/read/write — access arena-allocated values
///   5. byref<Arena<'a>> — thread arena through function calls
///
/// Dependencies: mutable bindings (sample 10a), arithmetic (sample 05)
module ArenaMemory

open Console
open Format

/// Allocate and initialize an int value in the arena
/// Returns nativeint address of allocated storage
let allocInt (arena: byref<Arena<'a>>) (value: int) : nativeint =
    let addr = Arena.alloc &arena (Platform.wordSize ())
    NativePtr.write (NativePtr.ofNativeInt<int> addr) value
    addr

/// Read an int from arena-allocated storage
let readInt (addr: nativeint) : int =
    NativePtr.read (NativePtr.ofNativeInt<int> addr)

/// Write an int to arena-allocated storage
let writeInt (addr: nativeint) (value: int) : unit =
    NativePtr.write (NativePtr.ofNativeInt<int> addr) value

[<EntryPoint>]
let main _ =
    // Create arena on main's stack — all allocations live here
    let arenaMem = NativePtr.stackalloc<byte> 4096
    let mutable arena = Arena.fromPointer (NativePtr.toNativeInt arenaMem) 4096

    Console.writeln "=== Arena Memory Test ==="

    // ─── Basic Arena Allocation ───
    Console.writeln "--- Basic Allocation ---"

    let addr1 = Arena.alloc &arena (Platform.wordSize ())
    NativePtr.write (NativePtr.ofNativeInt<int> addr1) 42
    Console.write "alloc and read: "
    Console.writeln (Format.int (NativePtr.read (NativePtr.ofNativeInt<int> addr1)))

    // ─── Helper Functions (byref<Arena> threading) ───
    Console.writeln ""
    Console.writeln "--- Arena Threading ---"

    let a = allocInt &arena 10
    let b = allocInt &arena 20
    let c = allocInt &arena 30
    Console.write "a="
    Console.write (Format.int (readInt a))
    Console.write ", b="
    Console.write (Format.int (readInt b))
    Console.write ", c="
    Console.writeln (Format.int (readInt c))

    // ─── Mutation of Arena-Allocated Values ───
    Console.writeln ""
    Console.writeln "--- Arena Mutation ---"

    let counter = allocInt &arena 0
    Console.write "initial: "
    Console.writeln (Format.int (readInt counter))

    writeInt counter (readInt counter + 1)
    Console.write "after +1: "
    Console.writeln (Format.int (readInt counter))

    writeInt counter (readInt counter + 10)
    Console.write "after +10: "
    Console.writeln (Format.int (readInt counter))

    // ─── Multiple Independent Values ───
    Console.writeln ""
    Console.writeln "--- Independent Values ---"

    let x = allocInt &arena 100
    let y = allocInt &arena 200
    Console.write "x="
    Console.write (Format.int (readInt x))
    Console.write ", y="
    Console.writeln (Format.int (readInt y))

    // Modify x without affecting y
    writeInt x 999
    Console.write "after x=999: x="
    Console.write (Format.int (readInt x))
    Console.write ", y="
    Console.writeln (Format.int (readInt y))

    0
