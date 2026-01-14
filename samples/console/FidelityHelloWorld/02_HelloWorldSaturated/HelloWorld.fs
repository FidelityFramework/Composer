module Examples.HelloWorldSaturated

// Arena<'lifetime> is now an FNCS intrinsic - no import needed
// BAREWire.Core.Capability is only needed for higher-level capability types

/// Demonstrates SATURATED function calls with ARENA-based memory management.
/// Uses FNCS Console intrinsics and FNCS Arena for deterministic allocation.
///
/// The arena is created on main's stack and passed to hello().
/// String data allocated from arena survives function returns.
let hello (arena: byref<Arena<'lifetime>>) =
    Console.write "Enter your name: "
    let name = Console.readlnFrom &arena  // Allocates from arena, survives this call
    Console.writeln $"Hello, {name}!"

[<EntryPoint>]
let main argv =
    // Create arena on main's stack - 4KB for string allocations
    let arenaMem = NativePtr.stackalloc<byte> 4096
    let mutable arena = Arena.fromPointer (NativePtr.toNativeInt arenaMem) 4096

    hello &arena
    0
