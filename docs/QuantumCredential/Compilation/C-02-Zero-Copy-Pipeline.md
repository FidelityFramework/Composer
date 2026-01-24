# C-02: Zero-Copy Pipeline

## Overview

This document explains why managed runtimes (Python, .NET, Java) are fundamentally unsuitable for security-critical entropy generation, and how Firefly's native compilation eliminates the security liabilities that managed memory creates.

**Key Principle**: Every memory copy creates another location where sensitive data resides. Managed runtimes copy implicitly; native compilation copies only when explicitly specified.

---

## For Software Engineers: The Hidden Copies

Consider this innocent-looking C# code:

```csharp
var entropyBytes = new byte[32];
rng.GetBytes(entropyBytes);
cryptoProvider.CreateKey(entropyBytes);
Array.Clear(entropyBytes, 0, entropyBytes.Length);
```

The developer believes:
1. Entropy is generated into `entropyBytes`
2. The key is created from that entropy
3. The entropy is securely cleared

What actually happens:
1. The runtime allocates `entropyBytes` on the managed heap
2. `GetBytes` may copy through intermediate buffers
3. `CreateKey` likely copies the bytes internally
4. The GC may have already copied the array during compaction
5. `Array.Clear` zeroes one location; copies remain elsewhere

The developer's mental model does not match reality. The entropy has propagated to multiple memory locations, only one of which gets cleared.

---

## The Managed Memory Model

### Garbage Collection and Copying

Managed runtimes use generational garbage collection. Objects move through generations:

```
Gen 0 (Eden) → Gen 1 → Gen 2 (Long-lived)
```

Each promotion may involve copying:
- The object moves to a new memory location
- The old location is not zeroed
- The old location is simply marked as available for reuse

For cryptographic material, this means:
- Entropy bytes may exist in Gen 0 location
- After GC, they also exist in Gen 1 location
- Neither location is explicitly cleared
- Memory dumps capture both

### Interop Marshaling

When managed code calls native libraries (OpenSSL, system crypto):

```csharp
// Managed array
byte[] managedBuffer = GetEntropy();

// Marshal to native - COPY
IntPtr nativeBuffer = Marshal.AllocHGlobal(managedBuffer.Length);
Marshal.Copy(managedBuffer, 0, nativeBuffer, managedBuffer.Length);

// Call native function
NativeCryptoFunction(nativeBuffer);

// Free native buffer - but managed copy still exists
Marshal.FreeHGlobal(nativeBuffer);
```

The managed buffer persists even after the native buffer is freed. The developer must remember to clear both. Most do not.

### Object Allocation

In languages like C#, even passing bytes between methods may involve allocations:

```csharp
byte[] ProcessEntropy(byte[] input)
{
    // New allocation
    byte[] output = new byte[input.Length];

    // Copy
    for (int i = 0; i < input.Length; i++)
        output[i] = Transform(input[i]);

    return output;  // Another reference, both buffers exist
}
```

Each intermediate step may create new copies. Functional-style code exacerbates this.

---

## The Attack Surface Expansion

Each copy expands the attack surface:

| Copies | Attack Surface |
|--------|----------------|
| 1 | One memory location to protect |
| 2 | Two locations; attacker needs to find either |
| 4 | Four locations; probability of capture quadruples |
| N | N locations; security degrades linearly |

### Memory Dump Attacks

If an attacker obtains a memory dump (through malware, cold boot attack, or crash dump):
- Every copy of sensitive data is captured
- Even "cleared" buffers may have live copies elsewhere
- GC heap fragmentation spreads data across memory

### Side-Channel Leakage

Multiple copies increase timing attack surface:
- Cache line access patterns reveal which copies are touched
- More copies mean more observable cache activity
- Patterns emerge from copy propagation

---

## Firefly's Zero-Copy Architecture

Firefly compiles F# to native code without a managed runtime. There is:
- No garbage collector
- No managed heap
- No implicit copying
- No marshaling layer

### Direct Hardware Access

```fsharp
// F# with Firefly - compiles to direct register access
let readChannel (ch: int) : uint16 =
    Platform.Bindings.ADC.readChannel ch
```

This compiles to:
1. SPI transaction setup (register writes)
2. Clock generation
3. MISO sampling
4. Value extraction

No intermediate buffers. No copies. The 10-bit ADC value flows directly from hardware register to program variable.

### Stack Allocation

Local variables live on the stack:

```fsharp
let generateEntropyByte () : byte =
    let s0 = readChannel 0  // Stack variable
    let s1 = readChannel 1  // Stack variable
    let s2 = readChannel 2  // Stack variable
    let s3 = readChannel 3  // Stack variable
    (byte s0) ^^^ (byte s1) ^^^ (byte s2) ^^^ (byte s3)
```

When the function returns:
- Stack frame is popped
- Memory is immediately available for reuse
- Next function call overwrites the values
- No lingering copies

### Explicit Lifetime Control

With arena allocation:

```fsharp
let generateEntropy (arena: Arena<'a>) (count: int) : byte[] =
    let buffer = Arena.allocate arena count
    for i in 0 .. count - 1 do
        buffer.[i] <- generateEntropyByte()
    buffer
```

The buffer exists in a known location with a known lifetime. When the arena is cleared:
- The memory can be explicitly zeroed
- No copies exist elsewhere
- The developer has complete control

---

## MLIR Representation

The zero-copy property is preserved through MLIR:

```mlir
func.func @generateEntropyByte() -> i8 {
    // Direct calls to ADC read - no intermediate allocations
    %s0 = func.call @adc_readChannel(%c0) : (index) -> i16
    %s1 = func.call @adc_readChannel(%c1) : (index) -> i16
    %s2 = func.call @adc_readChannel(%c2) : (index) -> i16
    %s3 = func.call @adc_readChannel(%c3) : (index) -> i16

    // Truncate to byte - value stays in register
    %b0 = arith.trunci %s0 : i16 to i8
    %b1 = arith.trunci %s1 : i16 to i8
    %b2 = arith.trunci %s2 : i16 to i8
    %b3 = arith.trunci %s3 : i16 to i8

    // XOR tree - all in registers
    %x01 = arith.xori %b0, %b1 : i8
    %x23 = arith.xori %b2, %b3 : i8
    %result = arith.xori %x01, %x23 : i8

    return %result : i8
}
```

The entire operation occurs in CPU registers. No memory allocation. No copying. Values flow from hardware through computation to output.

---

## Comparison: 4096 Bytes of Entropy

### Python Implementation

```python
entropy = bytearray(4096)
for i in range(4096):
    # Read 4 channels
    s0 = adc.read(0)  # Allocates int object
    s1 = adc.read(1)  # Allocates int object
    s2 = adc.read(2)  # Allocates int object
    s3 = adc.read(3)  # Allocates int object

    # XOR - allocates more int objects
    entropy[i] = (s0 ^ s1 ^ s2 ^ s3) & 0xFF
```

Memory behavior:
- 16,384 integer object allocations (4 channels × 4096 iterations)
- Each integer is 28+ bytes in Python
- ~450KB of allocations for 4KB of output
- GC will collect these, leaving memory traces

### .NET Implementation

```csharp
var entropy = new byte[4096];
for (int i = 0; i < 4096; i++)
{
    var s0 = adc.Read(0);  // May box
    var s1 = adc.Read(1);
    var s2 = adc.Read(2);
    var s3 = adc.Read(3);

    entropy[i] = (byte)((s0 ^ s1 ^ s2 ^ s3) & 0xFF);
}
```

Memory behavior:
- One heap allocation for array (may move during GC)
- Intermediate values may be boxed in some contexts
- SPI driver uses kernel buffers (copies at driver boundary)
- 2-4 copies per byte before it reaches the array

### Firefly Implementation

```fsharp
let entropy = Arena.allocate arena 4096
for i in 0 .. 4095 do
    entropy.[i] <- generateEntropyByte()
entropy
```

Memory behavior:
- One arena allocation (stack or explicit region)
- No intermediate copies
- Values flow from ADC register to XOR result to memory
- Total: 4096 bytes allocated, 4096 bytes written, zero copies

---

## Security Implications

### Managed Runtime Risks

| Risk | Cause | Impact |
|------|-------|--------|
| Data remnants | GC copying | Entropy persists in memory |
| Timing leaks | Heap access patterns | Side-channel information |
| Dump exposure | Multiple copies | Larger attack surface |
| Unpredictable cleanup | GC scheduling | Cannot guarantee erasure |

### Native Compilation Guarantees

| Property | Mechanism | Guarantee |
|----------|-----------|-----------|
| No hidden copies | Stack allocation, explicit arenas | Data exists in exactly specified locations |
| Predictable layout | Compiler-controlled memory | Known addresses, controllable access |
| Immediate cleanup | Stack pop, arena clear | Memory reused immediately |
| Minimal footprint | Register-based computation | Entropy in registers, not memory |

---

## Connection to Other Documents

- **E-02-Epsilon-Evaluation**: The epsilon data also benefits from zero-copy
- **C-01-SCF-Parallel-Pattern**: Parallel sampling must maintain zero-copy property
- **V-03-EM-Attack-Resistance**: Fewer copies means fewer side-channel opportunities

---

## Summary

Managed runtimes create implicit copies that expand the attack surface for cryptographic material. Firefly's native compilation eliminates this entire class of vulnerability:

| Aspect | Managed Runtime | Native Compilation |
|--------|-----------------|-------------------|
| Copy count | Unpredictable, many | Zero or explicit |
| Memory locations | Scattered by GC | Precise, controlled |
| Cleanup guarantee | None (GC dependent) | Explicit, immediate |
| Attack surface | Expands with time | Minimal, bounded |

For security-critical entropy generation, zero-copy is not an optimization; it is a security requirement. Native compilation is the only path to that requirement.
