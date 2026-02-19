/// BitwiseOps - Exercises idiomatic F# bitwise operators on integers
///
/// F#'s bitwise operators (&&&, |||, ^^^, ~~~, <<<, >>>) are type-preserving:
/// the result type always equals the operand type.  No hidden coercions,
/// no C-style byte-order magic — purely dimensional operations within a type.
///
/// This sample validates the lowering path:
///   F# source → PSG intrinsic → classifyAtomicOp → arith dialect MLIR
module BitwiseOps

[<EntryPoint>]
let main _ =
    // AND: isolate a bit field from packed flags
    let flags = 0b11010110   // 214
    let mask  = 0b00111100   // 60
    let field = flags &&& mask  // 0b00010100 = 20
    Console.write "AND: "
    Console.writeln (Format.int field)

    // OR: combine two disjoint flag sets
    let lo = 0b10100000   // 160
    let hi = 0b00000101   // 5
    let combined = lo ||| hi  // 0b10100101 = 165
    Console.write "OR:  "
    Console.writeln (Format.int combined)

    // XOR: toggle bits
    let x = 0b11001100   // 204
    let y = 0b01010101   // 85
    let toggled = x ^^^ y  // 0b10011001 = 153
    Console.write "XOR: "
    Console.writeln (Format.int toggled)

    // NOT: bitwise complement (all bits flipped)
    // ~~~0 = -1 on signed int (all ones in two's complement)
    let zero = 0
    let complement = ~~~zero   // -1
    Console.write "NOT: "
    Console.writeln (Format.int complement)

    // Shift left: scale by a power of 2
    let one = 1
    let shifted_left = one <<< 4   // 16
    Console.write "SHL: "
    Console.writeln (Format.int shifted_left)

    // Shift right: arithmetic (sign-preserving) divide by power of 2
    let big = 256
    let shifted_right = big >>> 3  // 32
    Console.write "SHR: "
    Console.writeln (Format.int shifted_right)

    0
