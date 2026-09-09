#I "../src/bin/Debug/net10.0"
#r "Fidelity.Data.dll"
#r "Clef.Compiler.Service.dll"
#r "Composer.dll"
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Patterns.MemoryPatterns

let cases = [
    "high unsigned bit", ValueRange.Bounded (128I, 255I), true
    "zero", ValueRange.point 0I, true
    "nonnegative unbounded", ValueRange.Above 0I, true
    "negative index", ValueRange.point -1I, false
    "mixed sign", ValueRange.Bounded (-128I, 127I), false
    "unknown", ValueRange.Unbounded, false
    "empty is not proof", ValueRange.Empty, false ]
for name, range, unsigned in cases do
    let operation = indexCastForRange range (V (1, 0)) (V (1, 1)) (TInt (IntWidth 8))
    let actual = match operation with
                 | MLIROp.IndexOp (IndexOp.IndexCastU _) -> true
                 | MLIROp.IndexOp (IndexOp.IndexCastS _) -> false
                 | _ -> failwithf "%s: unexpected operation %A" name operation
    if actual <> unsigned then failwithf "%s: index extension changed numeric meaning" name
printfn "PASS 7 array-index signedness cases"
