/// Sample 19 (library file): module-level values in a module other than the entry module.
/// Module-level values are program-lifetime slots initialized before main runs; every
/// reference, in any function of any module, reloads from the slot.
namespace Examples.ModuleValuesLib

type Space = { Name: string; Capacity: int64 }

module Spaces =
    let rodata : Space = { Name = "rodata"; Capacity = 4096L }
    let stack : Space = { Name = "stack"; Capacity = 8388608L }
    let all : Space array = [| rodata; stack |]
    let wordSize : int = 8
    let mutable counter : int = 0

    /// References module-level values from a function in the same module.
    let bump () : int =
        counter <- counter + wordSize
        counter

    /// Reads a record out of a module-level array.
    let first (spaces: Space array) : string =
        let s = Array.get spaces 0
        s.Name

    let total (spaces: Space array) : int64 =
        let mutable acc = 0L
        let mutable i = 0
        while i < Array.length spaces do
            let s = Array.get spaces i
            acc <- acc + s.Capacity
            i <- i + 1
        acc
