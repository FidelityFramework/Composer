/// Sample 14: Minimal Lazy Test
/// Basic lazy value creation and forcing
module LazyMinimalSample

[<EntryPoint>]
let main _ =
    // Simplest lazy value - wraps a literal
    let lazyFortyTwo = lazy 42

    // Force the lazy value
    let result = Lazy.force lazyFortyTwo

    // Output result
    Console.write "Lazy result: "
    Console.writeln (Format.int result)

    0
