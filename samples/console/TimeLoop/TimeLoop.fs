/// TimeLoop - Demonstrates DateTime intrinsics with millisecond precision
/// Shows current time as HH:MM:SS.mmm with delta from previous emission
module TimeLoop

/// Format a 2-digit number with leading zero if needed
let padTwo (n: int) : string =
    if n < 10 then
        String.concat2 "0" (Format.int n)
    else
        Format.int n

/// Format a 3-digit number with leading zeros if needed
let padThree (n: int) : string =
    if n < 10 then
        String.concat2 "00" (Format.int n)
    elif n < 100 then
        String.concat2 "0" (Format.int n)
    else
        Format.int n

/// Format milliseconds since epoch as HH:MM:SS.mmm (local time)
/// Uses DateTime intrinsics for component extraction
let formatTime (ms: int64) : unit =
    // Convert UTC to local time using platform timezone
    let localMs = DateTime.toLocal ms

    // Extract time components using DateTime intrinsics
    let hours = DateTime.hour localMs
    let minutes = DateTime.minute localMs
    let seconds = DateTime.second localMs
    let millis = DateTime.millisecond localMs

    // Print formatted time
    Console.write (padTwo hours)
    Console.write ":"
    Console.write (padTwo minutes)
    Console.write ":"
    Console.write (padTwo seconds)
    Console.write "."
    Console.write (padThree millis)

/// Display current time in a loop with delta tracking
/// Goal: Print the current time 5 times, once per second, showing ms delta
let displayTimeLoop (iterations: int) =
    Console.writeln "TimeLoop - DateTime Intrinsics Demo"
    Console.writeln ""

    // Show timezone offset for reference
    let tzOffsetSec = DateTime.utcOffset ()
    let tzOffsetHours = tzOffsetSec / 3600
    Console.write "Timezone offset: "
    Console.write (Format.int tzOffsetHours)
    Console.writeln " hours from UTC"
    Console.writeln ""

    // Get initial time using DateTime.now()
    let mutable lastTime = DateTime.now ()
    Console.write "Start: "
    formatTime lastTime
    Console.writeln " (local)"
    Console.writeln ""

    let mutable counter = 0
    while counter < iterations do
        // Sleep for 1 second (1,000,000,000 nanoseconds)
        Sys.nanosleep 1000000000

        // Get current time using DateTime.now()
        let currentTime = DateTime.now ()
        let delta = currentTime - lastTime

        // Format and display
        Console.write "Time: "
        formatTime currentTime
        Console.write " (local)  (delta: "
        Console.write (Format.int64 delta)
        Console.writeln " ms)"

        lastTime <- currentTime
        counter <- counter + 1

    Console.writeln ""
    Console.writeln "Done."

/// Entry point
[<EntryPoint>]
let main argv =
    let iterations = 5
    displayTimeLoop iterations
    0
