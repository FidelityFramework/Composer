# PH2-04: RA6M5 Bootstrap Options

## Two Valid Paths

There are two reasonable ways to start bringing workloads onto the EK-RA6M5.

### Option A: Direct Hardware Bring-Up

This is the cleaner long-term option.

Use Renesas FSP plus the generated bindings to talk to the board directly, and keep the runtime as small as possible.

Benefits:

- fewer moving parts
- clearer security story
- less translation between the workload and the board
- easier to evolve toward a full unikernel

Costs:

- more explicit work up front
- more responsibility for startup, I/O, and scheduling decisions

### Option B: FreeRTOS Bootstrap

This is the pragmatic early-bring-up option.

Use FreeRTOS to get tasking, coordination, and peripheral timing under control before the direct hardware path is fully settled.

Benefits:

- quicker first workload partitioning
- easier coordination of multiple concurrent activities
- familiar embedded development model

Costs:

- more runtime surface area
- more abstraction between the workload and the hardware
- more work to unwind later if we want a true unikernel

## Recommendation

Treat FreeRTOS as a bootstrap helper, not as the architectural destination.

If the direct hardware path is already clear enough to support the entropy circuit and the device security functions, prefer that path.

If the team needs a temporary scheduler and a simpler integration ramp, use FreeRTOS narrowly and plan the exit early.

## Decision Rule

Choose FreeRTOS only if it reduces time to a real, testable workload on the board.

Choose direct hardware if the extra runtime layer would only delay the binding work we already know we need.
