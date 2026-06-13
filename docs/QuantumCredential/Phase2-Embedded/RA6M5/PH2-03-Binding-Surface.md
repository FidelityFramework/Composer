# PH2-03: RA6M5 Binding Surface

## Purpose

This document defines the first hardware-facing surface for the EK-RA6M5 embedded track.

The RA6M5 does not ask us to target an STM32-style CMSIS HAL. The better fit is a binding layer centered on Renesas FSP, board support code, and the small set of direct hardware operations the workload really needs.

## What the Surface Must Expose

The first binding set should cover:

- board and clock initialization
- GPIO configuration and control
- ADC setup, channel selection, and scan triggering
- timer and delay primitives
- USB and serial I/O
- flash and secure storage operations
- TrustZone and secure-world handoff points

## Entropy-Specific Bindings

The entropy path needs special attention.

The generated bindings should make the following operations explicit:

- select entropy mux state
- trigger an ADC conversion
- read the sampled value
- repeat across both ADC channels
- surface timing and calibration data back to Clef

That is the minimal surface needed to make the quad-entropy circuit intelligible to the compiler and to the runtime.

## Binding Shape

The generated Clef side should stay close to the naming and dependency model used elsewhere in Composer:

- `Platform.Bindings.Renesas.*` for board-facing APIs
- thin wrapper modules for FSP-backed operations
- explicit types for register-like values and device handles
- no hidden dependence on a large runtime just to reach a peripheral

## What It Should Not Be

The binding layer should not become a second operating system.

It should not hide:

- ADC sequencing
- mux control
- security partition boundaries
- the distinction between hardware access and scheduler policy

If FreeRTOS is present, it should sit underneath or beside these bindings, not redefine them.
