# PH2-02: RA6M5 Hardware Platform

## Board Summary

The EK-RA6M5 gives the embedded plan a production-oriented microcontroller target with a stronger security posture than the older STM32L5 demo board.

## Core Capabilities

- Arm Cortex-M33 execution core
- TrustZone secure/non-secure partitioning
- Hardware unique key for device identity
- Secure crypto engine for protected key operations
- 2 MB flash and 512 KB SRAM
- USB, Ethernet, CAN FD, UART, SPI, and I2C support

## Sample Front End Hardware

The most important hardware feature for QuantumCredential is the analog sample path.

The quad-sample circuit uses two ADC channels with a mux shift so the software can sample all four analog lines through a controlled scan sequence.

That has a few implications:

- sampling order matters
- channel switching latency matters
- per-line calibration needs to be explicit
- the binding layer must expose enough control to keep acquisition deterministic

## Workload Implications

The sample pipeline is part of the kernel story because it feeds the credential workload itself.

The software has to:

- select the active mux state
- sample both ADC channels in a known sequence
- normalize the per-line measurements
- preserve timing and bias data for later combination

That makes the hardware platform doc inseparable from the binding surface doc.

## Design Consequence

The EK-RA6M5 is a device where the security boundary, sample acquisition, and workload execution model all need to be documented together.
