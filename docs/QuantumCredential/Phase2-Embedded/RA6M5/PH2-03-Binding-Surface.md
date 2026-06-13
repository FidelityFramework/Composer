# PH2-03: RA6M5 Binding Surface

## Purpose

This document defines the first hardware-facing surface for the EK-RA6M5 embedded track.

The RA6M5 binding surface is centered on Renesas FSP, board support code, and the small set of direct hardware operations the workload needs.

## What the Surface Must Expose

The first binding set should cover:

- board and clock initialization
- GPIO configuration and control
- ADC setup, channel selection, and scan triggering
- timer and delay primitives
- USB and serial I/O
- flash and secure storage operations
- TrustZone and secure-world handoff points

## Sample-Pipeline Bindings

The raw sample path needs special attention.

The generated bindings should make the following operations explicit:

- select mux state
- trigger an ADC conversion
- read the sampled value
- repeat across both ADC channels
- surface timing and calibration data back to Clef

That is the minimal surface needed to make the quad-sample front end intelligible to the compiler and to the runtime.

## Binding Shape

The generated Clef side should stay close to the naming and dependency model used elsewhere in Composer:

- `Platform.Bindings.Renesas.*` for board-facing APIs
- thin wrapper modules for FSP-backed operations
- explicit types for register-like values and device handles
- no hidden dependence on a large runtime just to reach a peripheral

## Binding Focus

Keep the binding layer focused on explicit hardware access:

- ADC sequencing
- mux control
- security partition boundaries
- the distinction between hardware access and scheduler policy

If FreeRTOS is present, it serves as a bootstrap helper beside these bindings.

## Taxonomy Boundary

The binding tree should follow a thin-branch / leaf split:

- family-wide Renesas contracts live in the branch layer
- EK-RA6M5 pin maps and quirks live in the leaf layer
- sample-pipeline helpers live above both as workload adapters

See [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) for the concrete work breakdown.
