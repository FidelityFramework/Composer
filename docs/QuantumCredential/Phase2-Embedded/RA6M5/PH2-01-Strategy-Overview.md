# PH2-01: RA6M5 Strategy Overview

## Purpose

This document defines the active embedded strategy for QuantumCredential on the Renesas EK-RA6M5. It assumes the binding generator path is now usable and that the next question is how to shape the RA6M5 workload pipeline.

## Why RA6M5

The RA6M5 is the preferred production target because it gives the project a stronger root-of-trust story than the earlier STM32L5 demo board.

- Hardware unique key support anchors device identity in silicon.
- TrustZone separates secure and non-secure workloads.
- Secure crypto hardware reduces the amount of sensitive software we need to carry.
- The board has enough headroom to support sample acquisition, crypto, and workload orchestration together.

## Architecture Posture

The RA6M5 path should be treated as a board-specific embedded platform.

The practical layering is:

1. Renesas FSP and board support code provide the operational surface.
2. Farscape generates the hardware-facing bindings as a distinct library building step.

3. Clef source and CCS compile the workload.
4. Composer lowers the workload to native code.

- FreeRTOS is optional and only used if it speeds up bootstrap. 

## Planning Goal

The goal is to get real workloads onto the EK-RA6M5 with as little translation loss as possible.

That means:

- preserve the sample-pipeline semantics in software
- expose the device security features as first-class APIs
- keep the binding layer explicit and auditable
- keep the design aligned with a clean hardware abstraction

## Immediate Questions

- Which portions of the workload need direct hardware access on day one?
- Which parts are shared and must have transitive connection between binding layers?
- What should be modeled as long-lived bindings versus use case/application specific code?

The remaining documents in this directory answer those questions in more detail.
