# PH2-01: RA6M5 Strategy Overview

## Purpose

This document defines the active embedded strategy for QuantumCredential on the Renesas EK-RA6M5. It assumes the binding generator path is now usable and that the next question is how to shape the RA6M5 workload pipeline.

## Why RA6M5

The RA6M5 is the preferred production target because it gives the project a stronger root-of-trust story than the earlier STM32L5 demo board.

- Hardware unique key support anchors device identity in silicon.
- TrustZone separates secure and non-secure workloads.
- Secure crypto hardware reduces the amount of sensitive software we need to carry.
- The board has enough headroom to support entropy acquisition, crypto, and workload orchestration together.

## Architecture Posture

The RA6M5 path should be treated as a board-specific embedded platform, not as a generic CMSIS exercise.

The practical layering is:

1. Clef source and CCS compile the workload.
2. Composer lowers the workload to native code.
3. Farscape generates the hardware-facing bindings.
4. Renesas FSP and board support code provide the operational surface.
5. FreeRTOS is optional and only used if it speeds up bootstrap.

## Planning Goal

The goal is to get real workloads onto the EK-RA6M5 with as little translation loss as possible.

That means:

- preserve the entropy circuit semantics in software
- expose the device security features as first-class APIs
- keep the binding layer explicit and auditable
- avoid forcing the design through an STM32L5-shaped abstraction

## Immediate Questions

- Which portions of the workload need direct hardware access on day one?
- Which parts can sit on top of a minimal runtime layer?
- What should be modeled as bindings versus compiler intrinsics?

The remaining documents in this directory answer those questions in more detail.
