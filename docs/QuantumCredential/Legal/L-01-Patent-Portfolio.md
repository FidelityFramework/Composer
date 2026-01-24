# L-01: Patent Portfolio

## Overview

QuantumCredential implements technology protected by pending patents from SpeakEZ Technologies. This document summarizes the patent portfolio and its relationship to the system architecture.

---

## Pending Applications

### US 63/780,027

**Title**: Air-Gapped Dual Network Architecture for QRNG Cryptographic Certificate Distribution via QR Code and Infrared Transfer in WireGuard Overlay Networks

**Coverage**:
- Quantum entropy harvesting from avalanche noise circuits
- Air-gapped credential generation (no network during entropy collection)
- Out-of-band credential distribution via QR code
- Infrared transfer as alternative channel
- WireGuard overlay network integration

**Implementation in QuantumCredential**:
- Four-channel avalanche circuit provides quantum entropy source
- YoshiPi operates offline during entropy generation
- QR code display for credential transfer to KeyStation
- Future: IR LED/receiver for alternative transfer path

### US 63/780,055

**Title**: Quantum-Resistant Hardware Security Module with Decentralized Identity Capabilities

**Coverage**:
- Hardware security module (HSM) architecture
- Post-quantum cryptographic operations (ML-KEM, ML-DSA)
- SHAKE-256 entropy conditioning
- Decentralized identity (DID) integration
- Self-sovereign credential model

**Implementation in QuantumCredential**:
- Hardware-anchored entropy generation
- ML-KEM for key encapsulation (future: credential encryption)
- ML-DSA for digital signatures (future: credential signing)
- SHAKE-256 used for entropy conditioning
- DID-compatible credential format (future)

### US 63/786,247

**Title**: System and Method for Zero-Copy Inter-Process Communication Using BARE Protocol

**Coverage**:
- BAREWire serialization protocol
- Zero-copy binary communication
- Type-safe memory management
- Cross-platform wire format

**Implementation in QuantumCredential**:
- Credential serialization using BAREWire format
- Zero-copy transfer between components
- Consistent format across YoshiPi, KeyStation, and desktop

---

## Relationship to Architecture

The patents protect the complete system architecture, not just individual components:

```
┌─────────────────────────────────────────────────────────┐
│                    US 63/780,027                        │
│                    Air-Gapped QRNG                      │
│  ┌──────────────┐      ┌───────────────┐               │
│  │ Avalanche    │      │ QR/IR         │               │
│  │ Entropy      │──────│ Distribution  │               │
│  │ (Offline)    │      │ (Air-Gapped)  │               │
│  └──────────────┘      └───────────────┘               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    US 63/780,055                        │
│                    Quantum-Resistant HSM                │
│  ┌──────────────┐      ┌───────────────┐               │
│  │ Post-Quantum │      │ Decentralized │               │
│  │ Crypto       │──────│ Identity      │               │
│  │ (ML-KEM/DSA) │      │ (DID/SSI)     │               │
│  └──────────────┘      └───────────────┘               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    US 63/786,247                        │
│                    BAREWire Protocol                    │
│  ┌──────────────┐      ┌───────────────┐               │
│  │ Zero-Copy    │      │ Type-Safe     │               │
│  │ Serialization│──────│ Wire Format   │               │
│  └──────────────┘      └───────────────┘               │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Status

| Patent | Core Claims | Demo Implementation |
|--------|------------|---------------------|
| 63/780,027 | Avalanche QRNG | Four-channel circuit complete |
| 63/780,027 | Air-gapped generation | YoshiPi offline operation |
| 63/780,027 | QR distribution | Planned for demo |
| 63/780,027 | IR transfer | Future enhancement |
| 63/780,055 | HSM architecture | Software implementation |
| 63/780,055 | ML-KEM | Integration planned |
| 63/780,055 | ML-DSA | Integration planned |
| 63/780,055 | DID support | Future enhancement |
| 63/786,247 | BAREWire | Integration planned |

---

## Confidentiality Note

Patent applications are public upon publication. Implementation details beyond what is disclosed in the applications may constitute trade secrets and should be treated with appropriate confidentiality.

---

## Connection to Other Documents

- **H-02-Avalanche-Circuit**: Implements 63/780,027 entropy generation
- **V-05-PostQuantum-Architecture**: Implements 63/780,055 crypto
- **D-01-Demo-Strategy**: Demonstrates patent claims
