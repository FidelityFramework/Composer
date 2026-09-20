# QuantumCredential interaction and transport experiments

These experiments extend the credential workflow with local touch interaction,
activity visualization and additional transfer methods. Each experiment needs
its own device integration and recorded result. The
[project overview](README.md) identifies the current framework and hardware
references.

## Hardware roles

YoshiPi and KeyStation can share application behavior while using different
boards and device drivers. YoshiPi has carrier ADC connections. KeyStation uses
Libre Computer AML-S905X-CC-V2 with the selected Waveshare HDMI/USB display.
An external ADC, avalanche source, camera or infrared transceiver is an
additional hardware selection for KeyStation.

Bidirectional credential exchange requires supported transmit and receive paths
on both devices. A common protocol can preserve message meaning across those
paths. Matching application roles do not establish matching circuit boards,
analog front ends or peripheral wiring.

## Authority-management interface

The UI can expose credential inspection, transfer status and authorized signing
operations through the service contract. Root-key generation and certificate
issuance depend on the selected authority design, provisioning and key-storage
implementation. Attaching an entropy circuit alone does not establish a usable
certificate authority.

Keep the KeyStation panel's interaction model separate from root-of-trust and
Renesas HUK integration. A synthetic service can exercise the UI while those
systems are developed. The
[cryptographic design](Validation/V-05-PostQuantum-Architecture.md) holds the
credential-processing research.

## Stretch Goal 1: Sweet Potato KeyStation

The selected board is Libre Computer **AML-S905X-CC-V2**, using the Amlogic
S905X, four Cortex-A53 cores, Mali-450 graphics and DDR4. The
[Fidelity.Platform board entry](../../../Fidelity.Platform/Hardware/Products/LibreComputer/AML_S905X_CC_V2/README.md)
owns the product definition and its primary sources.

The primary target is a Clef unikernel. A Linux AArch64 application remains a
separate deployment option and a hardware reference. The
[native port plan](../../../Fidelity.Platform/docs/SWEET_POTATO_UI_PORT.md)
assigns boot, Meson HDMI, USB touch and Mali driver work. An AArch64 target
triple alone does not provide the ABI, device drivers or display integration.

### Rack display and touch

KeyStation uses the Waveshare 7.9inch HDMI LCD, SKU 17916 (ASIN B087CNJYB4),
mounted horizontally in a 19-inch rack panel. The
[panel entry](../../../Fidelity.Platform/Hardware/Products/Waveshare/7_9inch_HDMI_LCD/README.md)
records its identifiers and 400 × 1280 physical matrix. HDMI carries the
image and USB carries touch. Physical scanout orientation,
logical layout and input coordinates must use compatible transforms.

### Shared UI behavior

[Fidelity.UI's KeyStation design](../../../Fidelity.UI/docs/09_sweet_potato_keystation.md)
uses cold functional components and owned reactive areas. The native path starts
with CPU rendering and adds restricted Mali acceleration. The hosted Linux path
can use Mesa and DRM/KMS or Wayland. WREN remains a browser/WebView realization
of the admitted component semantics.

Credential lists, selection, command status and activity history can share
behavior across those hosts. Each renderer must establish layout, text, input
and disposal behavior. A responsive CSS layout alone does not implement the
native surface or its resource ownership.

---

## Touch interaction

Controls need stable identity, focus, hit testing and consistent press/release
behavior. The selected Waveshare panel reports touch through USB. The native
S905X path must implement host enumeration and report handling, then transform
physical coordinates into the horizontal logical layout.

The Linux route obtains events through its selected input interface. A WebView
can handle browser events once the host delivers input to it. Neither route
establishes native touch support merely by constructing a component.

Exercise list selection, scrolling, editing and confirmation with the same
application actions on native and WREN hosts. Define gesture cancellation and
input-device disconnect behavior. A hardware report can contain multiple
contacts even when the first application only admits one active gesture.

## Entropy and activity visualization

The service owns acquisition, health checks and any retained sample history.
A visual area observes a bounded projection. Closing the plot releases visual
demand without implicitly stopping the acquisition service.

Use separate representations for latest status, ordered events and sampled
history. Limit plot updates to a useful display rate and keep acquisition timing
independent of painting. A waveform represents observed samples. Entropy quality
requires the selected validation method and its recorded evidence.

The native renderer can cache geometry or pixels within its budget. Browser
realization can use DOM or canvas facilities that preserve the admitted behavior.
The shared contract concerns state, events and lifetime, with rendering provided
by each backend.

## QR transfer

A QR route needs encoding, camera acquisition and decoding on the selected hosts.
Define payload size, framing and incomplete-transfer handling before displaying
multi-frame sequences. Record whether a transfer is pending, complete or rejected.
Credential verification belongs to the receiving service after reconstruction.

A USB camera on KeyStation adds device, power and bandwidth requirements. Its
support is separate from the accepted HDMI and touch path. UI responsiveness
must be measured while acquisition and decoding are active.

## Infrared transfer

Select the emitter, receiver and board connections before defining a driver.
Carrier generation and receive timing need device-specific timer, GPIO or other
peripheral support. The protocol needs framing, ordering, integrity checks and
a recovery policy for interrupted transfers.

A Clef-native driver owns those operations through the platform's MMIO contract.
A hosted implementation uses its selected Linux interfaces. Farscape can bind an
existing implementation where appropriate. Function declarations alone do not
provide infrared transmission or reception.

## Implementation sequence

| Step | Evidence needed |
| --- | --- |
| Service projection | Bounded snapshots, ordered command results and explicit ownership |
| Local controls | Selection, editing, focus and disposal with synthetic state |
| Native panel | Accepted S905X mode, render/input rotation and USB touch reports |
| Activity display | Bounded history, selective redraw and measured acquisition interference |
| USB credential transfer | Framing, receiving-side verification and failure reporting |
| QR or infrared extension | Selected hardware, protocol implementation and interrupted-transfer recovery |

Use the [native graphics acceptance ladder](../../../Fidelity.Platform/docs/SWEET_POTATO_UI_PORT.md#acceptance-ladder)
for KeyStation display work. Admission of fades and eased transitions follows
measured rendering and resource budgets. Live status updates remain application
behavior even when decorative motion is omitted.

## Demonstration flow

A completed demonstration should show acquisition status, credential generation,
transfer and receiving-side verification. Display the actual outcome of each
operation, including rejected or incomplete transfers. Keep the hardware,
firmware, compiler and library versions with the evidence.

USB transfer to a desktop client is a bounded first integration. QR scanning and
infrared transport add their own hardware, framing and recovery work. A native
Sweet Potato panel additionally depends on the S905X boot, display and input
implementation.

## Cross-references

- [Demo strategy](Demo/D-01-Demo-Strategy.md): YoshiPi and desktop Linux experiment.
- [YoshiPi hardware](Hardware/H-01-YoshiPi-Platform.md): Carrier and peripheral design.
- [Parallel compilation](Compilation/C-01-SCF-Parallel-Pattern.md): Sample-processing design.
- [Linux hardware bindings](Phase1-YoshiPi/PH1-02-Linux-Hardware-Bindings.md): Hosted device-access plan.
- [Post-quantum architecture](Validation/V-05-PostQuantum-Architecture.md): Credential-processing research.
