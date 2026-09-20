# QuantumCredential

QuantumCredential explores hardware entropy acquisition, credential generation
and transfer to a KeyStation management interface. Each hardware host needs an
accepted image and device implementation. The design documents describe the
intended system and its experiments, rather than establish end-to-end security
or deployment acceptance.

## Hardware and execution

The YoshiPi work uses a Raspberry Pi Zero 2 W carrier with ADC and touchscreen
connections. Its [hardware design](Hardware/H-01-YoshiPi-Platform.md) and
[Linux binding plan](Phase1-YoshiPi/PH1-02-Linux-Hardware-Bindings.md) cover that
host. ADC acquisition is device I/O. Independent channels alone do not establish
simultaneous sampling, statistical independence or a mapping to CPU cores.

KeyStation uses Libre Computer AML-S905X-CC-V2 with Amlogic S905X and DDR4. The
selected display is Waveshare's 7.9inch HDMI LCD, SKU 17916, ASIN B087CNJYB4.
Its primary execution objective is a Clef unikernel. Linux AArch64 provides a
separate hosted route and hardware reference. The
[board definition](../../../Fidelity.Platform/Hardware/Products/LibreComputer/AML_S905X_CC_V2/README.md)
and [native port plan](../../../Fidelity.Platform/docs/SWEET_POTATO_UI_PORT.md)
own the hardware facts and implementation sequence.

The [EK-RA6M5 product package](../../../Fidelity.Platform/Hardware/Products/Renesas/EK_RA6M5/README.md)
and [HelloBlinky profile](../../../Fidelity.Platform/Profiles/EK_RA6M5_HelloBlinky/)
provide the current MCU composition example. The
[STM32L5 and KeyStation comparison](Phase2-Embedded/STM32L5/PH2-02-Hardware-Platforms.md)
records the STM32L552 board facts and its remaining port requirements. MCU
credential processing and root-of-trust integration have separate acceptance
criteria from the KeyStation display.

## Framework integration

[Fidelity.Platform](../../../Fidelity.Platform/README.md) separates silicon facts,
board wiring, execution environments and application profiles. The
[device-access contracts](../../../Fidelity.Platform/docs/MMIO_CONTRACTS.md)
connect selected registers, mappings and workload grants to CCS checking and
Composer lowering. An available peripheral does not grant an application access
or supply its driver.

[Ariel](../../../Fidelity.Platform/Environments/Linux/x86_64/Ariel/README.md)
supplies the scheduling layer. The documented Linux implementation and its
carrier lifetime contracts provide a reference for the native environment work.
A unikernel can include scheduling and independently owned services.

[Farscape](../../../Farscape/README.md) generates bindings to C interfaces. The
corresponding implementation and its ABI remain dependencies. Native Clef
peripheral implementations use the selected platform's device-access contract.
Compiler behavior should follow declared operations and evidence, without
recognizing an application function name as an implicit driver.

[BAREWire](../../../BAREWire/README.md) supplies memory-layout and binary-data
facilities. The credential protocol must also define ownership, framing and
transfer outcomes. Shared storage and allocation costs depend on the selected
implementation and transport.

[Fidelity.UI](../../../Fidelity.UI/README.md) supplies the proposed shared
component semantics. KeyStation's native renderer and WREN's DOM/WebView route
must each implement the admitted behavior. Cold construction, owned demand and
view disposal apply independently of service lifetime.

## Design documents

| Document | Scope |
| --- | --- |
| [Demo strategy](Demo/D-01-Demo-Strategy.md) | YoshiPi and desktop Linux demonstration design |
| [YoshiPi hardware](Hardware/H-01-YoshiPi-Platform.md) | Carrier, ADC and peripheral integration |
| [Avalanche circuit](Hardware/H-02-Avalanche-Circuit.md) | Analog source design |
| [Parallel compilation](Compilation/C-01-SCF-Parallel-Pattern.md) | Proposed parallel processing of acquired samples |
| [Zero-copy pipeline](Compilation/C-02-Zero-Copy-Pipeline.md) | Buffer and transfer design |
| [Linux hardware bindings](Phase1-YoshiPi/PH1-02-Linux-Hardware-Bindings.md) | Hosted device-access work |
| [Post-quantum architecture](Validation/V-05-PostQuantum-Architecture.md) | Credential and cryptographic design |
| [Demo roadmap](06_January_Roadmap.md) | Delivery sequence and dependencies |
| [UI and transport experiments](07_Stretch_Goals.md) | Interaction, visualization and transfer options |
| [Embedded strategy](Phase2-Embedded/PH2-00-Embedded-Strategy.md) | MCU development plan |
| [Renesas work](Phase2-Embedded/RA6M5/README.md) | RA6M5 application planning |
| [Hardware comparison](Phase2-Embedded/STM32L5/PH2-02-Hardware-Platforms.md) | STM32L552 reference and S905X host requirements |
| [Document index](INDEX.md) | Remaining research and experiments |

## Demonstration evidence

A completed demonstration needs recorded results for acquisition and health
checks, credential operations, transport and receiving-side verification. Record
the hardware, firmware, compiler and library versions with those results.
Compiler output or a UI screenshot alone cannot establish entropy quality or
security of the credential system.

For native KeyStation graphics, follow the
[platform acceptance ladder](../../../Fidelity.Platform/docs/SWEET_POTATO_UI_PORT.md#acceptance-ladder).
It separates boot, display/input and GPU execution from the credential service.
The [patent portfolio document](Legal/L-01-Patent-Portfolio.md) owns the project's
intellectual-property references.
