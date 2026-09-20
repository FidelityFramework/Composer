# KeyStation and desktop UI realization

KeyStation uses Libre Computer AML-S905X-CC-V2 (Sweet Potato), with a Clef
unikernel as the primary deployment objective and Linux AArch64 as a hosted
option. The [board entry](../../../../../Fidelity.Platform/Hardware/Products/LibreComputer/AML_S905X_CC_V2/README.md)
owns product facts and display selection. The
[native port plan](../../../../../Fidelity.Platform/docs/SWEET_POTATO_UI_PORT.md)
owns boot and driver requirements.

## Shared components

[Fidelity.UI](../../../../../Fidelity.UI/README.md) defines cold functional
components, owned activation and demand-driven reactive areas. Optional
computation expressions use the same semantic operations. Application actions,
selection state and service projections can be shared between the KeyStation
panel, a desktop host and a WREN browser/WebView realization.

Shared semantics require implementations of layout, text, focus, input and
resource lifetime on each host. LVGL and Solid are research references for
components and rendering behavior. They are not mandatory runtimes for the
Clef-native engine.

## KeyStation rendering

The [KeyStation graphics design](../../../../../Fidelity.UI/docs/09_sweet_potato_keystation.md)
selects CPU reference rendering, native Meson HDMI and USB touch, then restricted
Mali-450 drawing. Fades and eased transitions acquire clock demand while active.
A hidden panel can release visual demand while a separately owned service keeps
its data projection current. Retained layers and prewarming need explicit
budgets.

The selected display is the Waveshare 7.9inch HDMI LCD, SKU 17916
(ASIN B087CNJYB4), in a horizontal 19-inch rack assembly. Follow the
[panel entry](../../../../../Fidelity.Platform/Hardware/Products/Waveshare/7_9inch_HDMI_LCD/README.md)
for its 400 × 1280 physical matrix and 1280 × 400 logical layout. Match the touch
transform to the render transform. No framebuffer command or input-device path
is selected before the actual mode and USB reports are established.

## Hosted Linux and desktop

The Sweet Potato Linux route uses the appropriate AArch64 ABI and userspace
bindings, with Mesa Lima and Meson DRM/KMS or a Wayland host. Desktop backends
supply their corresponding presentation and input interfaces. The
[rendering model](../../../../../Fidelity.UI/docs/03_rendering_backends.md)
separates those facilities from component semantics.

WREN realizes admitted components through the DOM/WebView. The browser owns
layout and painting for that surface. A native display renderer consumes the
native layout and paint output. Arbitrary CSS and GPU commands are outside the
portable component contract.

## Service integration and acceptance

The first panel can display synthetic device status, a selectable list, an editor
and activity history. Preserve ordered command results separately from replaceable
telemetry snapshots. UI disposal must release observations without implicitly
stopping the service being viewed.

Test text resizing, moved or removed translucent content, focus, interrupted
transitions and repeated view retirement. Compare partial updates with a full
redraw and retain per-buffer scene validity. Submitted GPU resources stay alive
through rendering completion and display release.

Hardware acceptance follows the
[platform ladder](../../../../../Fidelity.Platform/docs/SWEET_POTATO_UI_PORT.md#acceptance-ladder).
Root-of-trust, credential transport and storage encryption remain separate
integrations. The UI port does not establish their security properties.
