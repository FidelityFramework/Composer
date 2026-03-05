# D-01: GTK Window (FFI via Farscape-Generated Bindings)

> **Sample**: `25_GTKWindow` | **Status**: Planned | **Depends On**: C-01, Farscape binding generation

## 1. Executive Summary

This PRD validates Fidelity's ability to consume **Farscape-generated binding libraries** to call native C APIs — specifically GTK4 for desktop GUI applications. No FFI declarations appear in application code. The application imports generated `Fidelity.Gtk` libraries and calls functions directly.

**Key Insight**: GTK uses a C API with callback registration via function pointers and listener structs. Farscape's Layer 2 callback builder generation (Phase 1.6) handles both patterns. The application developer never writes `[<FidelityExtern>]` declarations — Farscape generates them from a pilot TOML project.

**Architecture**: This PRD exercises the **ExternCall pathway** end-to-end:
1. Farscape parses GTK4 headers → generates `Fidelity.Gtk` binding library (L1 + L2)
2. Application imports the generated library via fidproj dependency
3. CCS type-checks calls against `[<FidelityExtern>]` declarations
4. Baker saturates (no special handling needed)
5. `PlatformBindingResolution` resolves `[<FidelityExtern>]` metadata → `ExternCall` coeffect
6. Alex witnesses `ExternCall` via `pExternCallResolved` → MLIR `func.call`

## 2. Binding Generation (Farscape)

### 2.1 Pilot TOML

```toml
[library]
name = "gtk-4"
headers = ["/usr/include/gtk-4.0/gtk/gtk.h"]
include_paths = [
    "/usr/include/gtk-4.0",
    "/usr/include/glib-2.0",
    "/usr/lib/glib-2.0/include",
    "/usr/include/pango-1.0",
    "/usr/include/cairo",
    "/usr/include/gdk-pixbuf-2.0"
]
macro_prefixes = ["GTK_", "GDK_"]

[output]
mode = "fidelity"
directory = "../Bindings/Gtk"

[options]
opaque_handles = true

[callbacks]
# Auto-discovered from headers

[[namespace]]
name = "Fidelity.Gtk.Window"
description = "Window creation and management"
library = "gtk-4"
prefixes = ["gtk_window", "gtk_widget"]
functions = ["gtk_init"]

[[namespace]]
name = "Fidelity.Gtk.Application"
description = "GtkApplication lifecycle"
library = "gtk-4"
prefixes = ["gtk_application"]
```

GObject signal connection functions live in a separate library:

```toml
# gobject.pilot.toml
[library]
name = "gobject-2.0"
headers = ["/usr/include/glib-2.0/gobject/gsignal.h"]
include_paths = ["/usr/include/glib-2.0", "/usr/lib/glib-2.0/include"]

[output]
mode = "fidelity"
directory = "../Bindings/GObject"

[[namespace]]
name = "Fidelity.GObject.Signal"
description = "GObject signal connection"
library = "gobject-2.0"
prefixes = ["g_signal"]
```

### 2.2 Generated Output Structure

```
Bindings/Gtk/
├── Types.clef                  # Opaque handle types (GtkWidget, GtkWindow)
├── Window/
│   ├── Types.clef              # Namespace-local types
│   ├── Window.clef             # L1: [<FidelityExtern>] declarations
│   └── WindowApi.clef          # L2: idiomatic wrappers
├── Application/
│   ├── Application.clef        # L1
│   └── ApplicationApi.clef     # L2
└── Callbacks.clef              # L2: listener builders

Bindings/GObject/
├── Signal/
│   ├── Signal.clef             # L1: g_signal_connect_data
│   └── SignalApi.clef          # L2: safe signal connection
```

### 2.3 Generated L1 Declarations (Window.clef)

Farscape generates these — the application developer never writes them:

```clef
module Fidelity.Gtk.Window

[<FidelityExtern("gtk-4", "gtk_init")>]
let gtk_init () : unit = Unchecked.defaultof<unit>

[<FidelityExtern("gtk-4", "gtk_window_new")>]
let gtk_window_new () : nativeint = Unchecked.defaultof<nativeint>

[<FidelityExtern("gtk-4", "gtk_window_set_title")>]
let gtk_window_set_title (window: nativeint) (title: nativeint) : unit = Unchecked.defaultof<unit>

[<FidelityExtern("gtk-4", "gtk_window_set_default_size")>]
let gtk_window_set_default_size (window: nativeint) (width: int) (height: int) : unit = Unchecked.defaultof<unit>

[<FidelityExtern("gtk-4", "gtk_widget_show")>]
let gtk_widget_show (widget: nativeint) : unit = Unchecked.defaultof<unit>

[<FidelityExtern("gtk-4", "gtk_main")>]
let gtk_main () : unit = Unchecked.defaultof<unit>

[<FidelityExtern("gtk-4", "gtk_main_quit")>]
let gtk_main_quit () : unit = Unchecked.defaultof<unit>
```

### 2.4 Generated L2 Wrappers (WindowApi.clef)

```clef
module Fidelity.Gtk.Window.Api

open Fidelity.Gtk.Window

/// Initialize GTK. Must be called before any other GTK function.
let init () : unit =
    gtk_init ()

/// Create a new top-level window. Returns None if creation fails.
let windowNew () : option<nativeint> =
    let result = gtk_window_new ()
    if result = 0n then None
    else Some result

/// Set the window title.
let setTitle (window: nativeint) (title: nativeint) : unit =
    gtk_window_set_title window title

/// Set the default window size in pixels.
let setDefaultSize (window: nativeint) (width: int) (height: int) : unit =
    gtk_window_set_default_size window width height
```

### 2.5 Generated fidproj

```toml
[package]
name = "Fidelity.Gtk"
version = "0.1.0"
description = "Generated bindings for gtk-4."

[compilation]
target = "cpu"

[build]
output_kind = "library"
sources = [
    "Bindings/Gtk/Types.clef",
    "Bindings/Gtk/Window/Types.clef",
    "Bindings/Gtk/Window/Window.clef",
    "Bindings/Gtk/Window/WindowApi.clef",
    "Bindings/Gtk/Application/Application.clef",
    "Bindings/Gtk/Application/ApplicationApi.clef",
    "Bindings/Gtk/Callbacks.clef",
]

[platform]
runtime_model = "libc"
os = "linux"
arch = "x86_64"
word_size = 64

[dependencies]
Fidelity.Platform = { path = "Fidelity.Platform.fidproj" }
```

## 3. Application Code

### 3.1 Project Structure

Following the HelloArty pattern — thin `Program.clef` + behavior module:

```
HelloGtk/
├── HelloGtk.fidproj
└── src/
    ├── Program.clef            # Entry point — thin composition
    └── GtkApp.clef             # Application behavior
```

### 3.2 HelloGtk.fidproj

```toml
[package]
name = "HelloGtk"
version = "0.1.0"
description = "Clef Language + Fidelity Framework — Native GTK4 Window"

[compilation]
target = "cpu"

[dependencies]
platform = { path = "../Fidelity.Platform/CPU/Linux/x86_64/Fidelity.Platform.fidproj" }
gtk = { path = "../Fidelity.Platform/CPU/Linux/x86_64/Fidelity.Gtk.fidproj" }
gobject = { path = "../Fidelity.Platform/CPU/Linux/x86_64/Fidelity.GObject.fidproj" }

[build]
sources = ["src/GtkApp.clef", "src/Program.clef"]
output = "HelloGtk"
output_kind = "console"

[link]
libraries = ["gtk-4", "gobject-2.0"]
```

### 3.3 GtkApp.clef (Behavior)

```clef
module HelloGtk.GtkApp

open Fidelity.Gtk.Window.Api
open Fidelity.GObject.Signal.Api

/// Callback for window destroy signal.
let onDestroy (_widget: nativeint) (_data: nativeint) : unit =
    Fidelity.Gtk.Window.gtk_main_quit ()

/// Create and configure the application window.
let createWindow (title: nativeint) (width: int) (height: int) : option<nativeint> =
    match windowNew () with
    | Some window ->
        setTitle window title
        setDefaultSize window width height
        connectSignal window "destroy" onDestroy
        Some window
    | None ->
        None

/// Run the GTK main loop. Blocks until quit.
let runMainLoop () : unit =
    Fidelity.Gtk.Window.gtk_main ()
```

### 3.4 Program.clef (Entry Point)

```clef
module HelloGtk.Program

open Fidelity.Gtk.Window.Api
open HelloGtk.GtkApp

[<EntryPoint>]
let main _ =
    Console.writeln "=== GTK Window Test ==="

    init ()

    match createWindow "Fidelity GTK" 400 300 with
    | Some window ->
        Fidelity.Gtk.Window.gtk_widget_show window
        Console.writeln "Window shown, entering main loop..."
        runMainLoop ()
        Console.writeln "Main loop exited"
        0
    | None ->
        Console.writeln "Failed to create window"
        1
```

## 4. Compiler Pipeline (ExternCall Path)

No CCS or Composer modifications are needed for FFI. The ExternCall pathway is already implemented:

### 4.1 CCS: FidelityExtern Attribute Recognition

CCS recognizes `[<FidelityExtern>]` on function declarations and preserves the metadata (library name, symbol name) through type-checking. The `Unchecked.defaultof<T>` body is a compiler-recognized placeholder — CCS knows these are binding declarations, not implementations.

### 4.2 Baker: No Special Handling

Baker saturation treats extern declarations as leaf nodes. No decomposition needed.

### 4.3 PlatformBindingResolution Nanopass

`PlatformBindingResolution.fs` walks the PSG and resolves `[<FidelityExtern>]` metadata into `ExternCall` coeffects:

```
VarRef → Binding → metadata { library = "gtk-4", symbol = "gtk_window_new" }
    → ExternCall ("gtk-4", "gtk_window_new")
```

This is the **same mechanism** used by all Farscape-generated binding libraries (Wayland, libc, pthread, DRM, GBM, HIP, etc.).

### 4.4 Alex: pExternCallResolved Witness

The `pExternCallResolved` pattern in `PlatformPatterns.fs` witnesses `ExternCall` coeffects generically:

1. Reads the `ExternCall` coeffect for (library, symbol)
2. Emits `func.func private @symbol(...)` declaration with link attribute
3. Emits `func.call @symbol(args)` at the call site
4. Returns `TRValue` with the result SSA

No per-function witness code. One generic pattern handles all extern calls.

## 5. FFI Type Marshaling at the C Boundary

### 5.1 Pointer Types

GTK types are opaque pointers. At the MLIR level, all pointer types use `index`:

| Clef Type | MLIR Type | C Type |
|-----------|-----------|--------|
| `nativeint` | `index` | `void*` / `GtkWidget*` |
| `option<nativeint>` | `index` (0 = None) | nullable pointer |

The MLIR→LLVM lowering pass converts `index` to `!llvm.ptr` at the backend boundary.

### 5.2 Integer Types

C integer types are legitimate concretizations at the API boundary:

| Clef Type | MLIR Type | C Type |
|-----------|-----------|--------|
| `int` | `index` | `int` (platform-width) |
| `int32` | `i32` | `int32_t` / `gint` |
| `int64` | `i64` | `int64_t` / `gint64` |

The DTS resolves width at the ExternCall boundary. This is not premature concretization — C's `int32_t` is a genuine width demand.

### 5.3 String Types

String parameters at the C boundary require memref→pointer extraction:

| Clef Type | MLIR Extraction | C Type |
|-----------|-----------------|--------|
| `string` (literal) | `memref.get_global` → `memref.extract_aligned_pointer_as_index` | `const char*` |

This extraction is a **general FFI marshaling concern** handled by the ExternCall witness, not per-function special-casing.

### 5.4 Callback Function Pointers

GTK callbacks use C function pointer convention `void (*)(GtkWidget*, gpointer)`:

| Clef Type | MLIR | C Type |
|-----------|------|--------|
| `nativeint -> nativeint -> unit` (no captures) | function address as `index` | `void (*)(void*, void*)` |

Capture-free closures can be passed as C function pointers. The code pointer is directly usable as the callback address. Closures with captures would need a trampoline (future work, tracked in C-01).

## 6. MLIR Output Specification

### 6.1 External Function Declarations

```mlir
// Generated from [<FidelityExtern>] metadata via pExternCallResolved
func.func private @gtk_init() attributes { "link" = "gtk-4" }
func.func private @gtk_window_new() -> index attributes { "link" = "gtk-4" }
func.func private @gtk_window_set_title(index, index) attributes { "link" = "gtk-4" }
func.func private @gtk_window_set_default_size(index, i32, i32) attributes { "link" = "gtk-4" }
func.func private @gtk_widget_show(index) attributes { "link" = "gtk-4" }
func.func private @gtk_main() attributes { "link" = "gtk-4" }
func.func private @gtk_main_quit() attributes { "link" = "gtk-4" }
func.func private @g_signal_connect_data(index, index, index, index, index, i32) -> i64
    attributes { "link" = "gobject-2.0" }
```

### 6.2 Window Creation

```mlir
// init()
func.call @gtk_init() : () -> ()

// match windowNew() with
%window_raw = func.call @gtk_window_new() : () -> index
// Option wrapping: 0 = None, non-zero = Some
%zero = arith.constant 0 : index
%is_some = arith.cmpi ne, %window_raw, %zero : index
scf.if %is_some {
    // Some branch: set title, size, show, run loop
    %title = memref.get_global @str_fidelity_gtk : memref<13xi8>
    %title_ptr = memref.extract_aligned_pointer_as_index %title : memref<13xi8> -> index
    func.call @gtk_window_set_title(%window_raw, %title_ptr) : (index, index) -> ()

    %c400 = arith.constant 400 : i32
    %c300 = arith.constant 300 : i32
    func.call @gtk_window_set_default_size(%window_raw, %c400, %c300) : (index, i32, i32) -> ()

    func.call @gtk_widget_show(%window_raw) : (index) -> ()
    func.call @gtk_main() : () -> ()
} else {
    // None branch: error exit
}
```

### 6.3 Signal Connection

```mlir
// connectSignal window "destroy" onDestroy
%destroy_str = memref.get_global @str_destroy : memref<8xi8>
%destroy_ptr = memref.extract_aligned_pointer_as_index %destroy_str : memref<8xi8> -> index
%callback_ptr = llvm.mlir.addressof @onDestroy : !llvm.ptr
%callback_idx = llvm.ptrtoint %callback_ptr : !llvm.ptr to index
%null = arith.constant 0 : index
%c0 = arith.constant 0 : i32
func.call @g_signal_connect_data(%window_raw, %destroy_ptr, %callback_idx, %null, %null, %c0)
    : (index, index, index, index, index, i32) -> i64

// Callback function (capture-free, directly callable from C)
func.func @onDestroy(%widget: index, %data: index) {
    func.call @gtk_main_quit() : () -> ()
    func.return
}
```

## 7. Validation

### 7.1 Expected Behavior

- Window appears titled "Fidelity GTK"
- Window is 400x300 pixels
- Console prints "Window shown, entering main loop..."
- Closing window exits the program cleanly
- Console prints "Main loop exited"

### 7.2 Compilation Command

```bash
composer compile HelloGtk.fidproj
```

The fidproj dependency graph resolves `Fidelity.Gtk` and `Fidelity.GObject` automatically. The `[link]` section tells the linker to link against `libgtk-4.so` and `libgobject-2.0.so`.

### 7.3 Binding Generation Command

```bash
farscape generate --project gtk.pilot.toml
farscape generate --project gobject.pilot.toml
```

## 8. Implementation Checklist

### Phase 1: Binding Generation
- [ ] Create `gtk.pilot.toml` with GTK4 headers
- [ ] Create `gobject.pilot.toml` for signal connection
- [ ] Run `farscape generate` — verify L1/L2 output
- [ ] Verify fidproj generated correctly

### Phase 2: Application
- [ ] Create `HelloGtk.fidproj` with dependencies
- [ ] Write `GtkApp.clef` (behavior module)
- [ ] Write `Program.clef` (entry point)

### Phase 3: Compilation
- [ ] `composer compile HelloGtk.fidproj` succeeds
- [ ] ExternCall coeffects resolve for all GTK functions
- [ ] MLIR output contains correct `func.call @gtk_*` with link attributes

### Phase 4: Execution
- [ ] Binary links against GTK4 libraries
- [ ] Window appears and renders
- [ ] Destroy signal fires callback
- [ ] Clean shutdown via `gtk_main_quit`

## 9. Binding Unification: Sys.write → FidelityExtern

This PRD establishes the **single pathway** for all dynamic C library binding. The same `ExternCall` mechanism that calls `gtk_window_new` should also call libc's `write`. Currently, `Sys.write` uses a separate intrinsic pathway in CCS:

| Aspect | Current (Intrinsic) | Target (Unified ExternCall) |
|--------|---------------------|-----------------------------|
| CCS | Hardcoded `IntrinsicModule.Sys` | `[<FidelityExtern("c", "write")>]` in `Fidelity.Libc` |
| Coeffect | `LibcCall "write"` | `ExternCall ("c", "write")` |
| Witness | `pSysWriteIntrinsic` (dedicated) | `pExternCallResolved` (generic) |

The specialized memref→pointer extraction currently in `pSysWriteIntrinsic` will be generalized into FFI marshaling logic within the ExternCall pathway (see Section 5.3).

## 10. Related PRDs

- **C-01**: Closures — capture-free closures as C function pointers
- **D-02**: WebViewBasic — WebKit in GTK (multi-library L3 composition)
- **I-01**: SocketBasics — same ExternCall pathway for POSIX socket functions
- **T-01, T-02**: Threading/Mutex — same ExternCall pathway for pthread functions
