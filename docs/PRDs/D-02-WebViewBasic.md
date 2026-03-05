# D-02: WebKitGTK WebView (Multi-Library Composition)

> **Sample**: `26_WebViewBasic` | **Status**: Planned | **Depends On**: D-01 (GTKWindow)

## 1. Executive Summary

This PRD adds WebKitGTK WebView to GTK windows, enabling hybrid web/native applications. This is the foundation for the WREN (WebView + Region + Elmish + Native) stack vision.

**Key Insight**: WebKitGTK provides a C API similar to GTK. The same ExternCall pathway from D-01 applies. The application imports **two** Farscape-generated binding libraries (`Fidelity.Gtk` and `Fidelity.WebKit`) — demonstrating **Layer 3 composition** where two generated libraries come together in application code.

JavaScript<->Clef bridging is future work (PRD beyond scope).

## 2. Binding Generation (Farscape)

### 2.1 WebKit Pilot TOML

```toml
[library]
name = "webkitgtk-6.0"
headers = ["/usr/include/webkitgtk-6.0/webkit/webkit.h"]
include_paths = [
    "/usr/include/webkitgtk-6.0",
    "/usr/include/gtk-4.0",
    "/usr/include/glib-2.0",
    "/usr/lib/glib-2.0/include"
]
macro_prefixes = ["WEBKIT_"]

[output]
mode = "fidelity"
directory = "../Bindings/WebKit"

[options]
opaque_handles = true

[[namespace]]
name = "Fidelity.WebKit.View"
description = "WebView creation, content loading, and navigation"
library = "webkitgtk-6.0"
prefixes = ["webkit_web_view"]
```

### 2.2 Generated L1 Declarations (View.clef)

```clef
module Fidelity.WebKit.View

[<FidelityExtern("webkitgtk-6.0", "webkit_web_view_new")>]
let webkit_web_view_new () : nativeint = Unchecked.defaultof<nativeint>

[<FidelityExtern("webkitgtk-6.0", "webkit_web_view_load_uri")>]
let webkit_web_view_load_uri (view: nativeint) (uri: nativeint) : unit = Unchecked.defaultof<unit>

[<FidelityExtern("webkitgtk-6.0", "webkit_web_view_load_html")>]
let webkit_web_view_load_html (view: nativeint) (content: nativeint) (baseUri: nativeint) : unit = Unchecked.defaultof<unit>
```

### 2.3 Generated L2 Wrappers (ViewApi.clef)

```clef
module Fidelity.WebKit.View.Api

open Fidelity.WebKit.View

/// Create a new WebView widget. Returns None if creation fails.
let webViewNew () : option<nativeint> =
    let result = webkit_web_view_new ()
    if result = 0n then None
    else Some result

/// Load HTML content with a base URI.
let loadHtml (view: nativeint) (content: nativeint) (baseUri: nativeint) : unit =
    webkit_web_view_load_html view content baseUri

/// Navigate to a URL.
let loadUri (view: nativeint) (uri: nativeint) : unit =
    webkit_web_view_load_uri view uri
```

## 3. Application Code

### 3.1 Multi-Library Composition

This sample demonstrates **Layer 3 composition** — application code that depends on multiple Farscape-generated libraries:

```
HelloWebView/
├── HelloWebView.fidproj
└── src/
    ├── Program.clef
    └── WebApp.clef
```

### 3.2 HelloWebView.fidproj

```toml
[package]
name = "HelloWebView"
version = "0.1.0"
description = "Clef Language + Fidelity Framework — WebKitGTK WebView"

[compilation]
target = "cpu"

[dependencies]
platform = { path = "../Fidelity.Platform/CPU/Linux/x86_64/Fidelity.Platform.fidproj" }
gtk = { path = "../Fidelity.Platform/CPU/Linux/x86_64/Fidelity.Gtk.fidproj" }
webkit = { path = "../Fidelity.Platform/CPU/Linux/x86_64/Fidelity.WebKit.fidproj" }

[build]
sources = ["src/WebApp.clef", "src/Program.clef"]
output = "HelloWebView"
output_kind = "console"

[link]
libraries = ["gtk-4", "webkitgtk-6.0"]
```

### 3.3 WebApp.clef (Behavior)

```clef
module HelloWebView.WebApp

open Fidelity.Gtk.Window.Api
open Fidelity.WebKit.View.Api

let htmlContent = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: system-ui;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        h1 { font-size: 3em; }
    </style>
</head>
<body>
    <h1>Hello WREN Stack!</h1>
</body>
</html>
"""

/// Create a window with an embedded WebView showing HTML content.
let createWebWindow (title: nativeint) (width: int) (height: int) (html: nativeint) : option<nativeint> =
    match windowNew () with
    | Some window ->
        setTitle window title
        setDefaultSize window width height
        match webViewNew () with
        | Some webview ->
            loadHtml webview html "about:blank"
            Fidelity.Gtk.Window.gtk_window_set_child window webview
            Some window
        | None -> None
    | None -> None
```

### 3.4 Program.clef (Entry Point)

```clef
module HelloWebView.Program

open Fidelity.Gtk.Window.Api
open HelloWebView.WebApp

[<EntryPoint>]
let main _ =
    Console.writeln "=== WebView Basic Test ==="

    init ()

    match createWebWindow "WREN WebView" 800 600 htmlContent with
    | Some window ->
        Fidelity.Gtk.Window.gtk_widget_show window
        Console.writeln "WebView shown, entering main loop..."
        Fidelity.Gtk.Window.gtk_main ()
        0
    | None ->
        Console.writeln "Failed to create web window"
        1
```

## 4. Compiler Pipeline

No CCS or Composer modifications needed beyond D-01. The ExternCall pathway handles both GTK and WebKit function calls identically — they differ only in the library name in the `"link"` attribute.

## 5. MLIR Output Specification

### 5.1 WebKit Function Declarations

```mlir
// From Fidelity.WebKit — same ExternCall pattern as GTK
func.func private @webkit_web_view_new() -> index attributes { "link" = "webkitgtk-6.0" }
func.func private @webkit_web_view_load_uri(index, index) attributes { "link" = "webkitgtk-6.0" }
func.func private @webkit_web_view_load_html(index, index, index) attributes { "link" = "webkitgtk-6.0" }
```

### 5.2 WebView Creation and Loading

```mlir
%webview = func.call @webkit_web_view_new() : () -> index

%html_ref = memref.get_global @html_content : memref<...xi8>
%html = memref.extract_aligned_pointer_as_index %html_ref : memref<...xi8> -> index
%base_ref = memref.get_global @about_blank : memref<11xi8>
%base = memref.extract_aligned_pointer_as_index %base_ref : memref<11xi8> -> index
func.call @webkit_web_view_load_html(%webview, %html, %base) : (index, index, index) -> ()
```

## 6. Validation

### 6.1 Expected Behavior

- Window appears titled "WREN WebView" at 800x600
- WebView displays styled "Hello WREN Stack!" with gradient background
- Closing window exits program cleanly

### 6.2 Commands

```bash
farscape generate --project webkit.pilot.toml
composer compile HelloWebView.fidproj
```

## 7. Implementation Checklist

### Phase 1: Binding Generation
- [ ] Create `webkit.pilot.toml`
- [ ] Run `farscape generate` — verify L1/L2 WebKit output
- [ ] Verify fidproj generated with correct dependencies

### Phase 2: Application
- [ ] Create `HelloWebView.fidproj` with GTK + WebKit dependencies
- [ ] Write `WebApp.clef` and `Program.clef`
- [ ] Verify multi-library `open` resolution works

### Phase 3: Validation
- [ ] `composer compile` succeeds with both library dependencies
- [ ] Binary links against both `libgtk-4.so` and `libwebkitgtk-6.0.so`
- [ ] WebView renders HTML content

## 8. Future: JavaScript Bridge

Full WREN stack needs JavaScript<->Clef communication:

```clef
// Future API — via Farscape-generated WebKit bindings
Fidelity.WebKit.View.webkit_web_view_run_javascript webview "document.title = 'Updated'"

// Callback from JS to Clef — uses Farscape L2 callback registration
registerScriptMessageHandler webview "clef" onMessage
```

This is beyond the scope of Sample 26 but establishes the foundation.

## 9. Related PRDs

- **D-01**: GTKWindow — GTK foundation, ExternCall pathway definition
- **T-03 to T-05**: MailboxProcessor — UI event handling
- (Future): Elmish architecture, JS bridge
