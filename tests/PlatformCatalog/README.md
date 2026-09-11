# Platform catalogue integrity

Run `dotnet run --project tests/PlatformCatalog/PlatformCatalog.Tests.fsproj`
from the Composer repository. Optional arguments are additional consumer
`.fidproj` paths.

The F# runner uses CCS's real manifest and dependency resolver to check every
Fidelity.Platform manifest. It rejects missing sources, dependency cycles,
duplicate source identities and platform source lists that retain a non-Clef
extension. This checks the catalogue's composition paths; it does not claim
that scaffold packages have working drivers or deployment backends.

`PlatformComposition` exercises explicit export selection and declaration
provenance. `DeviceAccess`, `Mmio`, `MCU`, `IOMap` and `NativeCallbacks` check the
relevant lowering, image and native ABI behavior.
