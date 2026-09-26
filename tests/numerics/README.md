# Numeric reference oracles

Run the independent construction examples with the .NET host:

```sh
dotnet fsi tests/numerics/ConstructionOracles.fsx
```

The case identifiers come from the registered
[F-11 / C-08 validation inventory](../../docs/PRDs/Numeric_Validation_Cases.md).
This script computes exact rational reference quantities and explicit binary64
nearest-even rounding for the named finite-result cases. It checks cancellation,
exact versus rounded products, TwoSum residuals, fixed-point rescaling, once-only
initialization, partial capacity and loss during partial-state transport.

Its binary64 value model identifies both signs of zero as rational zero and
excludes nonfinite results. Signed-zero/status/nonfinite observations, other
formats and full construction-domain arguments retain their separate case
contracts. A few TwoSum fixtures do not prove its domain-wide theorem. The
independent model supplies fixture answers, not source-language semantics.

Every compiler acceptance run must compare actual CCS/Baker, Alex and backend
results with the applicable oracle and retain its input/dependency identity.
Running this script alone establishes no compiler or native-program pass.
