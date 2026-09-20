# Native formatter boundaries

Run from Composer with the existing compiler build:

```sh
dotnet fsi tests/PlatformFormat/Runner.fsx
```

The F# bootstrap runner uses Composer's existing regression process infrastructure.
It copies the actual peer Fidelity.Platform `Format.clef` and the Clef fixture
into a fresh temporary directory, uses the CompilerSurface platform declaration,
and retains source/compiler hashes, compilation artifacts and native output.
It requires successful compilation, stock MLIR verification, native exit zero,
empty stderr, and all 22 expected output lines. Cases include zero, positive and
negative decimal transitions, signed 64-bit endpoints of this platform, and
exactly representable signed fractions. These endpoints are test values, not
source width annotations.

Before printing, the source checks ASCII and multibyte UTF-8, an empty conversion,
a nonzero-offset slice, an unrelated integer array containing 4096, and snapshot
isolation in both directions. It mutates the original array after `fromBytes`,
then writes 4096 into a `toBytes` result and checks that the string is unchanged.
Failure exits 93. The public array remains an ordinary integer array; only the
internal encoding view has the byte representation.

September 20 acceptance: **PASS** — fresh compilation, stock MLIR verification,
native exit zero, empty stderr and exact output. Retained evidence:
`/tmp/composer-platform-format-c854c08cf4ff4ad8875fad9821954a8e/`.
`inputs.json` records the actual compiler and source hashes. This supersedes the
earlier `memref<?xi64>`/`memref<?xi8>` failure; Baker now settles storage and copies
before Alex witnesses the graph. The test runner belongs to Composer's bootstrap
test infrastructure; the earlier Python draft in Fidelity.Platform was removed.

The standard is the
[integer byte-unit conversion contract](../../../clef-lang-spec/spec/native-type-mappings.md#integer-byte-unit-conversions).
Current `fromBytes` admission proves closed ASCII buffers or immutable constant
UTF-8 sequences. This gate does not establish dynamic UTF-8 validation, arbitrary
floating-point formatting precision or formatting of nonfinite values.
