# Sequence startup order

This strict native oracle uses two ordered source files, each containing two
namespace/module declarations. Declaration names deliberately differ from
alphabetical order. Four effectful initializers must execute in project and
source order before `main`, including a binding whose value is never referenced.
That unreferenced initializer emits the first expected output line.

A module sequence forms once during startup and remains cold until enumeration.
Two manual enumerations in `main` and two enumerations through a named function
must each observe both scalar values. Shared pull counters check demand, while
startup counters and repeated reads detect duplicate initialization. Caller body
effects must occur after all initializers. Failure codes 201–205 identify the
five groups; the remaining fixed output appears only after every check passes.

The project uses `Linux_x86_64_Default`, including its declared program-lifetime
storage authority. Native validation is pending implementation and retained
runner evidence. This fixture does not admit cyclic initializer dependencies,
unknown escaping sequence destinations or aggregate callback captures.

```sh
dotnet run --project tests/NativeSequences/NativeSequences.Tests.fsproj -- src/bin/Debug/net10.0/Composer --sample 16g_SequenceStartup
```
