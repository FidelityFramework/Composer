# Dimensional vet programs

The program set of `clef/docs/fidelity/phg/Dimensional_Vetting_Plan.md` §3 and
`clef/docs/fidelity/phg/Dimensional_Step1_2_Design.md` §(g): one minimal program per rule that must
be rejected and one that must be accepted. The table `vet.sh` prints is the gate for every
hardening step of the plan (§5); a rule is green when both of its programs report as expected.

## Layout

```text
<rule>/{reject,accept}/
    Program.clef     the program: the common prelude of design note (g) plus the row's fragment,
                     ending in an [<EntryPoint>] main that returns 0
    <rule>.fidproj   cloned from BAREWire/samples/RoundTrip/RoundTrip.fidproj (name and source list
                     adjusted; the barewire dependency dropped)
    expect.toml      verdict, code, count, flags, pending, rule (see vet.sh header)
    targets/         vet.sh transcripts (vet.<name>.stdout, vet.<name>.stderr); ignored by git
```

Rules with no reject program (UoM-8, W-3, W-6, NS-4) have only `accept/`. W-3 is the differential
row: one `Program.clef` under two `.fidproj` platform contexts (Linux x86_64 and a Cortex-M33
descriptor), one table row each.

## Running

```bash
dotnet build /home/hhh/repos/Composer/src/Composer.fsproj   # vet.sh never builds
./vet.sh                 # every row reported; all pending: the baseline measurement
./vet.sh --through 2     # rows gated at steps 1-2 are judged; exit 0 only if they all match
./vet.sh --intermediates # keep -k intermediates under <leaf>/targets for inspection (60 s per leaf instead of 7 s)
```

`vet.sh` never calls `tests/regression/Runner.fsx`. The RoundTrip native gate
(`BAREWire/samples/RoundTrip`, diffed against `expected.txt`) runs alongside it, by hand.
