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

Rules with no reject program (UoM-8, UoM-10, W-3, W-6, NS-4) have only `accept/`. UoM-10 (added with
CS-6) is the generalisation row the sequence recommended: `let id (x: float<'u>) = x` used at `float<m>`
and `float<s>`, its scheme `float<'u> -> float<'u>`. W-3 is the differential
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

## Baseline readings

- 2026-09-04, before any changeset: 4 of 45 rows match.
- 2026-09-04, after CS-4 (the numeric type carries its dimension; the unifier solves measures): 16 of 45.
  Read with care: UoM-2, UoM-3, UoM-4 and UoM-9 rejects and NS-1/reject are green because the interim
  `'T -> 'T -> 'T` operator type rejects `*` and `/` between different measures; they move again when the
  step-3 schemes land. NS-2/reject and NS-3/reject were rejected before only by an arity accident on
  annotated leaves; they are accepted until step 8 supplies the range coeffect, and their rows stay
  step-8-pending.
- 2026-09-04, after CS-5 (one spelling table): 17 of 45. NS-4/accept is green because `float64` is now a
  type spelling (the explicit IEEE-64 seal), not because step 8's design-time witness exists.
- 2026-09-04, after CS-6 (carrier variables, operator schemes, generalisation): 24 of 46; `vet.sh --through 2`
  judges 10 of 10. NS-1/reject moved back to accept as foreseen: the interim operator type that rejected it is
  gone, and its own rule waits on step 8.

**Reading after CS-8 (2026-09-04):** 24 of 46, 10 of 10 judged through step 2, identical to CS-7 except that the ten M rows now print the located `CCS8706` (the step-5 types `Stack`, `Flash`, `Peripheral` are not defined yet) as their first error instead of a later `CCS8009` or a Composer type-mapping failure. Baseline transcript: `vet-cs8.txt` in the session scratchpad.

**D10 (2026-09-04):** there is no width-named type and no width suffix (`Dimensional_Range_Design.md`). The W rows and the M rows' handles were rewritten to `int`; W-1 now rejects the spellings themselves (CCS8706), W-5 is the coverage check at an 8-bit register boundary the Contracts leaf declares, W-6 joins call-site ranges. Readings taken before this date used the retired spellings.
