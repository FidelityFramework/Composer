# HelloDimensionsProof

An ordered two-file Lattice sample combining measured-type hover with the compiler-generated string and input-buffer obligations exercised by HelloProof. Open this directory as the project workspace. `Units.clef` declares meters and seconds and defines `speed`; `Main.clef` computes `velocity`, uses its measured result in a comparison, and runs a greeting.

The project reads the sibling `Fidelity.Platform` checkout through a relative dependency; that platform brings its BAREWire dependency. Keep Composer, Clef/CCS, Fidelity.Platform, and BAREWire at compatible revisions. The [Lattice integration plan](../../../docs/Lattice_Integration.md) describes the development server/client setup and current gates.

## Editor walk-through

1. Open `Main.clef` and hover over `velocity` or the call to `speed`. The inferred result should retain the dimension `m/s`.
2. Go to definition on `speed` to reach `Units.clef`.
3. Comment out `open HelloDimensionsProof.Units`. With the corrected compiler integration, `speed` becomes unresolved and its definition link disappears. Restore the import to recover both.
4. Replace `3.0<s>` with `3.0<m>` in `Main.clef`. The call then has an incompatible dimension. Restore seconds and check that the diagnostic clears after the edit.
5. Inspect the compiler obligations attached to the greeting's reachable string literals and the input operation. Where the client offers proof expansion, show the generated statement and its current dispatch status.

The measured calculation and string/buffer proofs demonstrate different compiler results. A discharged string-storage obligation does not prove the arithmetic calculation or the emitted executable. A generated obligation is pending until its current solver query has completed; changing the source must invalidate any verdict associated with the old snapshot.

For the native greeting, an available Composer CLI can compile from this directory with `composer compile HelloDimensionsProof.fidproj -k`; the program expects a name on standard input. Native compilation and artifact reconciliation are separate from the editor check. The original [HelloProof sample](https://github.com/FidelityFramework/ship-of-theseus/tree/main/HelloProof) remains the fuller reconciliation demonstration.
