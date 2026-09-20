# Captured sequence templates

This C-06 oracle requires the surrounding activation to cover every use of a
captured sequence template. It checks deferred repeated delegation, fresh
iteration state with shared mutable source cells, and two levels of captured
templates. Return codes 81–83 identify failed semantic groups.

Native acceptance passes on CCS `f4bbc287…432c1a`: fresh compilation, stock
MLIR verification, native exit zero and exact output. Evidence is retained at
`/tmp/composer-native-sequences-6f767e0ab8374941a1d041b0b9789985`.
Run through NativeSequences with
`--sample 15c_SequenceTemplateBorrows`; compilation, stock MLIR verification,
exact output and native exit zero are separate required stages. An escaping
capturing sequence requires additional residence evidence and is covered by
negative CCS tests rather than admitted by lexical nesting alone.
