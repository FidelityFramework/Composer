# Optional folds

Status: registered in the regression manifest; compilation, native execution
and exact output pass at the C-04 optional-fold waypoint.

This variant exercises `Option.fold` and `Option.foldBack`: callback argument
order, one invocation for `Some`, unchanged state for `None`, eager operand and
pipe order, stored operand snapshots, shared anonymous captures, independent
state/payload dimensions including inverse measures, and function-valued state.
It uses Platform's CompilerSurface dependency.

`fold` takes folder, state, option; its callback takes state, payload. `foldBack`
takes folder, option, state; its callback takes payload, state. Both retain the
three-argument operation boundary when the state is a function. Their canonical
contracts are specified in the
[Option specification](../../../../../clef-lang-spec/spec/option-operations-representation.md)
with this compiler waypoint.

Successful execution must exit zero and print exactly:

```text
=== Option folds Test ===
Callback order and empty state: pass
Factory and pipe order: pass
Stored operands and shared captures: pass
Independent measured state: pass
Function-valued state: pass
```

The companion `tests/NativeCallbacks/OptionFolds.clef` has 15 semantic groups
(failure codes 161–175), including both partial frontiers, both pipe directions,
record/function/unit payloads and ordered extra application of function-valued
state. Both native gates pass; the waypoint record retains their compiler and
artifact evidence.
