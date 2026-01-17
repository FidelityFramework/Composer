# PRD-13: Recursive Bindings

> **Sample**: `13_Recursion` | **Status**: In Progress | **Depends On**: PRD-11 (Closures), PRD-12 (HOFs)

## 1. Executive Summary

Recursive functions (`let rec`) require that a binding be visible within its own body. This PRD implements:
1. **Simple recursion** - `let rec f x = ... f ...`
2. **Nested recursion** - `let f x = let rec loop y = ... loop ... in loop x`
3. **Mutual recursion** - `let rec f x = ... g ... and g y = ... f ...`

## 2. The Core Challenge

A recursive function references itself before its definition is complete:

```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)  // VarRef to 'factorial' - but we're still defining it!
```

The VarRef to `factorial` inside the body needs a `defId` pointing to the Binding node. But at the time we check the body, the Binding node doesn't exist yet.

## 3. FNCS Implementation

### 3.1 Current State (Incomplete)

In `checkLetOrUse` for `let rec`:
```fsharp
let bindingEnv =
    if isRec then
        bindings |> List.fold (fun env binding ->
            let name = getBindingName binding
            let ty = freshTypeVar range
            addBinding name ty false None env  // NodeId = None!
        ) env
```

This adds bindings to the environment with `NodeId = None`, so self-referential VarRefs get `defId = None`.

### 3.2 Required Implementation

**Pre-create Binding nodes** to obtain NodeIds before checking bodies:

```fsharp
let checkLetOrUse ... =
    let bindings = letOrUse.Bindings
    let isRec = letOrUse.IsRecursive

    if isRec then
        // STEP 1: Pre-create Binding nodes to get NodeIds
        let bindingNodes = bindings |> List.map (fun binding ->
            let name = getBindingName binding
            let ty = freshTypeVar range
            let node = builder.Create(
                SemanticKind.Binding(name, false, true, false),  // isRec = true
                ty,
                range)
            (binding, name, ty, node))

        // STEP 2: Add ALL bindings to environment WITH their NodeIds
        let envWithBindings = bindingNodes |> List.fold (fun env (_, name, ty, node) ->
            addBinding name ty false (Some node.Id) env
        ) env

        // STEP 3: Check each body in the extended environment
        let checkedBindings = bindingNodes |> List.map (fun (binding, name, ty, bindingNode) ->
            // Check the body - VarRefs to 'name' now resolve to bindingNode.Id
            let bodyNode = checkBinding ... envWithBindings ...
            // Update the binding node with the actual body
            (bindingNode, bodyNode))

        // STEP 4: Build result
        ...
    else
        // Non-recursive: existing logic
        ...
```

### 3.3 Key Principle

> **The NodeId must exist before we need to reference it.**

For recursive bindings:
1. Create the Binding node (get NodeId)
2. Add to environment with that NodeId
3. Check body (VarRefs resolve via environment)
4. Connect body to Binding node

### 3.4 Files to Modify

| File | Change |
|------|--------|
| `Expressions/Bindings.fs` | Restructure `checkLetOrUse` for recursive bindings |

**No other FNCS files need changes.** The SemanticKind.Binding already has an `isRec` flag. VarRef already supports `defId: NodeId option`.

## 4. Firefly Implementation

### 4.1 SSAAssignment: Nested Function Naming

For nested functions with the same name (e.g., `loop` in multiple functions), qualify names by walking the Parent chain:

```fsharp
// In collectLambdas:
let findEnclosingFunctionName (startId: NodeId) : string option =
    let rec walk nodeId passedFirstLambda =
        match graph.Nodes.TryFind nodeId with
        | None -> None
        | Some n ->
            match n.Kind with
            | SemanticKind.Lambda _ when passedFirstLambda ->
                // Found enclosing Lambda - get parent Binding's name
                match n.Parent with
                | Some pid ->
                    match graph.Nodes.[pid].Kind with
                    | SemanticKind.Binding(name, _, _, _) -> Some name
                    | _ -> None
                | None -> None
            | _ ->
                match n.Parent with
                | Some pid -> walk pid true
                | None -> None
    walk startId false

// Usage: qualify nested names
match findEnclosingFunctionName lambdaId with
| Some enclosing -> sprintf "%s_%s" enclosing baseName
| None -> baseName
```

### 4.2 No Witness Changes Needed

With proper PSG (VarRefs have defIds), the existing witness code handles recursive calls correctly:
- VarRef has defId → lookup in lambdaNames → emit call

## 5. Verification

### 5.1 PSG Check
```
VarRef ("factorial", Some (NodeId 547))  // Self-reference has defId
```

### 5.2 MLIR Check
```mlir
llvm.func @factorial(%n: i32) -> i32 {
    ...
    %result = llvm.call @factorial(%n_minus_1) : (i32) -> i32  // Recursive call works
    ...
}
```

### 5.3 Execution Check
```
factorial 5: 120
factorialTail 5: 120
```

## 6. Implementation Checklist

### Phase 1: FNCS - Recursive Binding NodeIds
- [ ] Restructure `checkLetOrUse` to pre-create Binding nodes for `let rec`
- [ ] Add NodeIds to environment before checking bodies
- [ ] Verify: VarRef to self has `defId = Some nodeId`
- [ ] FNCS builds

### Phase 2: Firefly - Nested Function Naming
- [ ] SSAAssignment uses Parent chain for qualified names
- [ ] Verify: `@factorialTail_loop` not `@loop`
- [ ] Firefly builds

### Phase 3: Validation
- [ ] RecursionSimple compiles
- [ ] RecursionSimple executes correctly
- [ ] Samples 01-12 still pass

## 7. Related PRDs

- **PRD-11**: Closures - nested functions may capture
- **PRD-12**: HOFs - recursive functions as values
