/// SMTTransfer: the build-time dispatch -- the graph's obligations as `smt` dialect IR
///
/// Architecturally parallel to XDCTransfer. Both are transfers: pure functions
/// from settled data to text, with no traversal and no decision.
///   - XDCTransfer: the pin-mapping coeffect -> XDC constraints
///   - SMTTransfer: the graph's obligation nodes -> an `smt` verification module
///
/// Obligations are graph citizens: minted into the PSG by the Baker obligation
/// recipes at saturation (CCS Pass 5), each a node in V with a hyperedge in F
/// whose source set is the structure it constrains (C-01 14.5;
/// Obligation_Residency 3). This module reads those records from the graph and
/// transcribes them. Every fact in an ObligationBody was fixed at saturation;
/// nothing here computes.
///
/// One birth, two dispatches. The anchor name is the identity that travels:
/// CCS renders the same records to SMT-LIB at design time (06b); this module
/// renders them to `smt` dialect at build time (09), and mlir-translate
/// --export-smtlib carries the `smt.declare_fun` names through verbatim, so the
/// two dispatches pair one for one (HelloProof proof-trace 03).
///
/// Output: a standalone verification module beside the program (one
/// smt.solver scope per obligation, refutation style: `unsat` on every check
/// means every obligation holds).
module Alex.Traversal.SMTTransfer

open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.DimensionAlgebra
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

/// Transcribe one obligation into an isolated smt.solver scope.
/// SSA numbering is fresh per scope (solver regions are isolated).
let private scope (ob: ObligationInfo) : MLIROp list =
    let mutable n = -1
    let v () = n <- n + 1; Alex.Traversal.Values.solverValue n   // fresh per isolated solver scope: the SMT module, not the program graph
    let smt op = MLIROp.SMTOp op

    // Anchor discipline: %ob names the obligation; assert (ob = definition)
    // and (not ob). The declare_fun name is the identity that travels.
    let anchor (defSSA: SSA) (body: MLIROp list) : MLIROp list =
        let ob' = v ()
        let bind = v ()
        let neg = v ()
        body
        @ [ smt (SMTDeclareFun (ob', ob.Id, SMTBool))
            smt (SMTEq (bind, ob', defSSA, SMTBool))
            smt (SMTAssert bind)
            smt (SMTNot (neg, ob'))
            smt (SMTAssert neg) ]

    // The current SMT dialect has no Real sort. For these ground rational
    // facts, a/b <= c/d is exactly a*d <= c*b when b,d > 0. Transcribe the
    // products of integer constants rather than evaluating the comparison;
    // this preserves the source QF_LRA claim in a QF_LIA solver scope.
    let rationalComparisons (pairs: (ExactRational * ExactRational) list) : MLIROp list =
        let statements = ResizeArray<MLIROp>()
        let constant value =
            let ssa = v ()
            statements.Add(smt (SMTBigIntConstant(ssa, value)))
            ssa
        let product numerator denominator =
            let a, b, output = constant numerator, constant denominator, v ()
            statements.Add(smt (SMTIntMul(output, a, b)))
            output
        let clauses = pairs |> List.map (fun (left, right) ->
            if left.Denominator <= 0I || right.Denominator <= 0I then
                invalidArg "ob" (sprintf "Obligation '%s' requires positive rational denominators." ob.Id)
            let l = product left.Numerator right.Denominator
            let r = product right.Numerator left.Denominator
            let clause = v ()
            statements.Add(smt (SMTIntCmp(clause, SmtLe, l, r)))
            clause)
        let conjunction = v ()
        statements.Add(smt (SMTAnd(conjunction, clauses)))
        anchor conjunction (List.ofSeq statements)

    let integerComparisons (pairs: (bigint * bigint) list) : MLIROp list =
        let statements = ResizeArray<MLIROp>()
        let clauses = pairs |> List.map (fun (left, right) ->
            let l, r, comparison = v (), v (), v ()
            statements.Add(smt (SMTBigIntConstant(l, left)))
            statements.Add(smt (SMTBigIntConstant(r, right)))
            statements.Add(smt (SMTIntCmp(comparison, SmtLe, l, r)))
            comparison)
        let conjunction = v ()
        statements.Add(smt (SMTAnd(conjunction, clauses)))
        anchor conjunction (List.ofSeq statements)

    let ops =
        match ob.Body with
        | ObligationBody.FiniteLoopTrip model ->
            let statements = ResizeArray<MLIROp>()
            let constant value =
                let output = v ()
                statements.Add(smt (SMTBigIntConstant(output, value)))
                output
            let cmp pred left right =
                let output = v ()
                statements.Add(smt (SMTIntCmp(output, pred, left, right)))
                output
            let start, limit = constant model.InitialLower, constant model.LimitUpper
            let step, count, zero = constant model.MinimumStep, constant model.MaximumIterations, constant 0I
            let distance, ending = v (), v ()
            statements.Add(smt (SMTIntMul(distance, count, step)))
            statements.Add(smt (SMTIntAdd(ending, start, distance)))
            let clauses = [cmp SmtGt step zero; cmp SmtGe count zero
                           cmp (if model.Inclusive then SmtGt else SmtGe) ending limit]
            let conclusion = v ()
            statements.Add(smt (SMTAnd(conclusion, clauses)))
            anchor conclusion (List.ofSeq statements)
        | ObligationBody.AdditiveLoopInvariant model ->
            let statements = ResizeArray<MLIROp>()
            let constant value =
                let output = v ()
                statements.Add(smt (SMTBigIntConstant(output, value)))
                output
            let declare name =
                let output = v ()
                statements.Add(smt (SMTDeclareFun(output, name, SMTInt)))
                output
            let binary make left right =
                let output = v ()
                statements.Add(smt (make (output, left, right)))
                output
            let cmp pred left right =
                let output = v ()
                statements.Add(smt (SMTIntCmp(output, pred, left, right)))
                output
            let conjunction clauses =
                let output = v ()
                statements.Add(smt (SMTAnd(output, clauses)))
                output
            let implies premise conclusion =
                let negated, output = v (), v ()
                statements.Add(smt (SMTNot(negated, premise)))
                statements.Add(smt (SMTOr(output, [negated; conclusion])))
                output
            let lo, hi = constant model.InitialLower, constant model.InitialUpper
            let dlo, dhi = constant model.DeltaLower, constant model.DeltaUpper
            let lower, upper = constant model.Lower, constant model.Upper
            let count, zero, one = constant model.MaximumIterations, constant 0I, constant 1I
            let down, up = constant (min 0I model.DeltaLower), constant (max 0I model.DeltaUpper)
            let k, value, delta = declare "recurrence_k", declare "recurrence_value", declare "recurrence_delta"
            let envelope seed slope iteration = binary SMTIntAdd seed (binary SMTIntMul iteration slope)
            let lowAtK, highAtK = envelope lo down k, envelope hi up k
            let domain = conjunction [cmp SmtLe zero k; cmp SmtLe k count]
            let enclosed = conjunction [cmp SmtLe lower lowAtK; cmp SmtLe highAtK upper]
            let stepDomain = conjunction [cmp SmtLe zero k; cmp SmtLt k count
                                          cmp SmtLe lowAtK value; cmp SmtLe value highAtK
                                          cmp SmtLe dlo delta; cmp SmtLe delta dhi]
            let nextK, nextValue = binary SMTIntAdd k one, binary SMTIntAdd value delta
            let preserved = conjunction [cmp SmtLe (envelope lo down nextK) nextValue
                                         cmp SmtLe nextValue (envelope hi up nextK)]
            // Numeric consistency is part of the conclusion. Global assertions
            // here would hide malformed models behind an empty premise domain.
            let conclusion = conjunction [cmp SmtGe count zero; cmp SmtLe lo hi; cmp SmtLe dlo dhi
                                          cmp SmtLe lower upper; cmp SmtLe lower lo; cmp SmtLe hi upper
                                          implies domain enclosed; implies stepDomain preserved]
            anchor conclusion (List.ofSeq statements)
        | ObligationBody.FiniteAdditiveEffects(initial, contributions, lower, upper) ->
            let statements = ResizeArray<MLIROp>()
            let constant value =
                let output = v ()
                statements.Add(smt (SMTBigIntConstant(output, value)))
                output
            let binary make left right =
                let output = v ()
                statements.Add(smt (make(output, left, right)))
                output
            let cmp predicate left right =
                let output = v ()
                statements.Add(smt (SMTIntCmp(output, predicate, left, right)))
                output
            let conjunction values =
                let output = v ()
                statements.Add(smt (SMTAnd(output, values)))
                output
            let zero = constant 0I
            let mutable total = constant initial
            let mutable domain = [cmp SmtLe zero zero]
            let mutable counts = []
            for index, (maximum, delta) in List.indexed contributions do
                let actual = v ()
                statements.Add(smt (SMTDeclareFun(actual, sprintf "effect_count_%d" index, SMTInt)))
                let maximum, delta = constant maximum, constant delta
                counts <- cmp SmtGe maximum zero :: counts
                domain <- cmp SmtLe zero actual :: cmp SmtLe actual maximum :: domain
                total <- binary SMTIntAdd total (binary SMTIntMul actual delta)
            let enclosed = conjunction [cmp SmtLe (constant lower) total; cmp SmtLe total (constant upper)]
            let premise = conjunction domain
            let negated, implication = v (), v ()
            statements.Add(smt (SMTNot(negated, premise)))
            statements.Add(smt (SMTOr(implication, [negated; enclosed])))
            let conclusion = conjunction (implication :: counts)
            anchor conclusion (List.ofSeq statements)
        | ObligationBody.FiniteLinearRecurrence model ->
            let dimension = model.InitialLower.Length
            let vectorShape values = List.length values = dimension
            let matrixShape rows = List.length rows = dimension && List.forall vectorShape rows
            let wellShaped =
                (dimension = 1 || dimension = 2) && vectorShape model.InitialUpper &&
                vectorShape model.Lower && vectorShape model.Upper &&
                matrixShape model.CoefficientLower && matrixShape model.CoefficientUpper &&
                (model.Powers |> List.forall (fun step -> matrixShape step.Matrix))
            if not wellShaped then integerComparisons [1I, 0I]
            else
                let statements = ResizeArray<MLIROp>()
                let constant value =
                    let output = v ()
                    statements.Add(smt (SMTBigIntConstant(output, value)))
                    output
                let binary make left right =
                    let output = v ()
                    statements.Add(smt (make (output, left, right)))
                    output
                let compare pred left right =
                    let output = v ()
                    statements.Add(smt (SMTIntCmp(output, pred, left, right)))
                    output
                let equal left right =
                    let output = v ()
                    statements.Add(smt (SMTEq(output, left, right, SMTInt)))
                    output
                let vector values = values |> List.map constant |> List.toArray
                let matrix rows = rows |> List.map vector |> List.toArray
                let dot (left: SSA array) (right: SSA array) =
                    Array.map2 (binary SMTIntMul) left right |> Array.reduce (binary SMTIntAdd)
                let multiply (left: SSA array array) (right: SSA array array) =
                    Array.init dimension (fun row ->
                        Array.init dimension (fun column ->
                            dot left.[row] (Array.init dimension (fun k -> right.[k].[column]))))
                let zero, one, two = constant 0I, constant 1I, constant 2I
                let count = constant model.MaximumIterations
                let initialLo, initialHi = vector model.InitialLower, vector model.InitialUpper
                let coefficientLo, coefficientHi = matrix model.CoefficientLower, matrix model.CoefficientUpper
                let lower, upper = vector model.Lower, vector model.Upper
                let clauses = ResizeArray<SSA>()
                clauses.Add(compare SmtGe count zero)
                for row in 0 .. dimension - 1 do
                    clauses.Add(compare SmtLe zero initialLo.[row])
                    clauses.Add(compare SmtLe initialLo.[row] initialHi.[row])
                    clauses.Add(compare SmtLe lower.[row] zero)
                    clauses.Add(compare SmtLe initialHi.[row] (dot coefficientHi.[row] initialHi))
                    for column in 0 .. dimension - 1 do
                        clauses.Add(compare SmtLe zero coefficientLo.[row].[column])
                        clauses.Add(compare SmtLe coefficientLo.[row].[column] coefficientHi.[row].[column])
                let mutable previousExponent = zero
                let mutable previousMatrix =
                    Array.init dimension (fun row -> Array.init dimension (fun column -> if row = column then one else zero))
                for step in model.Powers do
                    let squared = multiply previousMatrix previousMatrix
                    let expected = if step.Odd then multiply squared coefficientHi else squared
                    let supplied = matrix step.Matrix
                    let exponent = constant step.Exponent
                    clauses.Add(equal exponent (binary SMTIntAdd (binary SMTIntMul two previousExponent) (if step.Odd then one else zero)))
                    for row in 0 .. dimension - 1 do
                        for column in 0 .. dimension - 1 do
                            clauses.Add(equal supplied.[row].[column] expected.[row].[column])
                    previousExponent <- exponent
                    previousMatrix <- supplied
                clauses.Add(equal previousExponent count)
                for row in 0 .. dimension - 1 do
                    clauses.Add(compare SmtLe (dot previousMatrix.[row] initialHi) upper.[row])
                let conclusion = v ()
                statements.Add(smt (SMTAnd(conclusion, List.ofSeq clauses)))
                // Malformed or false supplied certificates remain refutable;
                // no global assumption can conceal an inconsistent premise.
                anchor conclusion (List.ofSeq statements)
        | ObligationBody.MappedElementSpan model ->
            let statements = ResizeArray<MLIROp>()
            let declare name =
                let output = v ()
                statements.Add(smt (SMTDeclareFun(output, name, SMTInt)))
                output
            let constant value =
                let output = v ()
                statements.Add(smt (SMTBigIntConstant(output, value)))
                output
            let binary make left right =
                let output = v ()
                statements.Add(smt (make (output, left, right)))
                output
            let cmp pred left right =
                let output = v ()
                statements.Add(smt (SMTIntCmp(output, pred, left, right)))
                output
            let equal left right =
                let output = v ()
                statements.Add(smt (SMTEq(output, left, right, SMTInt)))
                output
            let assume value = statements.Add(smt (SMTAssert value))
            let baseAddress, bytes, index = declare "mapped_base", declare "mapped_bytes", declare "mapped_index"
            let zero, maximum = constant 0I, constant model.MaximumExtent
            let elementBytes, baseAlignment, elementAlignment = constant (bigint model.ElementBytes), constant (bigint model.BaseAlignment), constant (bigint model.ElementAlignment)
            cmp SmtGt baseAddress zero |> assume
            cmp SmtGt bytes zero |> assume
            cmp SmtLe bytes maximum |> assume
            cmp SmtLe baseAddress (binary SMTIntSub maximum bytes) |> assume
            equal (binary SMTIntMod bytes elementBytes) zero |> assume
            equal (binary SMTIntMod baseAddress baseAlignment) zero |> assume
            cmp SmtLe zero index |> assume
            cmp SmtLt index (binary SMTIntDiv bytes elementBytes) |> assume
            let start = binary SMTIntAdd baseAddress (binary SMTIntMul index elementBytes)
            let ending = binary SMTIntAdd start elementBytes
            let clauses = [cmp SmtLe baseAddress start; cmp SmtLe ending (binary SMTIntAdd baseAddress bytes)
                           cmp SmtLe ending maximum; equal (binary SMTIntMod start elementAlignment) zero]
            let conclusion = v ()
            statements.Add(smt (SMTAnd(conclusion, clauses)))
            anchor conclusion (List.ofSeq statements)
        | ObligationBody.IntegerLiteralRange (value, lower, upper) ->
            integerComparisons [ lower, value; value, upper ]
        | ObligationBody.IntegerRepresentationCoverage (lower, upper, minimum, maximum) ->
            integerComparisons [ minimum, lower; lower, upper; upper, maximum ]
        | ObligationBody.RealLiteralRange (value, lower, upper) ->
            rationalComparisons [ lower, value; value, upper ]
        | ObligationBody.RealRepresentationCoverage (lower, upper, minimum, maximum) ->
            rationalComparisons [ minimum, lower; lower, upper; upper, maximum ]
        | ObligationBody.ApplicationDimensions comparisons ->
            let statements = ResizeArray<MLIROp>()
            let constant value =
                let ssa = v ()
                statements.Add(smt (SMTIntConstant(ssa, int64 value)))
                ssa
            let equation left right =
                let l, r, output = constant left, constant right, v ()
                statements.Add(smt (SMTEq(output, l, r, SMTInt)))
                output
            let clauses =
                [ for _, expected, actual in comparisons do
                    match expected, actual with
                    | Some expected, Some actual ->
                        let coordinates project =
                            let left, right = project expected, project actual
                            Set.union (left |> Map.keys |> Set.ofSeq) (right |> Map.keys |> Set.ofSeq)
                            |> Set.toList
                            |> List.map (fun axis ->
                                (Map.tryFind axis left |> Option.defaultValue 0),
                                (Map.tryFind axis right |> Option.defaultValue 0))
                        let axes = coordinates (fun d -> d.Bases) @ coordinates (fun d -> d.Vars)
                        match axes with
                        | [] -> yield equation 0 0 // Both dimensions are explicitly dimensionless.
                        | _ ->
                            for left, right in axes do
                                yield equation left right
                    | _ -> yield equation 0 1 ] // Missing evidence cannot establish compatibility.
            let definition =
                match clauses with
                | [] -> equation 0 1
                | [single] -> single
                | _ ->
                    let conjunction = v ()
                    statements.Add(smt (SMTAnd(conjunction, clauses)))
                    conjunction
            anchor definition (List.ofSeq statements)
        | ObligationBody.DimensionalRelation (rule, left, right, result) ->
            let dimensions = left :: right :: Option.toList result
            let coordinates project =
                dimensions |> List.collect (project >> Map.toList >> List.map fst)
                |> Set.ofList |> Set.toList
                |> List.map (fun axis ->
                    let exponent dim = project dim |> Map.tryFind axis |> Option.defaultValue 0
                    exponent left, exponent right, Option.map exponent result)
            let axes = coordinates (fun d -> d.Bases) @ coordinates (fun d -> d.Vars)
            let statements = ResizeArray<MLIROp>()
            let constant value =
                let ssa = v ()
                statements.Add(smt (SMTIntConstant(ssa, int64 value)))
                ssa
            let equation left right =
                let ssa = v ()
                statements.Add(smt (SMTEq(ssa, left, right, SMTInt)))
                ssa
            let clauses =
                [ for left, right, output in axes do
                    let l, r = constant left, constant right
                    match rule, output with
                    | DimensionalRule.Product, Some output
                    | DimensionalRule.Quotient, Some output ->
                        let expected = v ()
                        statements.Add(smt (if rule = DimensionalRule.Product then SMTIntAdd(expected, l, r) else SMTIntSub(expected, l, r)))
                        yield equation (constant output) expected
                    | DimensionalRule.SameDimension, Some output ->
                        yield equation l r
                        yield equation (constant output) l
                    | DimensionalRule.Comparison, None -> yield equation l r
                    | _ -> () ]
            let definition =
                if (rule = DimensionalRule.Comparison) <> Option.isNone result then
                    equation (constant 0) (constant 1)
                else
                    match clauses with
                    | [] -> equation (constant 0) (constant 0)
                    | [single] -> single
                    | _ ->
                        let conjunction = v ()
                        statements.Add(smt (SMTAnd(conjunction, clauses)))
                        conjunction
            anchor definition (List.ofSeq statements)
        | ObligationBody.StorageReservation (len, storage) ->
            let s = v ()
            let l = v ()
            let one = v ()
            let sum = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (s, int64 storage))
                  smt (SMTIntConstant (l, int64 len))
                  smt (SMTIntConstant (one, 1L))
                  smt (SMTIntAdd (sum, l, one))
                  smt (SMTEq (def, s, sum, SMTInt)) ]
        | ObligationBody.ViewContainment (view, len, storage) ->
            let vw = v ()
            let l = v ()
            let s = v ()
            let eq = v ()
            let lt = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (vw, int64 view))
                  smt (SMTIntConstant (l, int64 len))
                  smt (SMTIntConstant (s, int64 storage))
                  smt (SMTEq (eq, vw, l, SMTInt))
                  smt (SMTIntCmp (lt, SmtLt, vw, s))
                  smt (SMTAnd (def, [eq; lt])) ]
        | ObligationBody.NulSentinel lastByte ->
            let b = v ()
            let z = v ()
            let def = v ()
            anchor def
                [ smt (SMTBVConstant (b, int64 lastByte, 8))
                  smt (SMTBVConstant (z, 0L, 8))
                  smt (SMTEq (def, b, z, SMTBV 8)) ]
        | ObligationBody.StaticStorageLayout (slots, usedSize, allocationSize, poolAlignment, capacity, spaceAlignment, granularity) ->
            // Every placement is a concrete compiler fact. No adjacency premise
            // assumes the layout that this claim is meant to establish.
            let statements = ResizeArray<MLIROp>()
            let constant (value: bigint) =
                let result = v ()
                statements.Add(smt (SMTBigIntConstant(result, value)))
                result
            let cmp pred a b =
                let result = v ()
                statements.Add(smt (SMTIntCmp(result, pred, a, b)))
                result
            let equal a b =
                let result = v ()
                statements.Add(smt (SMTEq(result, a, b, SMTInt)))
                result
            let any clauses =
                match clauses with
                | [only] -> only
                | _ ->
                    let result = v ()
                    statements.Add(smt (SMTOr(result, clauses)))
                    result
            let zero = constant 0I
            let divisible value divisor =
                // Invalid zero/negative alignments fail a separate conjunct;
                // use a defined modulus so they cannot introduce vacuity.
                let denominator = constant (bigint (max 1 divisor))
                let remainder = v ()
                statements.Add(smt (SMTIntMod(remainder, value, denominator)))
                equal remainder zero
            let used, allocation = constant (bigint usedSize), constant (bigint allocationSize)
            let pool, space, grain, cap = constant (bigint poolAlignment), constant (bigint spaceAlignment), constant (bigint granularity), constant (bigint capacity)
            let clauses = ResizeArray<SSA>()
            for term in [pool; space; grain] do clauses.Add(cmp SmtGt term zero)
            clauses.Add(cmp SmtGe used zero)
            clauses.Add(cmp SmtGe cap zero)
            clauses.Add(cmp SmtLe used allocation)
            clauses.Add(cmp SmtLe allocation cap)
            clauses.Add(cmp SmtGe pool space)
            clauses.Add(divisible allocation granularity)
            clauses.Add(divisible pool spaceAlignment)
            let placements = slots |> List.map (fun (offset, length, alignment) ->
                let beginAt = constant (bigint offset)
                let lengthTerm = constant (bigint length)
                let align = constant (bigint alignment)
                let endAt = v ()
                statements.Add(smt (SMTIntAdd(endAt, beginAt, lengthTerm)))
                clauses.Add(cmp SmtGe beginAt zero)
                clauses.Add(cmp SmtGt lengthTerm zero)
                clauses.Add(cmp SmtGt align zero)
                clauses.Add(divisible beginAt alignment)
                clauses.Add(divisible pool alignment)
                clauses.Add(cmp SmtLe endAt allocation)
                clauses.Add(cmp SmtLe endAt used)
                beginAt, endAt)
            match placements with
            | [] -> clauses.Add(equal used zero)
            | _ -> clauses.Add(placements |> List.map (fun (_, ending) -> equal ending used) |> any)
            for i, (leftBegin, leftEnd) in List.indexed placements do
                for rightBegin, rightEnd in List.skip (i + 1) placements do
                    clauses.Add(any [cmp SmtLe leftEnd rightBegin; cmp SmtLe rightEnd leftBegin])
            let definition = v ()
            statements.Add(smt (SMTAnd(definition, List.ofSeq clauses)))
            anchor definition (List.ofSeq statements)
        | ObligationBody.ContinuationLayout (slots, extent, alignment) ->
            let statements = ResizeArray<MLIROp>()
            let constant value =
                let result = v ()
                statements.Add(smt (SMTBigIntConstant(result, value)))
                result
            let binary make left right =
                let result = v ()
                statements.Add(smt (make (result, left, right)))
                result
            let cmp predicate left right =
                let result = v ()
                statements.Add(smt (SMTIntCmp(result, predicate, left, right)))
                result
            let equal left right =
                let result = v ()
                statements.Add(smt (SMTEq(result, left, right, SMTInt)))
                result
            let any values =
                let result = v ()
                statements.Add(smt (SMTOr(result, values)))
                result
            let zero, one = constant 0I, constant 1I
            let powerOfTwo value =
                [0 .. 30] |> List.map (fun exponent -> equal value (constant (1I <<< exponent))) |> any
            let aligned endpoint required =
                // Nonpositive alignments fail their own clause; use a defined
                // divisor while transcribing the exact rounding equation.
                let divisor = constant (bigint (max 1 required))
                let adjustment = binary SMTIntSub divisor one
                let rounded = binary SMTIntDiv (binary SMTIntAdd endpoint adjustment) divisor
                binary SMTIntMul rounded divisor
            let extentTerm, alignmentTerm = constant (bigint extent), constant (bigint alignment)
            let clauses = ResizeArray<SSA>()
            clauses.Add(cmp SmtGe extentTerm zero)
            clauses.Add(powerOfTwo alignmentTerm)
            clauses.Add(cmp SmtGe alignmentTerm one)
            let mutable previous = zero
            let placements = slots |> List.map (fun (offset, bytes, alignment) ->
                let beginAt, length, required = constant (bigint offset), constant (bigint bytes), constant (bigint alignment)
                let endAt = binary SMTIntAdd beginAt length
                clauses.Add(cmp SmtGe beginAt zero)
                clauses.Add(cmp SmtGt length zero)
                clauses.Add(powerOfTwo required)
                clauses.Add(cmp SmtGe alignmentTerm required)
                let divisor = constant (bigint (max 1 alignment))
                clauses.Add(equal (binary SMTIntMod beginAt divisor) zero)
                clauses.Add(equal (binary SMTIntMod alignmentTerm divisor) zero)
                clauses.Add(equal beginAt (aligned previous alignment))
                clauses.Add(cmp SmtLe endAt extentTerm)
                previous <- endAt
                beginAt, endAt, required)
            clauses.Add(any (equal alignmentTerm one :: (placements |> List.map (fun (_, _, required) -> equal alignmentTerm required))))
            clauses.Add(equal extentTerm (aligned previous alignment))
            for index, (leftBegin, leftEnd, _) in List.indexed placements do
                for rightBegin, rightEnd, _ in List.skip (index + 1) placements do
                    clauses.Add(any [cmp SmtLe leftEnd rightBegin; cmp SmtLe rightEnd leftBegin])
            let definition = v ()
            statements.Add(smt (SMTAnd(definition, List.ofSeq clauses)))
            anchor definition (List.ofSeq statements)
        | ObligationBody.ConsecutiveLayout (storages, span, capacity) ->
            let sizes = List.toArray storages
            let n' = sizes.Length
            let bases = [ for i in 0 .. n' - 1 -> v (), sprintf "b%d" i ]
            let baseSSA i = fst bases[i]
            let declOps =
                [ for (ssa, name) in bases -> smt (SMTDeclareFun (ssa, name, SMTInt)) ]
            // b0 >= 0
            let zero = v ()
            let nn = v ()
            let nonNeg =
                [ smt (SMTIntConstant (zero, 0L))
                  smt (SMTIntCmp (nn, SmtGe, baseSSA 0, zero))
                  smt (SMTAssert nn) ]
            // adjacency premises: b[i] = b[i-1] + size[i-1]
            let adjacency =
                [ for i in 1 .. n' - 1 do
                    let c = v ()
                    let a = v ()
                    let eq = v ()
                    yield smt (SMTIntConstant (c, int64 sizes[i-1]))
                    yield smt (SMTIntAdd (a, baseSSA (i-1), c))
                    yield smt (SMTEq (eq, baseSSA i, a, SMTInt))
                    yield smt (SMTAssert eq) ]
            // pairwise disjointness: end_i <= b_j OR end_j <= b_i
            let mutable disjSSAs = []
            let disjoint =
                [ for i in 0 .. n' - 1 do
                    for j in i + 1 .. n' - 1 do
                        let si = v ()
                        let ei = v ()
                        let li = v ()
                        let sj = v ()
                        let ej = v ()
                        let lj = v ()
                        let d = v ()
                        disjSSAs <- disjSSAs @ [d]
                        yield smt (SMTIntConstant (si, int64 sizes[i]))
                        yield smt (SMTIntAdd (ei, baseSSA i, si))
                        yield smt (SMTIntCmp (li, SmtLe, ei, baseSSA j))
                        yield smt (SMTIntConstant (sj, int64 sizes[j]))
                        yield smt (SMTIntAdd (ej, baseSSA j, sj))
                        yield smt (SMTIntCmp (lj, SmtLe, ej, baseSSA i))
                        yield smt (SMTOr (d, [li; lj])) ]
            // span: b[n-1] + size[n-1] = b[0] + span
            let lastSz = v ()
            let hi = v ()
            let spanC = v ()
            let lo = v ()
            let spanEq = v ()
            let def = v ()
            // smt.and is variadic with a two-operand minimum: with exactly one
            // disjointness pair (two strings), use that pair's SSA directly
            let allDisj, allDisjOps =
                match disjSSAs with
                | [only] -> only, []
                | _ -> let a = v () in a, [ smt (SMTAnd (a, disjSSAs)) ]
            // where a declared space bounds the layout: span <= capacity
            let fitsSSAs, fitsOps =
                match capacity with
                | Some cap ->
                    let c = v ()
                    let f = v ()
                    [f], [ smt (SMTIntConstant (c, cap)); smt (SMTIntCmp (f, SmtLe, spanC, c)) ]
                | None -> [], []
            let spanOps =
                [ smt (SMTIntConstant (lastSz, int64 sizes[n' - 1]))
                  smt (SMTIntAdd (hi, baseSSA (n' - 1), lastSz))
                  smt (SMTIntConstant (spanC, int64 span))
                  smt (SMTIntAdd (lo, baseSSA 0, spanC))
                  smt (SMTEq (spanEq, hi, lo, SMTInt)) ]
                @ allDisjOps
                @ fitsOps
                @ [ smt (SMTAnd (def, [allDisj; spanEq] @ fitsSSAs)) ]
            anchor def (declOps @ nonNeg @ adjacency @ disjoint @ spanOps)
        | ObligationBody.ConcatCopyBound (leftLen, rightLen) ->
            let ll = v ()
            let lr = v ()
            let al = v ()
            let zero = v ()
            let geL = v ()
            let geR = v ()
            let decls =
                [ smt (SMTDeclareFun (ll, "len_l", SMTInt))
                  smt (SMTDeclareFun (lr, "len_r", SMTInt))
                  smt (SMTDeclareFun (al, "alloc", SMTInt))
                  smt (SMTIntConstant (zero, 0L))
                  smt (SMTIntCmp (geL, SmtGe, ll, zero))
                  smt (SMTAssert geL)
                  smt (SMTIntCmp (geR, SmtGe, lr, zero))
                  smt (SMTAssert geR) ]
            // pin an operand to its concrete byte length where the graph has it
            let pin (ssa: SSA) (len: int option) : MLIROp list =
                match len with
                | Some n ->
                    let c = v ()
                    let eq = v ()
                    [ smt (SMTIntConstant (c, int64 n))
                      smt (SMTEq (eq, ssa, c, SMTInt))
                      smt (SMTAssert eq) ]
                | None -> []
            // emission-contract premise: alloc = len_l + len_r
            let sum = v ()
            let aeq = v ()
            let premise =
                [ smt (SMTIntAdd (sum, ll, lr))
                  smt (SMTEq (aeq, al, sum, SMTInt))
                  smt (SMTAssert aeq) ]
            // conclusion: both copy windows lie within the allocation
            let c1 = v ()
            let c2 = v ()
            let def = v ()
            let conclusion =
                [ smt (SMTIntCmp (c1, SmtLe, ll, al))
                  smt (SMTIntCmp (c2, SmtLe, sum, al))
                  smt (SMTAnd (def, [c1; c2])) ]
            anchor def (decls @ pin ll leftLen @ pin lr rightLen @ premise @ conclusion)

        | ObligationBody.CapacityPositive cap ->
            let c = v ()
            let zero = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (c, cap))
                  smt (SMTIntConstant (zero, 0L))
                  smt (SMTIntCmp (def, SmtGt, c, zero)) ]
        | ObligationBody.CapacityFits (cap, spaceCap) ->
            let c = v ()
            let sc = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (c, cap))
                  smt (SMTIntConstant (sc, spaceCap))
                  smt (SMTIntCmp (def, SmtLe, c, sc)) ]
        | ObligationBody.InputBufferBound (count, allocation) ->
            let c = v ()
            let a = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (c, count))
                  smt (SMTIntConstant (a, allocation))
                  smt (SMTIntCmp (def, SmtLe, c, a)) ]
        | ObligationBody.InputCopyBound (cap, bound) ->
            // for all r, 1 <= r <= cap: r - 1 <= bound
            let r = v ()
            let one = v ()
            let capC = v ()
            let geOne = v ()
            let leCap = v ()
            let rm1 = v ()
            let b = v ()
            let def = v ()
            anchor def
                [ smt (SMTDeclareFun (r, "r", SMTInt))
                  smt (SMTIntConstant (one, 1L))
                  smt (SMTIntCmp (geOne, SmtGe, r, one))
                  smt (SMTAssert geOne)
                  smt (SMTIntConstant (capC, cap))
                  smt (SMTIntCmp (leCap, SmtLe, r, capC))
                  smt (SMTAssert leCap)
                  smt (SMTIntSub (rm1, r, one))
                  smt (SMTIntConstant (b, bound))
                  smt (SMTIntCmp (def, SmtLe, rm1, b)) ]

    let logic, transformation =
        match ob.Body with
        | ObligationBody.RealLiteralRange _ | ObligationBody.RealRepresentationCoverage _ ->
            "QF_LIA", [ MLIROp.RawMLIR "// Exact rational bounds: QF_LRA comparisons with positive denominators cleared to QF_LIA." ]
        | _ -> ob.Logic, []
    [ MLIROp.RawMLIR (sprintf "// %s: %s" ob.Id ob.Statement)
      MLIROp.RawMLIR (sprintf "// origin: %s" ob.Source)
    ] @ transformation @ [ smt (SMTSolver (smt (SMTSetLogic logic) :: ops @ [ smt SMTCheck ])) ]

/// The verification module from the graph's obligations.
/// Pure function: ObligationInfo list -> MLIR text. The list is
/// ObligationDischarge.ofGraph, the same list the design-time dispatch rendered.
let transfer (obs: ObligationInfo list) : string =
    obs
    |> List.collect scope
    |> moduleToString (Error "the obligations module carries no pointer-sized type") "obligations"
