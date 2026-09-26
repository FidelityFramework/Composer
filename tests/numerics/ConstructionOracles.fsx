// Independent host-side reference quantities for F-11 / C-08 fixtures.
// This script does not run Composer and cannot certify a compiler gate.
open System
open System.Numerics

[<StructuralEquality; StructuralComparison>]
type Rational = private { Numerator: bigint; Denominator: bigint }

let rational (n: bigint) (d: bigint) =
    if d = 0I then invalidArg "d" "A reference quantity needs a nonzero denominator"
    let sign = if d < 0I then -1I else 1I
    let divisor = BigInteger.GreatestCommonDivisor(n, d)
    { Numerator = sign * n / divisor; Denominator = sign * d / divisor }

let whole n = rational n 1I
let zero = whole 0I
let one = whole 1I
let add a b = rational (a.Numerator*b.Denominator + b.Numerator*a.Denominator) (a.Denominator*b.Denominator)
let negate a = rational -a.Numerator a.Denominator
let subtract a b = add a (negate b)
let multiply a b = rational (a.Numerator*b.Numerator) (a.Denominator*b.Denominator)
let compareValue a b = compare (a.Numerator*b.Denominator) (b.Numerator*a.Denominator)
let powerOfTwo exponent =
    if exponent >= 0 then whole (1I <<< exponent)
    else rational 1I (1I <<< -exponent)

// Decode finite binary64 from its representation, without decimal conversion.
let representedBinary64 (value: double) =
    let bits = BitConverter.DoubleToUInt64Bits value
    let sign = if bits >>> 63 = 0UL then 1I else -1I
    let exponent = int ((bits >>> 52) &&& 0x7ffUL)
    let fraction = bigint (bits &&& 0x000fffffffffffffUL)
    if exponent = 0x7ff then invalidArg "value" "This oracle's domain is finite values"
    if exponent = 0 then multiply (whole (sign*fraction)) (powerOfTwo -1074)
    else multiply (whole (sign*((1I <<< 52) + fraction))) (powerOfTwo (exponent-1023-52))

type Rounding = NearestEven | TowardZero | TowardNegative | TowardPositive

let roundInteger mode (value: Rational) =
    let q = value.Numerator / value.Denominator
    let r = value.Numerator % value.Denominator
    let direction = bigint value.Numerator.Sign
    match mode with
    | TowardZero -> q
    | TowardNegative -> if r < 0I then q - 1I else q
    | TowardPositive -> if r > 0I then q + 1I else q
    | NearestEven ->
        match compare (2I * BigInteger.Abs r) value.Denominator with
        | n when n < 0 -> q
        | 0 when q.IsEven -> q
        | _ -> q + direction

let quantize mode step value =
    if step.Numerator <= 0I then invalidArg "step" "A quantization step must be positive"
    let count = rational (value.Numerator * step.Denominator) (value.Denominator * step.Numerator)
    multiply (whole (roundInteger mode count)) step

// Exact nearest-even reference for finite binary64 results, including subnormals.
// Overflow and special-value observations require distinct fixture contracts.
let binary64 value =
    if value = zero then zero
    else
        let magnitude = rational (BigInteger.Abs value.Numerator) value.Denominator
        let estimate = int (magnitude.Numerator.GetBitLength() - magnitude.Denominator.GetBitLength())
        let exponent = if compareValue magnitude (powerOfTwo estimate) < 0 then estimate - 1 else estimate
        let rounded = quantize NearestEven (powerOfTwo (max -1074 (exponent - 52))) value
        let maximum = subtract (powerOfTwo 1024) (powerOfTwo 971)
        if compareValue (rational (BigInteger.Abs rounded.Numerator) rounded.Denominator) maximum > 0 then
            invalidArg "value" "Finite-result oracle overflow"
        rounded

let mutable checkedCount = 0
let equal name expected actual =
    if actual <> expected then failwithf "%s: expected %A; received %A" name expected actual
    checkedCount <- checkedCount + 1
let different name left right =
    if left = right then failwithf "%s: mutation did not distinguish the contracts" name
    checkedCount <- checkedCount + 1

// Basic decoding/rounding controls anchor the independent arithmetic model.
equal "binary64 one" one (representedBinary64 1.0)
equal "binary64 exact tenth encoding" (rational 3602879701896397I 36028797018963968I) (representedBinary64 0.1)
equal "binary64 minimum subnormal" (powerOfTwo -1074) (representedBinary64 Double.Epsilon)
equal "binary64 maximum finite" (subtract (powerOfTwo 1024) (powerOfTwo 971)) (representedBinary64 Double.MaxValue)
equal "half minimum subnormal ties to even zero" zero (binary64 (powerOfTwo -1075))
equal "three half subnormals ties to even second" (multiply (whole 2I) (powerOfTwo -1074)) (binary64 (multiply (whole 3I) (powerOfTwo -1075)))
equal "negative subnormal rounding" (negate (powerOfTwo -1074)) (binary64 (negate (multiply (rational 3I 4I) (powerOfTwo -1074))))
equal "one midpoint ties to even" one (binary64 (add one (powerOfTwo -53)))

// C08a01: preserving the rounded source order differs from exact accumulation.
let large = powerOfTwo 54
let terms = [large; one; negate large]
let sequential = terms |> List.fold (fun state term -> binary64 (add state term)) zero
let exact = terms |> List.fold add zero |> binary64
equal "C08a01 sequential rounded" zero sequential
equal "C08a01 exact represented terms" one exact
different "C08a01 illegal replacement control" sequential exact

// C08a02: exact products and rounded-product terms have different meanings.
let a = add one (powerOfTwo -27)
let b = subtract one (powerOfTwo -27)
equal "C08a02 inputs exactly represented" (a, b) (binary64 a, binary64 b)
let exactProductSum = subtract (multiply a b) one |> binary64
let roundedProductSum = subtract (binary64 (multiply a b)) one |> binary64
equal "C08a02 exact products" (negate (powerOfTwo -54)) exactProductSum
equal "C08a02 rounded products" zero roundedProductSum
different "C08a02 contraction changes contract" exactProductSum roundedProductSum

// C08a03: error-free TwoSum on named finite, non-overflowing fixtures.
let twoSum x y =
    let sum = binary64 (add x y)
    let virtualY = binary64 (subtract sum x)
    let virtualX = binary64 (subtract sum virtualY)
    let residualY = binary64 (subtract y virtualY)
    let residualX = binary64 (subtract x virtualX)
    sum, binary64 (add residualX residualY)
for x, y in [powerOfTwo 53, one; representedBinary64 0.1, representedBinary64 0.2; large, negate large] do
    let sum, residual = twoSum x y
    equal "C08a03 TwoSum denotation" (add x y) (add sum residual)
let sum, residual = twoSum (powerOfTwo 53) one
equal "C08a03 nonzero residual" one residual
different "C08a03 dropping residual" (add sum residual) sum

// F11c02: scale changes are arithmetic operations with independent fidelity.
equal "F11c02 exact aligned addition" (rational 7I 16I) (add (rational 3I 8I) (rational 1I 16I))
let product = multiply (rational 3I 8I) (rational 5I 16I)
equal "F11c02 exact product" (rational 15I 128I) product
equal "F11c02 rescale nearest" (rational 1I 8I) (quantize NearestEven (rational 1I 16I) product)
equal "F11c02 rescale error" (rational 1I 128I) (subtract (quantize NearestEven (rational 1I 16I) product) product)
equal "F11c02 positive even tie" (whole 2I) (quantize NearestEven one (rational 5I 2I))
equal "F11c02 negative even tie" (whole -2I) (quantize NearestEven one (rational -5I 2I))
equal "F11c02 negative floor" (whole -3I) (quantize TowardNegative one (rational -5I 2I))
equal "F11c02 negative ceiling" (whole -2I) (quantize TowardPositive one (rational -5I 2I))
equal "F11c02 truncate toward zero" (whole -2I) (quantize TowardZero one (rational -5I 2I))

// C08b01/b02: partitions preserve term multiplicity and the initial value once;
// capacity constrains intermediates even when the final result cancels.
let initial = 7I
let integerTerms = [10I; 20I; -30I; 4I]
let expectedSum = initial + List.sum integerTerms
for split in 0 .. integerTerms.Length do
    let left, right = List.splitAt split integerTerms
    equal "C08b01 partition and initial once" expectedSum (initial + List.sum left + List.sum right)
different "C08b01 duplicate initialization" expectedSum (2I*initial + List.sum integerTerms)
let capacityFits bound values = values |> List.scan (+) 0I |> List.forall (fun x -> BigInteger.Abs x <= bound)
equal "C08b02 fitting final cancellation" 0I (List.sum [7I; 7I; -7I; -7I])
equal "C08b02 overflowing partial" false (capacityFits 10I [7I; 7I; -7I; -7I])
equal "C08b02 fitting alternate order" true (capacityFits 10I [7I; -7I; 7I; -7I])

// C08b05: exact-state transfer cannot silently round a partial before merging.
let partial = add large one
equal "C08b05 exact state transport" one (subtract partial large)
equal "C08b05 rounded partial loses one" zero (subtract (binary64 partial) large)

printfn "PASS %d independent reference checks for F11c02 / C08a01-a03 / C08b01-b02,b05" checkedCount
printfn "Compiler gates were not run by this reference-oracle script."
