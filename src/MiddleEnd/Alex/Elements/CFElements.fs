/// Atomic portable control-flow dialect operations. Region construction stays
/// in structured-control Patterns; this operation checks a witnessed condition.
module internal Alex.Elements.CFElements

open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types

let pAssert (condition: SSA) (diagnostic: string) : PSGParser<MLIROp> =
    parser { return MLIROp.Assert(condition, diagnostic) }
