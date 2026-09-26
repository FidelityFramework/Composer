/// Native mapping requires a source-published acquisition/release ABI and actual callback contract.
module Alex.Patterns.MappedViewPatterns

open XParsec
open XParsec.Parsers
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

let pMappedCall : PSGParser<MLIROp list * TransferResult> =
    fail (Message "Native mapping requires a source-published acquisition/release ABI and canonical callback code/environment contract")
