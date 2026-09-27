/// Atomic inert portable declaration of a complete source spatial plan.
module internal Alex.Elements.SpatialElements

open XParsec.Parsers
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

let pSpatialModule (plan: SpatialModuleWitness) : PSGParser<MLIROp> = preturn (MLIROp.SpatialModule plan)
