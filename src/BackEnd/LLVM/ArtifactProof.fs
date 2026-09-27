/// The artifact clerk checks the current linked image against admitted source
/// claims. It does not author source semantics or infer a missing memory plan.
module BackEnd.LLVM.ArtifactProof

open System
open System.Diagnostics
open System.Globalization
open System.IO
open System.Security.Cryptography
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Core.Types.Pipeline
open Core.Types.WitnessArtifacts

module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission
module Catalog = Core.WitnessArtifacts
module Names = Alex.Traversal.Values

let private require condition message = if not condition then invalidOp message
let private need = function Ok value -> value | Error reason -> invalidOp reason
let private only message = function [value] -> value | _ -> invalidOp message
let private sha (bytes: byte array) = SHA256.HashData bytes |> Convert.ToHexString
let private fileHash path = use stream = File.OpenRead path in SHA256.HashData stream |> Convert.ToHexString
let private textHash (text: string) = Encoding.UTF8.GetBytes text |> sha
let private number (value: bigint) = value.ToString(CultureInfo.InvariantCulture)

/// Only a complete current catalog and both actual dispatch receipts can
/// construct this plan. Detached ELF facts cannot authorize native publication.
type WriteOrigin = { Site: NodeId; Origin: NodeId; Entry: int; Count: bigint; Extent: bigint; Anchor: string }
type private Claim =
    | Storage of int | View of int | Sentinel of int | Layout
    | Borrow of WriteOrigin list

type Plan = private {
    Input: BackEndInput
    Source: SourceProofReceipt
    Mlir: MlirProofReceipt
    Pool: StaticStringPool option
    Claims: (ObligationInfo * Claim) list
    SolverClaims: ObligationInfo list
    Writes: WriteOrigin list
}

let private sameKeys (left: Map<_,_>) (right: Map<_,_>) = Set.ofSeq left.Keys = Set.ofSeq right.Keys

/// Inventory checks read published aliases/formals and exact typed operations.
/// They do not follow source syntax to discover a replacement source meaning.
let prepare (input: BackEndInput) : Result<Plan,string> =
    try
        WitnessedInput.validate input |> need
        let catalog = input.Catalog.Value
        let proof = catalog.Proof |> Option.defaultWith (fun () -> invalidOp "Artifact proof requires the current source and MLIR receipts, including an empty obligation inventory")
        Catalog.validateProof catalog.Scope proof |> need
        let source = proof.Source |> Option.defaultWith (fun () -> invalidOp "Artifact proof has no current PSG receipt")
        let mlir = proof.Mlir |> Option.defaultWith (fun () -> invalidOp "Artifact proof has no current MLIR discharge receipt")
        let graph = Catalog.graph catalog.Scope
        let publication = Publication.tryRead graph |> need
        let boundary, callable, memory, storage = publication.Boundary, publication.Callable, publication.Memory, publication.Storage
        require (catalog.Units |> List.forall (fun unit -> not unit.HasOpaqueTargetContent)) "Unsupported artifact family: opaque source target content"
        require input.WritableStorage.IsEmpty "Unsupported artifact family: writable program storage"
        require (storage.Lazies.IsEmpty && storage.Sequences.IsEmpty && storage.ProgramStorage.Entries.IsEmpty)
            "Unsupported artifact family: lazy, sequence or program allocation"
        for KeyValue(site, operation) in memory.Operations do
            let unsupported=
                match operation with
                | MemoryWitnessOperation.BufferExtent _ -> None
                | MemoryWitnessOperation.ArrayExtent _ -> Some "array extent"
                | MemoryWitnessOperation.ArrayAccess _ -> Some "guarded array access"
                | MemoryWitnessOperation.ArrayLiteral _ -> Some "array residence and initialization"
                | MemoryWitnessOperation.ArrayAllocation _ -> Some "array allocation and initialization"
                | MemoryWitnessOperation.Address _ -> Some "actual-place address"
                | MemoryWitnessOperation.StringView _ -> Some "constructed string snapshot and byte view"
            unsupported |> Option.iter (fun family -> invalidOp (sprintf "Unsupported required artifact memory operation at %d: %s" (NodeId.value site) family))
        Alex.Traversal.StaticStorageValidation.validate graph input.Operations |> need
        let pool = graph.StaticStringPool
        let entries = pool |> Option.map _.Entries |> Option.defaultValue []
        let origins = entries |> List.mapi (fun index entry -> entry.NodeIds |> List.map (fun node -> node,index)) |> List.concat
        require ((origins |> List.map fst |> Set.ofList).Count=origins.Length)
            "Static pool contains a duplicated source origin identity"
        let originEntries = Map.ofList origins
        let obligationNodes = graph.Nodes |> Map.toList |> List.choose (fun (id,node) ->
            match node.Kind with SemanticKind.Obligation info when List.contains info proof.Obligations -> Some(id,info) | _ -> None)
        let infoAt id = obligationNodes |> List.choose (fun (node,info) -> if node=id then Some info else None) |> only "Published proof has no unique current obligation citizen"
        let all = input.Operations |> List.collect (Catalog.flatten >> Seq.toList)
        let definitions = catalog.Units |> List.collect _.Definitions
        let expectedMemory = ResizeArray<MemRefOp>()
        for entry in entries do
            let pool=pool.Value
            let storageType=TMemRefStatic(pool.Size,TInt(IntWidth 8))
            let contentType=TMemRefStatic(entry.Length,TInt(IntWidth 8))
            let viewType=TMemRef(TInt(IntWidth 8))
            for origin in entry.NodeIds do
                let value ordinal=Names.value origin ordinal
                expectedMemory.Add(MemRefOp.GetGlobal(value 0,pool.Symbol,storageType))
                expectedMemory.Add(MemRefOp.View(value 1,value 0,value 3,storageType,contentType))
                expectedMemory.Add(MemRefOp.Cast(value 2,value 1,contentType,viewType))
        let mutable expectedDimensions : Map<NodeId,MemRefOp> = Map.empty
        let functions = definitions |> List.choose (fun row ->
            match row.Operation with
            | MLIROp.FuncOp(FuncDef(_,_,_,body,_)) -> Some(row.Occurrence.Focus.Id,body |> List.collect (Catalog.flatten >> Seq.toList))
            | _ -> None)
        let scalarName owner id = Names.resultOf Core.Types.Dialects.TargetPlatform.CPU graph [owner] id
        let bufferName owner id =
            let target = callable.AliasTargets.TryFind id |> Option.defaultWith (fun () -> invalidOp "Artifact operand has no published alias endpoint")
            match callable.Arguments.TryFind owner |> Option.bind (Map.tryFind target) with
            | Some [ordinal] -> Arg ordinal
            | Some _ -> invalidOp "Unsupported artifact buffer formal components"
            | None when originEntries.ContainsKey target -> Names.value target 2
            | None ->
                let owners=callable.Arguments |> Map.toList |> List.choose (fun (owner,arguments) -> if arguments.ContainsKey target then Some(NodeId.value owner) else None)
                invalidOp (sprintf "Unsupported artifact buffer operand %d (published target %d, definition owner %d, formal owners %A): no published local formal or static-pool origin"
                    (NodeId.value id) (NodeId.value target) (NodeId.value owner) owners)
        let adaptation owner site operand expected (body: MLIROp list) =
            let raw = scalarName owner operand
            match expected, Alex.Traversal.TransferTypes.meetFor graph site operand with
            | None,None -> raw
            | Some expected,Some(actual,destination) when expected=actual ->
                let fromType,toType = TInt(IntWidth actual.From),TInt(IntWidth actual.To)
                let operation =
                    match actual.Adapt with
                    | MeetKind.ExtendUnsigned -> ArithOp.ExtUI(destination,raw,fromType,toType)
                    | MeetKind.ExtendSigned -> ArithOp.ExtSI(destination,raw,fromType,toType)
                    | MeetKind.Truncate -> ArithOp.TruncI(destination,raw,fromType,toType)
                    | _ -> invalidOp "Unsupported artifact write adaptation"
                require (body |> List.filter ((=) (MLIROp.ArithOp operation)) |> List.length = 1)
                    "Artifact write lost its exact canonical source adaptation"
                destination
            | _ -> invalidOp "Artifact write adaptation disagrees with source publication"
        require (sameKeys boundary.IntrinsicWrites boundary.IntrinsicWriteProofs) "Artifact write proof inventory is incomplete"
        let writes = boundary.IntrinsicWrites |> Map.toList |> List.collect (fun (site,call) ->
            let pool = pool |> Option.defaultWith (fun () -> invalidOp "Required bounded write has no static string pool")
            let declaration = boundary.IntrinsicWriteImports[call.Import]
            let view = boundary.ByteViews.TryFind call.Buffer |> Option.defaultWith (fun () -> invalidOp "Write has no published byte view")
            let extent = boundary.StringExtents.TryFind call.Count |> Option.defaultWith (fun () -> invalidOp "Write has no published string extent")
            require (view.ExtentSource=extent.ExtentSource && view.StaticOrigins=extent.StaticOrigins && not view.StaticOrigins.IsEmpty)
                "Write buffer/count do not retain the same complete source origin domain"
            let candidates = functions |> List.collect (fun (owner,body) -> body |> List.choose (function
                | MLIROp.FuncOp(FuncCall(results,symbol,args)) when symbol=declaration.Symbol && (results |> List.exists (fun value -> value.SSA=scalarName owner site)) -> Some(owner,body,results,args)
                | _ -> None))
            let owner,body,results,args = candidates |> only "Required write has no unique typed call occurrence"
            let fd = adaptation owner site call.Fd call.FdAdaptation body
            let count = adaptation owner site call.Count call.CountAdaptation body
            let buffer = bufferName owner view.Source
            let expected = List.zip [fd;buffer;count] (IntrinsicWriteAbi.parameters declaration) |> List.map (fun (ssa,ty) -> {SSA=ssa;Type=ty})
            require (args=expected && results=[{SSA=scalarName owner site;Type=BoundaryAbi.scalarType declaration.Result}])
                "Artifact write fd/buffer/count/result differ from the exact ordered source operands"
            adaptation owner site site call.ResultAdaptation body |> ignore
            // The actual dimension operation must read that same descriptor.
            let extentOperation = memory.Operations.TryFind call.Count |> Option.defaultWith (fun () -> invalidOp "Write count has no published memory extent")
            let extentFact = match extentOperation with MemoryWitnessOperation.BufferExtent fact -> fact | _ -> invalidOp "Unsupported write count operation"
            require (bufferName owner extentFact.Source=buffer) "Write count measures a different source descriptor"
            let dimensions = body |> List.choose (function
                | MLIROp.MemRefOp(MemRefOp.Dim(result,source,index,ty)) when result=Names.value call.Count 1 -> Some(source,index,ty)
                | _ -> None)
            require (dimensions=[buffer,Names.value call.Count 0,IntrinsicWriteAbi.bufferType declaration]) "Write count lost its actual descriptor dimension correspondence"
            let dimension=MemRefOp.Dim(Names.value call.Count 1,buffer,Names.value call.Count 0,IntrinsicWriteAbi.bufferType declaration)
            match expectedDimensions.TryFind call.Count with
            | Some previous -> require (previous=dimension) "Shared write count has conflicting descriptor correspondence"
            | None -> expectedDimensions<-expectedDimensions.Add(call.Count,dimension)
            let zero=MLIROp.IndexOp(IndexOp.IndexConst(Names.value call.Count 0,0L))
            let resultType=Alex.CodeGeneration.TypeMapping.scalarCarrierType extentFact.Result
            let conversion=
                if extentFact.IndexUnsigned then IndexOp.IndexCastU(scalarName owner call.Count,Names.value call.Count 1,TIndex,resultType)
                else IndexOp.IndexCastS(scalarName owner call.Count,Names.value call.Count 1,TIndex,resultType)
            require (body |> List.filter ((=) zero) |> List.length = 1) "Write count lost its dimension ordinal"
            require (body |> List.filter ((=) (MLIROp.IndexOp conversion)) |> List.length = 1) "Write count lost its source-settled dimension conversion"
            let proofs = boundary.IntrinsicWriteProofs[site]
            require (proofs |> List.map _.Ordinal |> List.sort = [-1;0;1;2]) "Write proof ordinal inventory is incomplete"
            for row in proofs do require ((infoAt row.Obligation).Body=row.Body) "Write proof citizen differs from its published body"
            let borrow = proofs |> List.filter (fun row -> row.Ordinal=1) |> only "Write has no unique borrow proof"
            let info = infoAt borrow.Obligation
            let origins = view.StaticOrigins |> Map.toList |> List.map (fun (origin,length) ->
                let entry = originEntries.TryFind origin |> Option.defaultWith (fun () -> invalidOp "Write origin has no admitted pool entry")
                require (bigint pool.Entries[entry].Length=length) "Write origin extent differs from the admitted pool"
                {Site=site;Origin=origin;Entry=entry;Count=extent.StaticOrigins[origin];Extent=length;Anchor=info.Id})
            let body = origins |> List.map (fun origin -> origin.Count,origin.Extent,bigint pool.Entries[origin.Entry].StorageLength)
            require (borrow.Body=ObligationBody.StringBorrowBound body) "Write borrow proof omits or changes a source origin"
            origins)
        let physicalCalls = all |> List.filter (function
            | MLIROp.FuncOp(FuncCall(_,symbol,_)) -> boundary.IntrinsicWriteImports.Values |> Seq.exists (fun import -> import.Symbol=symbol)
            | _ -> false)
        require (physicalCalls.Length=boundary.IntrinsicWrites.Count) "Artifact contains an unpaired or duplicated bounded write"
        let usedViews = boundary.IntrinsicWrites.Values |> Seq.map _.Buffer |> Set.ofSeq
        require (usedViews=Set.ofSeq boundary.ByteViews.Keys) "Unsupported required artifact family: byte views used outside bounded writes"
        let usedCounts = boundary.IntrinsicWrites.Values |> Seq.map _.Count |> Set.ofSeq
        require (Set.ofSeq memory.Operations.Keys=usedCounts && Set.ofSeq boundary.StringExtents.Keys=usedCounts)
            "Unsupported required artifact family: memory extents outside bounded writes"
        // An empty source claim inventory cannot authorize residual legacy
        // memory operations. Account for every typed memory instruction, even
        // when the graph has no pool, before permitting artifact extraction.
        for dimension in expectedDimensions.Values do expectedMemory.Add dimension
        let actualMemory=all |> List.choose (function MLIROp.MemRefOp operation -> Some operation | _ -> None)
        let remaining=ResizeArray<MemRefOp>(expectedMemory)
        for operation in actualMemory do
            let index=remaining.IndexOf operation
            require (index>=0) (sprintf "Unsupported required artifact typed memory operation: %A" operation)
            remaining.RemoveAt index
        require (remaining.Count=0) "Artifact memory instruction inventory omits a source-owned pool view or write extent"
        for operation in all do
            match operation with
            | MLIROp.MmioLoad _ | MLIROp.MmioStore _ | MLIROp.GlobalString _
            | MLIROp.GlobalMemref _ | MLIROp.GlobalArray _ | MLIROp.RawMLIR _ ->
                invalidOp (sprintf "Unsupported required artifact memory or opaque target form: %A" operation)
            | _ -> ()
        let entryFor info =
            let node = obligationNodes |> List.choose (fun (id,current) -> if current=info then Some id else None) |> only "Artifact claim lacks unique source citizen"
            let origins = graph.Edges |> List.filter (fun edge -> edge.Target=node && edge.Role=EdgeRole.Constrains)
                          |> List.collect _.Sources |> List.choose originEntries.TryFind |> List.distinct
            origins |> only ("Artifact claim has no unique pool origin: " + info.Id)
        let claims,solverClaims = proof.Obligations |> List.fold (fun (claims,solver) info ->
            let add claim = (info,claim)::claims,solver
            match info.Body with
            | ObligationBody.StorageReservation(length,size) ->
                let entry=entryFor info
                require (entries[entry].Length=length && entries[entry].StorageLength=size) "Storage claim differs from pool"
                add(Storage entry)
            | ObligationBody.ViewContainment(view,length,size) ->
                let entry=entryFor info
                require (view=length && entries[entry].Length=length && entries[entry].StorageLength=size) "View claim differs from pool"
                add(View entry)
            | ObligationBody.NulSentinel last ->
                require (last=0) "Unsupported sentinel claim"
                add(Sentinel(entryFor info))
            | ObligationBody.StaticStorageLayout(slots,used,size,alignment,capacity,spaceAlignment,granularity) ->
                let pool=pool |> Option.defaultWith (fun () -> invalidOp "Required layout has no string pool")
                require (slots=(pool.Entries |> List.map (fun entry -> entry.Offset,entry.StorageLength,1)) &&
                         (used,size,alignment,capacity,spaceAlignment,granularity)=(pool.UsedSize,pool.Size,pool.Alignment,pool.Capacity,pool.SpaceAlignment,pool.Granularity)) "Layout claim differs from exact pool plan"
                add Layout
            | ObligationBody.StringBorrowBound _ ->
                let origins=writes |> List.filter (fun origin -> origin.Anchor=info.Id)
                require (not origins.IsEmpty) "Required borrow claim has no physical bounded write"
                add(Borrow origins)
            | ObligationBody.ConsecutiveLayout _ | ObligationBody.ContinuationLayout _ | ObligationBody.ConcatCopyBound _
            | ObligationBody.CapacityPositive _ | ObligationBody.CapacityFits _ | ObligationBody.InputBufferBound _
            | ObligationBody.InputCopyBound _ | ObligationBody.MappedElementSpan _ | ObligationBody.SpatialKernelPartition _ ->
                invalidOp (sprintf "Unsupported required artifact obligation '%s' (%s): %A" info.Id info.Source info.Body)
            | ObligationBody.FiniteLoopTrip _ | ObligationBody.AdditiveLoopInvariant _ | ObligationBody.FiniteLinearRecurrence _
            | ObligationBody.FiniteAdditiveEffects _ | ObligationBody.IntegerLiteralRange _ | ObligationBody.IntegerRepresentationCoverage _
            | ObligationBody.IntegerDivisorNonzero _ | ObligationBody.IntegerShiftCount _ | ObligationBody.ApplicationDimensions _
            | ObligationBody.RealLiteralRange _ | ObligationBody.RealRepresentationCoverage _ | ObligationBody.DimensionalRelation _ -> claims,info::solver) ([],[])
        match pool with
        | Some _ ->
            require (claims |> List.filter (fun (_,claim) -> claim=Layout) |> List.length = 1) "Static pool lacks exactly one required layout claim"
            for index in 0..entries.Length-1 do
                for expected in [Storage index;View index;Sentinel index] do
                    require (claims |> List.exists (fun (_,claim) -> claim=expected)) "Static pool entry lacks storage/view/sentinel claim coverage"
        | None -> require (claims.IsEmpty && writes.IsEmpty) "Required artifact claim has no admitted pool"
        Ok {Input=input;Source=source;Mlir=mlir;Pool=pool;Claims=List.rev claims;SolverClaims=List.rev solverClaims;Writes=writes}
    with error -> Error("Artifact claim admission: " + error.Message)

/// Addresses are ELF virtual addresses. A PIE loader's load bias and execution
/// permissions remain explicit environmental assumptions, not guessed bases.
type ObservedSection = { Name:string; Address:bigint; Offset:int; Bytes:byte list; Flags:uint64 }
type ObservedLoad = { Address:bigint; MemorySize:bigint; Offset:bigint; FileSize:bigint; Flags:uint64 }
type ObservedPool = { Symbol:string; Section:int; Address:bigint; Offset:int; Size:int }
type ObservedImage = { Sha256:string; Sections:ObservedSection list; Loads:ObservedLoad list; Pool:ObservedPool option }
type private SectionHeader = { Name:int; Kind:uint64; Flags:uint64; Address:bigint; Offset:int; Size:int; Link:int; EntrySize:int }

/// Direct bounded extraction; no old binary, textual disassembly or plugin is
/// involved. Exact object-symbol identity connects the pool to its linked bytes.
let observe (path:string) (poolSymbol:string option) : Result<ObservedImage,string> =
    try
        let bytes=File.ReadAllBytes path
        let bounds at count = require (at>=0 && count>=0 && at<=bytes.Length-count) "Truncated ELF artifact"
        let number at count =
            bounds at count
            let mutable value=0UL
            for i in 0..count-1 do value<-value ||| (uint64 bytes[at+i] <<< (8*i))
            value
        let index value = require (value<=uint64 Int32.MaxValue) "ELF exceeds bounded observer"; int value
        bounds 0 64
        require (bytes[0..6]=[|127uy;69uy;76uy;70uy;2uy;1uy;1uy|] && number 18 2=62UL && number 20 4=1UL && List.contains (number 16 2) [2UL;3UL])
            "Artifact proof requires a linked AMD64 ELF64 little-endian image"
        let at,width,count,names=number 40 8 |> index,number 58 2 |> index,number 60 2 |> index,number 62 2 |> index
        require (count>0 && width>=64 && count<=bytes.Length/width && names<count) "Unsupported ELF section table"
        bounds at (width*count)
        let sections=Array.init count (fun ordinal ->
            let at=at+width*ordinal
            {Name=number at 4 |> index;Kind=number (at+4) 4;Flags=number (at+8) 8;Address=bigint(number (at+16) 8)
             Offset=number (at+24) 8 |> index;Size=number (at+32) 8 |> index;Link=number (at+40) 4 |> index;EntrySize=number (at+56) 8 |> index})
        let name (table:SectionHeader) offset =
            bounds table.Offset table.Size
            require (table.Kind=3UL && offset>=0 && offset<table.Size) "Invalid ELF string table"
            let mutable last=offset
            while last<table.Size && bytes[table.Offset+last]<>0uy do last<-last+1
            require (last<table.Size) "Unterminated ELF name"
            Encoding.UTF8.GetString(bytes,table.Offset+offset,last-offset)
        let pool=poolSymbol |> Option.map (fun symbol ->
            let matches=
                [for table in sections do
                    if table.Kind=2UL || table.Kind=11UL then
                        require (table.EntrySize>=24 && table.Size%table.EntrySize=0 && table.Link<count) "Invalid ELF symbol table"
                        bounds table.Offset table.Size
                        for ordinal in 0..table.Size/table.EntrySize-1 do
                            let at=table.Offset+ordinal*table.EntrySize
                            if name sections[table.Link] (number at 4 |> index)=symbol then
                                let section=number (at+6) 2 |> index
                                require (number (at+4) 1=1UL && number (at+5) 1 &&& 3UL=0UL && table.Kind=2UL && section>0 && section<count)
                                    "Pool symbol is not a defined local ELF object with default visibility"
                                yield section,bigint(number (at+8) 8),number (at+16) 8 |> index]
                |> List.distinct
            let section,address,size=matches |> only ("Expected exactly one linked pool symbol: "+symbol)
            let header=sections[section]
            let offset=address-header.Address
            require (header.Kind=1UL && header.Flags &&& 2UL<>0UL && size>0 && offset>=0I && offset+bigint size<=bigint header.Size) "Pool exceeds allocated file-backed section"
            section,address,size,header.Offset+int offset,symbol)
        let programAt,programWidth,programCount=number 32 8 |> index,number 54 2 |> index,number 56 2 |> index
        require (programCount>0 && programCount<>65535 && programWidth>=56 && programCount<=bytes.Length/programWidth) "Unsupported ELF program table"
        bounds programAt (programWidth*programCount)
        let loads=
            [for ordinal in 0..programCount-1 do
                let at=programAt+ordinal*programWidth
                if number at 4=1UL then
                    let fileOffset,fileSize=number (at+8) 8 |> index,number (at+32) 8 |> index
                    bounds fileOffset fileSize
                    let memorySize=bigint(number (at+40) 8)
                    require (bigint fileSize<=memorySize) "Invalid ELF LOAD file/memory extent"
                    yield {Address=bigint(number (at+16) 8);MemorySize=memorySize;Offset=bigint fileOffset;FileSize=bigint fileSize;Flags=number (at+4) 4}]
        require (not loads.IsEmpty) "ELF has no LOAD evidence"
        let selected=
            sections |> Array.indexed |> Array.toList |> List.filter (fun (ordinal,header) ->
                match pool with
                | Some(section,_,_,_,_) -> ordinal=section
                | None -> header.Kind=1UL && header.Flags &&& 2UL<>0UL && header.Size>0)
        require (not selected.IsEmpty) "ELF has no nonempty allocated byte section to account for"
        let observed=selected |> List.map (fun (_,header) ->
            require (header.Size<=1048576) "Artifact section exceeds the admitted one-MiB Rocq byte-accounting budget"
            bounds header.Offset header.Size
            require (loads |> List.exists (fun load ->
                load.Address<=header.Address && header.Address+bigint header.Size<=load.Address+load.MemorySize &&
                load.Offset+header.Address-load.Address=bigint header.Offset && header.Address-load.Address+bigint header.Size<=load.FileSize))
                "Allocated section has no complete file-backed LOAD correspondence"
            {Name=name sections[names] header.Name;Address=header.Address;Offset=header.Offset;Bytes=bytes[header.Offset..header.Offset+header.Size-1] |> Array.toList;Flags=header.Flags})
        let pool=pool |> Option.map (fun (section,address,size,offset,symbol) ->
            {Symbol=symbol;Section=selected |> List.findIndex (fun (ordinal,_) -> ordinal=section);Address=address;Offset=offset;Size=size})
        Ok {Sha256=sha bytes;Sections=observed;Loads=loads;Pool=pool}
    with error -> Error("Artifact ELF extraction: "+error.Message)

let private safeComment (text:string) = text.Replace("(*","( *").Replace("*)","* )").Replace("\r"," ").Replace("\n"," ")

let private render (plan:Plan) (image:ObservedImage) =
    let text=StringBuilder()
    let line (value:string)=text.AppendLine(value) |> ignore
    let bytes values=values |> List.map string |> String.concat "; "
    let lemma name proposition proof = line(sprintf "Lemma %s : %s." name proposition);line(sprintf "Proof. %s Qed." proof)
    let decimal (value:bigint)=number value
    line "(* Current ELF facts and source-linked storage claims."
    line "   Foreign extents are byte-accounted, not source-proved."
    line "   Trusted: ELF extraction, compiler/backend realization, Rocq/kernel/stdlib,"
    line "   loader load bias/permissions and the declared syscall environment."
    line "   Arithmetic/dimensional solver claims remain solver scope; no instruction equivalence is asserted. *)"
    line(sprintf "(* invocation %O; ELF SHA256 %s *)" plan.Source.Evidence.Invocation image.Sha256)
    line "From Stdlib Require Import ZArith Lia List."
    line "Import ListNotations. Open Scope Z_scope."
    let mutable theoremNames=[]
    let prove name proposition proof=lemma name proposition proof;theoremNames<-name::theoremNames
    for ordinal,section in List.indexed image.Sections do
        let prefix=sprintf "section_%d" ordinal
        line(sprintf "(* ELF section %s; non-pool bytes below are trusted foreign extents. *)" (safeComment section.Name))
        line(sprintf "Definition %s_address : Z := %s." prefix (decimal section.Address))
        line(sprintf "Definition %s_size : Z := %d." prefix section.Bytes.Length)
        line(sprintf "Definition %s_bytes : list Z := [%s]." prefix (bytes section.Bytes))
        let ranges=
            match image.Pool with
            | Some pool when pool.Section=ordinal ->
                let offset=int(pool.Address-section.Address)
                [if offset>0 then yield "trusted_foreign_prefix",0,offset
                 yield "source_pool",offset,pool.Size
                 if offset+pool.Size<section.Bytes.Length then yield "trusted_foreign_suffix",offset+pool.Size,section.Bytes.Length-offset-pool.Size]
            | _ -> ["trusted_foreign_entire_section",0,section.Bytes.Length]
        let names=ranges |> List.mapi (fun index (role,offset,size) ->
            let name=sprintf "%s_extent_%d" prefix index
            line(sprintf "(* %s *)" role)
            line(sprintf "Definition %s_address : Z := %s." name (decimal(section.Address+bigint offset)))
            line(sprintf "Definition %s_size : Z := %d." name size)
            line(sprintf "Definition %s_bytes : list Z := [%s]." name (section.Bytes |> List.skip offset |> List.take size |> bytes))
            prove (name+"_bytes_length") (sprintf "Z.of_nat (length %s_bytes) = %s_size" name name) "reflexivity."
            name)
        prove (prefix+"_bytes_accounted") (sprintf "%s_bytes = %s" prefix (names |> List.map (fun name -> name+"_bytes") |> String.concat " ++ ")) "reflexivity."
        let tiles=[yield sprintf "%s_address = %s_address" names.Head prefix
                   for index in 1..names.Length-1 do yield sprintf "%s_address = %s_address + %s_size" names[index] names[index-1] names[index-1]
                   yield sprintf "%s_address + %s_size = %s_address + %s_size" (List.last names) (List.last names) prefix prefix]
        prove (prefix+"_tiled") (String.concat " /\\ " tiles) "repeat split; reflexivity."
        let load=image.Loads |> List.find (fun load ->
            load.Address<=section.Address && section.Address+bigint section.Bytes.Length<=load.Address+load.MemorySize &&
            load.Offset+section.Address-load.Address=bigint section.Offset && section.Address-load.Address+bigint section.Bytes.Length<=load.FileSize)
        prove (prefix+"_file_backed_load")
            (sprintf "%s <= %s /\\ %s + %d <= %s + %s /\\ %s + %s - %s = %d /\\ %s - %s + %d <= %s"
                (decimal load.Address) (decimal section.Address) (decimal section.Address) section.Bytes.Length (decimal load.Address) (decimal load.MemorySize)
                (decimal load.Offset) (decimal section.Address) (decimal load.Address) section.Offset (decimal section.Address) (decimal load.Address) section.Bytes.Length (decimal load.FileSize)) "lia."
    match plan.Pool,image.Pool with
    | Some pool,Some actual ->
        require (actual.Symbol=pool.Symbol) "Observed pool symbol differs from source inventory"
        let section=image.Sections[actual.Section]
        let relative=int(actual.Address-section.Address)
        let actualBytes=section.Bytes |> List.skip relative |> List.take actual.Size
        line(sprintf "Definition pool_address : Z := %s." (decimal actual.Address))
        line(sprintf "Definition pool_size : Z := %d." actual.Size)
        line(sprintf "Definition pool_bytes : list Z := [%s]." (bytes actualBytes))
        let numeric name proposition=prove name proposition "cbv [pool_address pool_size]; lia."
        prove "pool_bytes_correspond" (sprintf "pool_bytes = [%s]" (bytes pool.Bytes)) "reflexivity."
        numeric "pool_extent_corresponds" (sprintf "pool_size = %d" pool.Size)
        numeric "pool_within_section" (sprintf "%s <= pool_address /\\ pool_address + pool_size <= %s + %d" (decimal section.Address) (decimal section.Address) section.Bytes.Length)
        prove "pool_alignment" (sprintf "Z.modulo pool_address %d = 0" pool.Alignment) "vm_compute; reflexivity."
        prove "section_readonly" (sprintf "Z.land %d 5 = 0" section.Flags) "vm_compute; reflexivity."
        let overlapping=image.Loads |> List.filter (fun load -> load.Address<actual.Address+bigint actual.Size && actual.Address<load.Address+load.MemorySize)
        require (not overlapping.IsEmpty) "Pool lacks LOAD permission evidence"
        for ordinal,load in List.indexed overlapping do
            prove (sprintf "pool_load_%d_readonly" ordinal) (sprintf "Z.land %d 7 = 4" load.Flags) "vm_compute; reflexivity."
        for ordinal,(info,claim) in List.indexed plan.Claims do
            let name=sprintf "claim_%d" ordinal
            line(sprintf "(* source anchor %s; %s *)" (safeComment info.Id) (safeComment info.Source))
            let storage index=pool.Entries[index]
            match claim with
            | Storage index ->
                let entry=storage index
                numeric name (sprintf "%d = %d + 1 /\\ 0 <= %d /\\ %d + %d <= pool_size" entry.StorageLength entry.Length entry.Offset entry.Offset entry.StorageLength)
            | View index ->
                let entry=storage index
                numeric name (sprintf "0 <= %d /\\ %d < %d /\\ %d + %d <= pool_size" entry.Length entry.Length entry.StorageLength entry.Offset entry.Length)
            | Sentinel index ->
                let entry=storage index
                prove name (sprintf "nth %d pool_bytes 1 = 0" (entry.Offset+entry.Length)) "reflexivity."
            | Layout ->
                let clauses=[yield sprintf "0 <= %d /\\ %d <= pool_size /\\ pool_size <= %d" pool.UsedSize pool.UsedSize pool.Capacity
                             for entry in pool.Entries do yield sprintf "0 <= %d /\\ %d + %d <= %d" entry.Offset entry.Offset entry.StorageLength pool.UsedSize
                             for i in 0..pool.Entries.Length-1 do
                                 for j in i+1..pool.Entries.Length-1 do
                                     let left,right=pool.Entries[i],pool.Entries[j]
                                     yield sprintf "(%d + %d <= %d \\/ %d + %d <= %d)" left.Offset left.StorageLength right.Offset right.Offset right.StorageLength left.Offset]
                numeric name (String.concat " /\\ " clauses)
                prove (name+"_space_alignment") (sprintf "Z.modulo pool_address %d = 0" pool.SpaceAlignment) "vm_compute; reflexivity."
                prove (name+"_granularity") (sprintf "Z.modulo pool_size %d = 0" pool.Granularity) "vm_compute; reflexivity."
            | Borrow origins ->
                let clauses=origins |> List.map (fun origin ->
                    let entry=storage origin.Entry
                    sprintf "(0 <= %s /\\ %s = %s /\\ %s = %d /\\ %s < %d /\\ pool_address <= pool_address + %d /\\ pool_address + %d + %s <= pool_address + pool_size)"
                        (decimal origin.Count) (decimal origin.Count) (decimal origin.Extent) (decimal origin.Extent) entry.Length (decimal origin.Count) entry.StorageLength entry.Offset entry.Offset (decimal origin.Count))
                numeric name (String.concat " /\\ " clauses)
    | None,None -> line "(* Complete admitted inventory contains no required source memory artifact claim. All accounted bytes are foreign to that empty inventory. *)"
    | _ -> invalidOp "Source and observed pool inventories differ"
    for name in List.rev theoremNames do line("Print Assumptions "+name+".")
    text.ToString(),List.rev theoremNames

let private run executable arguments =
    let start=ProcessStartInfo(executable,UseShellExecute=false,RedirectStandardOutput=true,RedirectStandardError=true)
    for argument in arguments do start.ArgumentList.Add argument
    use child=new Process(StartInfo=start)
    require (child.Start()) ("Cannot start required proof tool "+executable)
    let output,errors=child.StandardOutput.ReadToEndAsync(),child.StandardError.ReadToEndAsync()
    if not(child.WaitForExit 30000) then
        child.Kill true
        child.WaitForExit 5000 |> ignore
        invalidOp ("Required proof tool timed out: "+executable)
    require (Task.WhenAll([|output :> Task;errors :> Task|]).Wait 5000) "Required proof tool did not close redirected I/O"
    require (child.ExitCode=0) (sprintf "%s exited %d: %s%s" executable child.ExitCode output.Result errors.Result)
    output.Result,errors.Result

let private identify (executable:string) arguments =
    let path=
        if Path.IsPathRooted executable then executable
        else
            let paths=Environment.GetEnvironmentVariable "PATH" |> Option.ofObj |> Option.defaultValue ""
            paths.Split(Path.PathSeparator) |> Array.map (fun directory -> Path.Combine(directory,executable)) |> Array.tryFind File.Exists
            |> Option.defaultWith (fun () -> invalidOp ("Required proof tool is unavailable: "+executable))
    let file=FileInfo path
    let target=file.ResolveLinkTarget true
    let path=if isNull target then file.FullName else target.FullName
    let hash=fileHash path
    let version,_=run path ["--version"]
    require (not(String.IsNullOrWhiteSpace version) && fileHash path=hash) "Proof executable changed while identifying its version"
    {Executable=path;Sha256=hash;Version=version.Trim();Arguments=arguments}

type CheckedRocq = {
    Tool:ProofTool; SourcePath:string; ObjectPath:string; SourceSha256:string; ObjectSha256:string
    StandardOutput:string; StandardError:string
}

/// A generated file or an old .vo is never a successful checker result.
let checkWith executable (directory:string) (proof:string) : Result<CheckedRocq,string> =
    try
        Directory.CreateDirectory directory |> ignore
        let path=Path.GetFullPath(Path.Combine(directory,"MemoryMap.v"))
        let compiled=Path.ChangeExtension(path,".vo")
        if File.Exists compiled then File.Delete compiled
        File.WriteAllText(path,proof,UTF8Encoding(false))
        let hash=fileHash path
        let tool=identify executable ["compile";path]
        require (fileHash tool.Executable=tool.Sha256) "Rocq executable changed before checking"
        let output,errors=run tool.Executable tool.Arguments
        File.WriteAllText(Path.Combine(directory,"rocq.stdout"),output)
        File.WriteAllText(Path.Combine(directory,"rocq.stderr"),errors)
        require (fileHash tool.Executable=tool.Sha256 && fileHash path=hash) "Rocq executable or generated proof changed while checking"
        require (File.Exists compiled && FileInfo(compiled).Length>0L) "Rocq did not produce a fresh checked object"
        Ok {Tool=tool;SourcePath=path;ObjectPath=compiled;SourceSha256=hash;ObjectSha256=fileHash compiled;StandardOutput=output;StandardError=errors}
    with error -> Error("Required Rocq artifact proof: "+error.Message)

type ArtifactClaimEvidence = { Anchor:string; Source:string; Theorem:string }
type ArtifactOriginEvidence = { Site:int; Origin:int; Entry:int; Count:string; Extent:string; Anchor:string }
type ForeignExtentEvidence = { Section:string; Role:string; Address:string; Size:int; BytesSha256:string }
type ArtifactReceipt = {
    Invocation:Guid
    Stage:string
    Scope:string
    ArtifactSha256:string
    PortableMlirSha256:string
    RuntimeMlirSha256:string
    SourceQuerySha256:string
    MlirInputSha256:string
    MlirQuerySha256:string
    SourceEvidence:ProofStageEvidence
    MlirEvidence:ProofStageEvidence
    ComposerSha256:string
    CcsSha256:string
    Rocq:CheckedRocq
    Claims:ArtifactClaimEvidence array
    Origins:ArtifactOriginEvidence array
    SolverOnlyAnchors:string array
    ForeignExtents:ForeignExtentEvidence array
    Theorems:string array
    Assumptions:string array
}

let verifyWith executable directory (runtimeText:string) artifact (plan:Plan) : Result<ArtifactReceipt,string> =
    try
        prepare plan.Input |> need |> ignore
        let image=observe artifact (plan.Pool |> Option.map _.Symbol) |> need
        let proof,theorems=render plan image
        Directory.CreateDirectory directory |> ignore
        File.WriteAllText(Path.Combine(directory,"portable.mlir"),plan.Input.Text)
        File.WriteAllText(Path.Combine(directory,"runtime.mlir"),runtimeText)
        File.WriteAllText(Path.Combine(directory,"source.smt2"),plan.Source.Query)
        File.WriteAllText(Path.Combine(directory,"obligations.mlir"),plan.Mlir.Text)
        File.WriteAllText(Path.Combine(directory,"lowered.smt2"),plan.Mlir.Query)
        let checkedProof=checkWith executable directory proof |> need
        require (fileHash artifact=image.Sha256) "Linked artifact changed while Rocq checked its model"
        prepare plan.Input |> need |> ignore
        let foreign=image.Sections |> List.mapi (fun ordinal section ->
            let ranges=
                match image.Pool with
                | Some pool when pool.Section=ordinal ->
                    let offset=int(pool.Address-section.Address)
                    [if offset>0 then yield "prefix",0,offset
                     if offset+pool.Size<section.Bytes.Length then yield "suffix",offset+pool.Size,section.Bytes.Length-offset-pool.Size]
                | _ -> ["entire-section",0,section.Bytes.Length]
            ranges |> List.map (fun (role,offset,size) ->
                {Section=section.Name;Role="trusted-foreign-"+role;Address=number(section.Address+bigint offset);Size=size
                 BytesSha256=section.Bytes |> List.skip offset |> List.take size |> List.toArray |> sha})) |> List.concat
        let receipt=
            {Invocation=plan.Source.Evidence.Invocation;Stage="elf-rocq";Scope="Linux AMD64 ELF static string pool and complete bounded-write origin correspondence; whole selected-section bytes accounted"
             ArtifactSha256=image.Sha256;PortableMlirSha256=textHash plan.Input.Text;RuntimeMlirSha256=textHash runtimeText
             SourceQuerySha256=plan.Source.Evidence.QuerySha256;MlirInputSha256=plan.Mlir.Evidence.InputSha256;MlirQuerySha256=plan.Mlir.Evidence.QuerySha256
             SourceEvidence=plan.Source.Evidence;MlirEvidence=plan.Mlir.Evidence
             ComposerSha256=fileHash typeof<BackEndInput>.Assembly.Location;CcsSha256=fileHash typeof<SemanticGraph>.Assembly.Location;Rocq=checkedProof
             Claims=plan.Claims |> List.mapi (fun ordinal (info,_) -> {Anchor=info.Id;Source=info.Source;Theorem=sprintf "claim_%d" ordinal}) |> List.toArray
             Origins=plan.Writes |> List.map (fun write -> {Site=NodeId.value write.Site;Origin=NodeId.value write.Origin;Entry=write.Entry;Count=number write.Count;Extent=number write.Extent;Anchor=write.Anchor}) |> List.toArray
             SolverOnlyAnchors=plan.SolverClaims |> List.map _.Id |> List.toArray;ForeignExtents=List.toArray foreign;Theorems=List.toArray theorems
             Assumptions=[|"CCS source semantics and obligation encoding; source and MLIR cvc5 verdicts are solver-dependent"
                           "Clerical ELF/typed-operand extraction and compiler/backend lowering; no instruction-level semantic equivalence theorem"
                           "ELF loader mapping, load bias and permissions; declared Linux AMD64 syscall and external runtime behavior"
                           "Rocq kernel, selected executable and installed Stdlib; foreign extents are byte-accounted, not source-proved"
                           "Current graph-object publication identity; no accepted-source-revision or deeply frozen checker-cell claim"|]}
        File.WriteAllText(Path.Combine(directory,"artifact-receipt.json"),JsonSerializer.Serialize(receipt,JsonSerializerOptions(WriteIndented=true)))
        Ok receipt
    with error -> Error("Required artifact stage: "+error.Message)

/// Link to a same-directory provisional file. The existing output remains
/// untouched until the exact fresh ELF and generated Rocq object are checked.
let publishWith executable (input:BackEndInput) runtimeText (context:BackEndContext) (produce:string -> Result<unit,string>) =
    try
        let target=context.TargetTripleOverride |> Option.bind (TargetProfiles.linuxAmd64Syscalls context)
        require target.IsSome "Required artifact proof has no admitted profile for the selected target (supported: Linux AMD64 ELF)"
        let plan=prepare input |> need
        let output=Path.GetFullPath context.OutputPath
        let directory=Path.GetDirectoryName output
        Directory.CreateDirectory directory |> ignore
        let invocation=Guid.NewGuid().ToString("N")
        let provisional=Path.Combine(directory,"."+Path.GetFileName(output)+"."+invocation+".provisional")
        let evidence=Path.Combine(directory,Path.GetFileName(output)+".proof-"+invocation)
        try
            produce provisional |> need
            let receipt=verifyWith executable evidence runtimeText provisional plan |> need
            require (fileHash provisional=receipt.ArtifactSha256) "Verified artifact changed before publication"
            File.Move(provisional,output,true)
            Ok(NativeBinary output)
        finally
            if File.Exists provisional then File.Delete provisional
    with error -> Error("Native artifact publication refused: "+error.Message)

let publish input runtimeText context produce = publishWith "rocq" input runtimeText context produce
