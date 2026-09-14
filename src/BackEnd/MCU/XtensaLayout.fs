/// Linker layout for a ROM-loaded, all-SRAM ESP32-S3 image.
///
/// The Cortex-M generator in Layout.fs emits two regions, FLASH and RAM, and
/// copies `.data` from the first to the second at runtime via `AT>`. Neither
/// half of that shape holds here:
///
///  * There is no executable flash region. The mask ROM loads every segment
///    into SRAM and jumps, so `.data` has no load address distinct from its
///    virtual address -- the ROM loader has already done the copy by the time
///    the entry symbol runs. Only `.bss` needs zeroing.
///
///  * The SRAM is three banks and the middle one is DUAL-MAPPED: SRAM1 answers
///    to the instruction bus at 0x4037_8000 and to the data bus at
///    0x3FC8_8000, and those are the same silicon. A script that fills an IRAM
///    region upward from its base and a DRAM region upward from its base puts
///    `.text` and `.data` on top of each other, with no diagnostic from the
///    linker: the addresses do not overlap, only the memory does.
///
/// The partition is therefore explicit. `Sram1InstructionBytes` says how much
/// of the shared bank the instruction side takes; the data side starts after
/// it. The generated script asserts the split holds, so a bad declaration
/// fails at link time rather than on the board.
module BackEnd.MCU.XtensaLayout

open System.IO
open Core.Types.Pipeline

let origin (space: BAREWire.Platform.MemorySpace) =
    space.Base |> Option.defaultWith (fun () -> failwith ("No base for " + space.Name))

/// The instruction and data windows this image may use, after partitioning the
/// shared bank. Computed once so the script and the image verifier agree.
type Windows = {
    /// Instruction bus: SRAM0 plus the instruction share of SRAM1. These are
    /// contiguous in the instruction address space (SRAM0 ends exactly where
    /// SRAM1 begins), so they present as one region.
    IramOrigin: int64
    IramBytes: int64
    /// Data bus: the data share of SRAM1 plus all of SRAM2, also contiguous.
    DramOrigin: int64
    DramBytes: int64
}

/// Partition the shared bank and check every premise the split relies on.
let windows (target: XtensaTarget) : Windows =
    let sram0, sram1, sram2 = target.Sram0, target.Sram1, target.Sram2
    let dataBase = target.Sram1DataBase
    let instructionShare = int64 target.Image.Sram1InstructionBytes

    if instructionShare < 0L || instructionShare > sram1.Capacity then
        failwithf "Sram1InstructionBytes %d is outside the shared bank's %d bytes" instructionShare sram1.Capacity
    // The ROM loader copies words, and the vector block needs a 1024-byte
    // boundary inside the instruction window.
    if instructionShare % int64 target.Image.VectorAlignment <> 0L then
        failwithf "Sram1InstructionBytes %d must be a multiple of the %d-byte vector alignment so the data side starts aligned"
            instructionShare target.Image.VectorAlignment
    // SRAM0 must abut SRAM1 on the instruction bus for one region to cover both.
    if origin sram0 + sram0.Capacity <> origin sram1 then
        failwithf "%s does not abut %s on the instruction bus" sram0.Name sram1.Name
    // The data share must abut SRAM2 on the data bus for the same reason.
    if dataBase + sram1.Capacity <> origin sram2 then
        failwithf "%s's data-bus window does not abut %s" sram1.Name sram2.Name
    // The data window ends where the ROM loader's own memory and the cache's
    // begin, not where the silicon does. What lies above the limit is not
    // the image's to use, however much of it the part has.
    let dramOrigin = dataBase + instructionShare
    if target.DataLimit <= dramOrigin then
        failwithf "DataLimit 0x%X leaves no data window: the data side of the shared bank starts at 0x%X"
            target.DataLimit dramOrigin
    if target.DataLimit > origin sram2 + sram2.Capacity then
        failwithf "DataLimit 0x%X lies beyond the data-bus SRAM" target.DataLimit

    { IramOrigin = origin sram0
      IramBytes = sram0.Capacity + instructionShare
      DramOrigin = dramOrigin
      DramBytes = target.DataLimit - dramOrigin }

let generate (target: XtensaTarget) destination =
    Directory.CreateDirectory destination |> ignore
    let w = windows target
    let image = target.Image
    let stackTop = w.DramOrigin + w.DramBytes
    let stackBottom = stackTop - int64 image.StackBytes
    if image.StackBytes <= 0 || int64 image.StackBytes >= w.DramBytes || image.StackBytes % 16 <> 0 then
        failwith "Invalid Xtensa stack reservation; the call ABI requires 16-byte stack alignment"

    // The offset of the data window inside the shared bank, used by the script's
    // non-overlap assertion. Expressed as an instruction-bus address so the two
    // views can be compared in one arithmetic domain.
    let sharedBankInstructionBase = origin target.Sram1
    let dataShareAsInstructionAddress = sharedBankInstructionBase + int64 image.Sram1InstructionBytes

    let script =
        sprintf """/* Generated by Composer from BAREWire declarations; do not edit.
 *
 * All-SRAM image. The Espressif mask ROM loads each segment to its address and
 * jumps to %s, so there is no load-address/virtual-address split and no
 * runtime .data copy: only .bss is zeroed by startup.
 *
 * SRAM1 is dual-mapped. IRAM holds SRAM0 plus the first 0x%X bytes of SRAM1;
 * DRAM holds the rest of SRAM1 plus SRAM2. The final ASSERT proves the two
 * windows do not name the same silicon.
 */
ENTRY(%s)
MEMORY {
  IRAM (rx) : ORIGIN = 0x%X, LENGTH = 0x%X
  DRAM (rw) : ORIGIN = 0x%X, LENGTH = 0x%X
}
__stack_top = 0x%X;
__stack_bottom = 0x%X;
__shared_bank_instruction_base = 0x%X;
__shared_bank_data_split = 0x%X;
/* One program header per kind, declared rather than inferred. Left to itself
 * the linker folds a writable .data and the NOBITS .bss behind it into one
 * PT_LOAD whose memsz exceeds its filesz, and the image writer refuses that:
 * the ROM loader copies file bytes and zero-fills nothing, so .bss must stay
 * a segment of its own with no file bytes, cleared by startup. */
PHDRS {
  text PT_LOAD;
  data PT_LOAD;
  bss  PT_LOAD;
}
SECTIONS {
  /* VECBASE ignores its low bits, so the vector block leads the image at a
   * %d-byte boundary. It is code, not a table of addresses. */
  .vectors ORIGIN(IRAM) : ALIGN(%d) { KEEP(*(.vectors)) KEEP(*(.vectors.*)) } > IRAM :text
  .text : ALIGN(4) {
    /* Literal pools MUST precede the code that references them. Xtensa's L32R
     * loads a constant at a NEGATIVE offset from the instruction -- it can only
     * reach backwards, up to 256 KB. Emit .text first and every movi in the
     * startup code fails to link with "relocation R_XTENSA_SLOT0_OP out of
     * range", because the literal ended up ahead of its use. */
    *(.literal .literal.*)
    *(.text .text.*)
  } > IRAM :text
  __iram_end = .;

  /* Initialized data is placed, not copied: the ROM loader delivers it.
   * Read-only data lives here, on the DATA bus, and not with the code: the
   * instruction bus serves only aligned 32-bit loads, and the compiler
   * freely turns a match into a byte or halfword table read with l8ui/l16ui,
   * which through the instruction window is a LoadStoreError. */
  .data ORIGIN(DRAM) : ALIGN(16) {
    __data_start = .; *(.data*) *(.rodata .rodata.*) . = ALIGN(4); __data_end = .;
  } > DRAM :data
  .bss (NOLOAD) : ALIGN(16) {
    __bss_start = .; *(.bss*) *(COMMON) . = ALIGN(4); __bss_end = .;
  } > DRAM :bss
  .noinit (NOLOAD) : ALIGN(16) { *(.noinit*) } > DRAM :bss
  __dram_end = .;
  .stack __stack_bottom (NOLOAD) : { . += %d; } > DRAM :bss
  /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.xt.prop*) *(.xt.lit*) }
}
ASSERT(SIZEOF(.vectors) == %d, "Wrong vector block size")
ASSERT((ADDR(.vectors) & (%d - 1)) == 0, "Vector block is not VECBASE-aligned")
ASSERT(__dram_end <= __stack_bottom, "Data and stack collide")
ASSERT(__iram_end <= ORIGIN(IRAM) + LENGTH(IRAM), "IRAM overflow")
/* The dual-mapping check: instruction-side use must stop before the byte of
 * the shared bank where data-side use begins. Without this, .text and .data
 * silently occupy the same SRAM at two different addresses. */
ASSERT(__iram_end <= __shared_bank_data_split,
       "IRAM use crosses into the data side of the shared SRAM bank")
"""
            image.EntrySymbol image.Sram1InstructionBytes image.EntrySymbol
            w.IramOrigin w.IramBytes w.DramOrigin w.DramBytes
            stackTop stackBottom
            sharedBankInstructionBase dataShareAsInstructionAddress
            image.VectorAlignment image.VectorAlignment
            image.StackBytes
            target.Vectors.Layout.Size image.VectorAlignment

    File.WriteAllText(Path.Combine(destination, "memory.ld"), script)
    // Startup assembly reads these rather than hardcoding the same numbers.
    File.WriteAllText(
        Path.Combine(destination, "layout.inc"),
        sprintf ".set VECTOR_BYTES, %d\n.set VECTOR_ALIGNMENT, %d\n.set STACK_BYTES, %d\n"
            target.Vectors.Layout.Size image.VectorAlignment image.StackBytes)
    Tools.writeJson (Path.Combine(destination, "layout.json"))
        {| platform = target.PlatformId
           iramOrigin = w.IramOrigin; iramBytes = w.IramBytes
           dramOrigin = w.DramOrigin; dramBytes = w.DramBytes
           sharedBankInstructionBytes = image.Sram1InstructionBytes
           sharedBankCapacity = target.Sram1.Capacity
           stackBottom = stackBottom; stackTop = stackTop
           vectorBytes = target.Vectors.Layout.Size
           vectorAlignment = image.VectorAlignment
           dataCopiedByRomLoader = true |}
    w
