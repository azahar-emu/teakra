#pragma once

#include "common_types.h"

namespace Teakra::Cycles {

// Instruction timings measured on 3DS hardware. An instruction costs one cycle
// per program word plus the extras below. Control flow instructions cost the totals below instead.

constexpr u32 MmioAccess = 1; // extra, per MMIO load or store

constexpr u32 BranchTaken = 3;     // total: br, brr, call, callr
constexpr u32 IndirectBranch = 4;  // total: calla, mov to pc
constexpr u32 ReturnTaken = 4;     // total: ret, rets, reti, retic
constexpr u32 NotTaken = 1;        // extra: a conditional branch, call or return that is not taken
constexpr u32 ProgramReadJump = 7; // total: movpdw (jump to an address read from program memory)

constexpr u32 ProgramReadToRegister = 4; // total: movp to a register
constexpr u32 ProgramReadToData = 3;     // total: movp to data memory

constexpr u32 BlockRepeatStoreRestore = 2; // total: bkrepsto, bkreprst

} // namespace Teakra::Cycles
