#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <exception>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stack>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xbyak/xbyak.h>
#include "bit.h"
#include "core_timing.h"
#include "cycles.h"
#include "decoder.h"
#include "interpreter.h"
#include "interrupt_requests.h"
#include "jit_regs.h"
#include "memory_interface.h"
#include "operand.h"
#include "register.h"
#include "shared_memory.h"
#include "xbyak_abi.h"

#ifdef WIN32
static constexpr bool IsWindows() {
    return true;
}
#else
static constexpr bool IsWindows() {
    return false;
}
#endif

namespace Teakra {

// Marks an instruction (or a variant of it) that the JIT does not emit. The block is cut before
// it and the instruction runs on the interpreter instead.
#define NOT_IMPLEMENTED()                                                                          \
    do {                                                                                           \
        unimplemented = true;                                                                      \
        return;                                                                                    \
    } while (0)

void SaveJitRegisters(const JitRegisters& jit, RegisterState& state);
void LoadJitRegisters(JitRegisters& jit, const RegisterState& state);

class EmitX64 {
    static constexpr std::size_t MaxCodeSize = 64 * 1024 * 1024;
    static constexpr std::size_t CodeCacheMargin = 1024 * 1024;
    static constexpr u32 DefaultMaxBlockInstructions = 64;
    static constexpr s64 LoopBudget = 2048;
    // Technically there's prpage which would make this 22 bits wide, but it's always zero so whatever.
    static constexpr std::size_t BlockCacheSize = 1ULL << 18;

public:
    EmitX64(CoreTiming& core_timing, MemoryInterface& mem, Interpreter& interpreter,
            RegisterState& state)
        : core_timing(core_timing), mem(mem), interpreter(interpreter), state(state),
          c(MaxCodeSize) {
        c.setDefaultJmpNEAR(true);
        block_cache = std::make_unique<BlockList[]>(BlockCacheSize);
        EmitDispatcher();
        Reset();
    }

    using RunCodeFuncType = void (*)(EmitX64*);
    RunCodeFuncType run_code{};

    using BlockFunc = void (*)(JitRegisters*);

    struct BlockSwapState {
        Mod1 mod1{};
        Mod0 mod0{};
        Mod2 mod2{};
        std::array<ArpU, 4> arp{};
        std::array<ArU, 2> ar{};
        u16 pad{};
    };

    struct BlockKey {
        BlockSwapState curr{};
        BlockSwapState shadow{};
        Cfg cfgi, cfgj;
        u16 stepi0, stepj0;
        Cfg cfgib, cfgjb;
        u16 stepi0b, stepj0b;
        u32 pc;
        u32 pad;

        FORCE_INLINE bool operator==(const BlockKey& other) const {
            return std::memcmp(this, &other, sizeof(BlockKey)) == 0;
        }
    };

    static_assert(sizeof(BlockKey) == 64);

    struct Block {
        BlockFunc func;
        // Entry point for linked blocks, with the registers already loaded. Null for fallback
        // blocks.
        const void* body;
        s32 cycles;
    };

    struct LinkSlot {
        static constexpr u32 NoPc = 0xFFFFFFFF;
        struct Entry {
            u32 pc = NoPc;
            const void* code = nullptr;
        };
        std::array<Entry, 2> entries{};
        u32 next = 0;
        BlockKey key{};
    };

    CoreTiming& core_timing;
    MemoryInterface& mem;
    Interpreter& interpreter;
    RegisterState& state;
    JitRegisters regs;
    Xbyak::CodeGenerator c;
    s64 cycles_remaining = 0;
    // Cycles run but not yet applied to the timers and BTDMPs. They are applied before an MMIO
    // access or a fallback, when one of them needs to be signaled (pending_ticks above ticks_to_event),
    // and at the end of Run.
    u64 pending_ticks = 0;
    u64 ticks_to_event = 0;
    Xbyak::Label block_exit;
    std::size_t dispatcher_size = 0;
    const std::vector<Matcher<EmitX64>> decoders = GetDecoderTable<EmitX64>();
    std::unordered_set<u32> bkrep_end_locations;
    std::unordered_set<u32> rep_end_locations;
    u32 max_block_instructions = DefaultMaxBlockInstructions;
    u64 fallback_count = 0;
    std::exception_ptr pending_exception;
    bool compiling = false;
    bool pc_written = false;
    bool unimplemented = false;
    bool flush_pending = false;
    bool state_dirty = false;
    bool link_ok = false;
    // Set while emitting a repetition that is followed by another one.
    // Only sticky flags are computed.
    bool light_flags = false;
    // Set by the handlers whose flags light_flags handles completely and that read no flags.
    bool light_ok = false;
    std::deque<LinkSlot> link_slots;
    u32 instruction_cycles = 0;
    u64 program_write_counter = 0;
    u16 compiled_cpc = 1;
    using BlockList = std::vector<std::pair<BlockKey, Block>>;
    std::unique_ptr<BlockList[]> block_cache;
    std::vector<u32> cached_pcs;
    std::stack<u32> call_stack;
    std::vector<u32> visited_pcs;
    Block* current_blk{};
    BlockKey blk_key{};
    BlockKey entry_key{};
    u32 block_start_pc = 0;
    Xbyak::Label* block_body = nullptr;

    struct FarCode {
        Xbyak::Label entry;
        Xbyak::Label ret;
        std::function<void()> emit;
    };
    std::vector<std::unique_ptr<FarCode>> far_code;

    void Reset() {
        regs.Reset();
        requests.Reset();
        bkrep_end_locations.clear();
        rep_end_locations.clear();
        ClearCache();
        SyncMemoryInterfaceUnit();
        program_write_counter = mem.GetProgramWriteCounter();
        compiled_cpc = regs.cpc;
        state_dirty = false;
        cycles_remaining = 0;
        pending_ticks = 0;
        UpdateTicksToEvent();
    }

    void UpdateTicksToEvent() {
        ticks_to_event = core_timing.GetMaxSkip(std::numeric_limits<u64>::max());
    }

    void FlushTicks() {
        if (pending_ticks != 0) {
            core_timing.Tick(std::exchange(pending_ticks, 0));
        }
        UpdateTicksToEvent();
    }

    void AddTicks(u64 ticks) {
        pending_ticks += ticks;
        if (pending_ticks > ticks_to_event) {
            FlushTicks();
        }
    }

    void ClearOverrun() {
        cycles_remaining = 0;
    }

    void Run(u64 cycles) {
        if (state_dirty) {
            LoadJitRegisters(regs, state);
            state_dirty = false;
        }
        SyncMemoryInterfaceUnit();
        // Cycles that the previous Run went over are taken from this one.
        cycles_remaining = static_cast<s64>(cycles) + std::min<s64>(cycles_remaining, 0);
        current_blk = nullptr;
        regs.idle = false;
        UpdateTicksToEvent();
        run_code(this);
        FlushTicks();
        if (pending_exception) {
            cycles_remaining = 0;
            std::rethrow_exception(std::exchange(pending_exception, nullptr));
        }
    }

    void SaveState() {
        if (!state_dirty) {
            SaveJitRegisters(regs, state);
        }
        state_dirty = true;
    }

    void ClearCache() {
        for (const u32 pc : cached_pcs) {
            block_cache[pc].clear();
        }
        cached_pcs.clear();
        link_slots.clear();
        regs.exit_slot = nullptr;
        c.setSize(dispatcher_size);
        flush_pending = false;
    }

    void AddBkrepEnd(u32 address) {
        if (bkrep_end_locations.insert(address).second) {
            flush_pending = true;
        }
    }

    void AddRepEnd(u32 address) {
        if (rep_end_locations.insert(address).second) {
            flush_pending = true;
        }
    }

    void SyncMemoryInterfaceUnit() {
        const MemoryInterfaceUnit& miu = mem.GetMemoryInterfaceUnit();
        u8* const raw = mem.GetSharedMemory().raw;
        const auto page_pointer = [raw](u16 page) {
            return raw + (MemoryInterfaceUnit::DataMemoryOffset +
                          (page & 1) * MemoryInterfaceUnit::DataMemoryBankSize) *
                             2;
        };
        regs.mmio_base = miu.mmio_base;
        if (miu.page_mode == 0) {
            regs.data_lo = page_pointer(miu.z_page);
            regs.data_hi = regs.data_lo;
            regs.data_split = 0xFFFF;
        } else {
            regs.data_lo = page_pointer(miu.x_page);
            regs.data_hi = page_pointer(miu.y_page);
            regs.data_split = static_cast<u16>(
                std::min<u32>(miu.x_size[0] * MemoryInterfaceUnit::XYSizeResolution, 0xFFFF));
        }
    }

    void EmitDispatcher() {
        run_code = c.getCurr<RunCodeFuncType>();
        ABI_PushRegistersAndAdjustStack(c, ABI_ALL_CALLEE_SAVED, 8, 16);

        Xbyak::Label dispatcher_start, dispatcher_end;
        c.L(dispatcher_start);

        // Look up (or compile) the block at the current pc. A null result means the cycle budget
        // is spent.
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(this));
        CallFarFunction(c, LookupNewBlockThunk);
        c.test(ABI_RETURN, ABI_RETURN);
        c.jz(dispatcher_end);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(&regs));
        c.jmp(ABI_RETURN);
        c.L(block_exit);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(this));
        CallFarFunction(c, BlockExitThunk);
        c.test(ABI_RETURN, ABI_RETURN);
        c.jz(dispatcher_end);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(&regs));
        c.jmp(ABI_RETURN);
        c.L(dispatcher_end);
        ABI_PopRegistersAndAdjustStack(c, ABI_ALL_CALLEE_SAVED, 8, 16);
        c.ret();
        c.ready();
        dispatcher_size = c.getSize();
    }

    static BlockFunc LookupNewBlockThunk(void* this_ptr) {
        return reinterpret_cast<EmitX64*>(this_ptr)->LookupNewBlock();
    }

    static BlockFunc BlockExitThunk(void* this_ptr) {
        auto* self = reinterpret_cast<EmitX64*>(this_ptr);
        self->DoInterruptsAndTick();
        return self->LookupNewBlock();
    }

    BlockFunc LookupNewBlock() {
        if (cycles_remaining <= 0) {
            return nullptr;
        }

        const u64 counter = mem.GetProgramWriteCounter();
        if (counter != program_write_counter) {
            program_write_counter = counter;
            flush_pending = true;
        }
        if (regs.cpc != compiled_cpc) {
            compiled_cpc = regs.cpc;
            flush_pending = true;
        }
        if (flush_pending || c.getSize() + CodeCacheMargin > MaxCodeSize) {
            ClearCache();
        }

        // State for bank exchange.
        blk_key.pc = regs.pc;
        std::memcpy(&blk_key.cfgi, &regs.cfgi, sizeof(u16) * 8);
        // Current state
        std::memcpy(&blk_key.curr.mod1, &regs.mod1, sizeof(u16) * 3);
        std::memcpy(&blk_key.curr.arp, &regs.arp, sizeof(regs.arp) + sizeof(regs.ar));
        // Shadow state.
        std::memcpy(&blk_key.shadow.mod1, &regs.mod1b, sizeof(u16) * 3);
        std::memcpy(&blk_key.shadow.arp, &regs.arpb, sizeof(regs.arpb) + sizeof(regs.arb));

        BlockKey key = blk_key;
        key.pc = 0;
        LookupBlock();
        if (auto* slot = static_cast<LinkSlot*>(std::exchange(regs.exit_slot, nullptr));
            slot && current_blk->body && slot->key == key) {
            slot->entries[slot->next] = {regs.pc, current_blk->body};
            slot->next ^= 1;
        }

        requests.Collect();
        // Check if we are idle, and skip ahead, unless a request is on its way to the core
        if (regs.idle && !requests.Any()) {
            FlushTicks();
            u64 skipped = core_timing.Skip(cycles_remaining - 1);
            cycles_remaining -= skipped;
            // Skip additional tick so to let components fire interrupts
            if (cycles_remaining > 1) {
                cycles_remaining--;
                core_timing.Tick();
            }
            UpdateTicksToEvent();
            requests.Collect();
        }

        const bool interrupt_pending =
            regs.ie && ((regs.im[0] && regs.ip[0]) || (regs.im[1] && regs.ip[1]) ||
                        (regs.im[2] && regs.ip[2]) || (regs.imv && regs.ipv));
        const s64 next_event = static_cast<s64>(std::min<u64>(
            std::min<u64>(ticks_to_event - pending_ticks, LoopBudget) + 1, requests.NextArrival()));
        regs.budget = interrupt_pending || flush_pending
                          ? 0
                          : static_cast<s32>(std::min({cycles_remaining, LoopBudget, next_event}));
        regs.budget_start = regs.budget;
        return current_blk->func;
    }

    static void DoInterruptsAndTickThunk(void* this_ptr) {
        reinterpret_cast<EmitX64*>(this_ptr)->DoInterruptsAndTick();
    }

    FORCE_INLINE void LookupBlock() {
        auto& vec = block_cache[regs.pc];
        for (auto& [key, block] : vec) {
            if (key == blk_key) {
                current_blk = &block;
                return;
            }
        }

        if (vec.empty()) {
            cached_pcs.push_back(regs.pc);
        }
        auto& [key, blk] = vec.emplace_back();
        key = blk_key;
        current_blk = &blk;
        CompileBlock(blk);
    }

    void DoInterruptsAndTick() {
        const u64 cycles = static_cast<s64>(regs.budget_start) - regs.budget;
        AddTicks(cycles - std::exchange(regs.ticked_cycles, 0));
        cycles_remaining -= cycles;

        // Requests raised while the block ran start counting at its end.
        const u32 arrived = requests.Advance(cycles);
        for (u32 i = 0; i < 3; ++i) {
            if (arrived & (1u << i)) {
                regs.ip[i] = 1;
            }
        }
        if (arrived & (1u << InterruptRequests::Vectored)) {
            regs.ipv = 1;
        }
        requests.Collect();

        // I am not sure if a single instruction loop is interruptable and how it is handled,
        // so just disable interrupt for it for now.
        if (regs.ie && !regs.rep) {
            bool interrupt_handled = false;
            for (u32 i = 0; i < regs.im.size(); ++i) {
                if (regs.im[i] && regs.ip[i]) {
                    regs.ip[i] = 0;
                    regs.ie = 0;
                    PushPC();
                    regs.pc = 0x0006 + i * 8;
                    regs.idle = false;
                    interrupt_handled = true;
                    if (regs.ic[i]) {
                        RunContextStore();
                    }
                    break;
                }
            }
            if (!interrupt_handled && regs.imv && regs.ipv) {
                regs.ipv = 0;
                regs.ie = 0;
                PushPC();
                regs.pc = vinterrupt_address;
                regs.idle = false;
                interrupt_handled = true;
                if (vinterrupt_context_switch) {
                    RunContextStore();
                }
            }
            if (interrupt_handled) {
                AddTicks(Cycles::InterruptEntry);
                cycles_remaining -= Cycles::InterruptEntry;
            }
        }
    }

    void RunContextStore() {
        SaveJitRegisters(regs, state);
        interpreter.ContextStore();
        LoadJitRegisters(regs, state);
    }

    static void FallbackThunk(void* this_ptr) {
        reinterpret_cast<EmitX64*>(this_ptr)->Fallback();
    }

    void Fallback() {
        ++fallback_count;
        FlushTicks();
        SaveJitRegisters(regs, state);
        // Exceptions cannot unwind through emitted code, stop the dispatcher and rethrow from Run.
        try {
            regs.budget -= interpreter.Step() + mem.TakeMmioAccesses() * Cycles::MmioAccess;
        } catch (...) {
            pending_exception = std::current_exception();
            cycles_remaining = 0;
        }
        LoadJitRegisters(regs, state);
        SyncMemoryInterfaceUnit();
        UpdateTicksToEvent();
        if (regs.rep) {
            AddRepEnd(regs.pc);
        }
        if (regs.lp) {
            for (u16 i = 0; i < regs.bcn; ++i) {
                AddBkrepEnd(regs.bkrep_stack[i].end);
            }
        }
    }

    void EmitBlockEntry() {
        c.mov(REGS, ABI_PARAM1);
        c.mov(R0_1_2_3, qword[REGS + offsetof(JitRegisters, r)]);
        c.mov(R4_5_6_7, qword[REGS + offsetof(JitRegisters, r) + sizeof(u16) * 4]);
        c.mov(FACTORS, qword[REGS + offsetof(JitRegisters, y)]);
        c.mov(A[0], qword[REGS + offsetof(JitRegisters, a)]);
        c.mov(A[1], qword[REGS + offsetof(JitRegisters, a) + sizeof(u64)]);
        c.mov(B[0], qword[REGS + offsetof(JitRegisters, b)]);
        c.mov(B[1], qword[REGS + offsetof(JitRegisters, b) + sizeof(u64)]);
        c.movzx(FLAGS, word[REGS + offsetof(JitRegisters, flags)]);
    }

    void CompileBlock(Block& blk) {
        const u32 start_pc = regs.pc;
        const std::size_t block_start = c.getSize();
        blk.func = c.getCurr<BlockFunc>();
        blk.cycles = 0;
        EmitBlockEntry();
        blk.body = c.getCurr();
        Xbyak::Label body;
        c.L(body);
        link_ok = true;
        block_body = &body;
        block_start_pc = start_pc;
        entry_key = blk_key;

        call_stack = {};
        visited_pcs.clear();
        far_code.clear();
        u32 instructions = 0;

        EmitLightRepetitions();

        compiling = true;
        while (compiling) {
            // Following static jumps can loop back into the block, so stop there and also cap the
            // block length to keep interrupt latency bounded.
            if (instructions >= max_block_instructions ||
                std::find(visited_pcs.begin(), visited_pcs.end(), regs.pc) != visited_pcs.end()) {
                StorePC(regs.pc);
                break;
            }
            visited_pcs.push_back(regs.pc);

            const u32 current_pc = regs.pc;
            const u16 opcode = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
            const auto& decoder = decoders[opcode];
            u16 expand_value = 0;
            if (decoder.NeedExpansion()) {
                expand_value = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
            }
            const u32 next_pc = regs.pc;

            const std::size_t code_pos = c.getSize();
            const std::size_t far_code_count = far_code.size();
            const BlockKey key_backup = blk_key;
            const auto call_stack_backup = call_stack;
            const s32 cycles_backup = blk.cycles;
            unimplemented = false;
            pc_written = false;
            instruction_cycles = 0;
            try {
                decoder.call(*this, opcode, expand_value);
            } catch (const Xbyak::Error&) {
                unimplemented = true;
            }

            if (unimplemented) {
                c.setSize(code_pos);
                far_code.resize(far_code_count);
                blk_key = key_backup;
                call_stack = call_stack_backup;
                blk.cycles = cycles_backup;
                if (instructions == 0) {
                    c.setSize(block_start);
                    far_code.clear();
                    EmitFallbackBlock(blk);
                    regs.pc = start_pc;
                    return;
                }
                compiling = true;
                StorePC(current_pc);
                break;
            }

            ++instructions;
            blk.cycles += InstructionCycles(decoder);

            const bool rep_end = rep_end_locations.contains(current_pc);
            const bool bkrep_end = bkrep_end_locations.contains(next_pc - 1);
            if (rep_end || bkrep_end) {
                EmitLoopEnd(current_pc, next_pc, rep_end, bkrep_end, blk.cycles);
            } else if (!compiling && !pc_written) {
                StorePC(regs.pc);
            }
        }

        EmitBlockEnd(blk.cycles);
        EmitFarCode();
        regs.pc = start_pc;
    }

    // A block of one instruction under rep repeats it in emitted code. A repetition that is
    // followed by another one runs a variant that computes only the sticky flags, because the next
    // one overwrites the others. The last repetition always runs the full code below.
    void EmitLightRepetitions() {
        const u32 start_pc = regs.pc;
        if (!rep_end_locations.contains(start_pc)) {
            return;
        }
        const u16 opcode = mem.ProgramRead(start_pc | (regs.prpage << 18));
        const auto& decoder = decoders[opcode];
        const u32 length = decoder.NeedExpansion() ? 2 : 1;
        if (bkrep_end_locations.contains(start_pc + length - 1)) {
            return;
        }
        const u16 expand_value =
            decoder.NeedExpansion() ? mem.ProgramRead((start_pc + 1) | (regs.prpage << 18)) : 0;

        const BlockKey key_backup = blk_key;
        const auto call_stack_backup = call_stack;
        const std::size_t far_code_count = far_code.size();
        // Emits the light variant. Returns its cycles, or 0 if the instruction has none.
        const auto emit = [&]() -> u32 {
            regs.pc = start_pc + length;
            light_flags = true;
            light_ok = false;
            unimplemented = false;
            instruction_cycles = 0;
            try {
                decoder.call(*this, opcode, expand_value);
            } catch (const Xbyak::Error&) {
                unimplemented = true;
            }
            light_flags = false;
            const bool ok = light_ok && !unimplemented && blk_key == entry_key;
            blk_key = key_backup;
            call_stack = call_stack_backup;
            regs.pc = start_pc;
            return ok ? InstructionCycles(decoder) : 0;
        };

        const std::size_t code_pos = c.getSize();
        const u32 cycles = emit();
        c.setSize(code_pos);
        far_code.resize(far_code_count);
        if (cycles == 0) {
            return;
        }

        Xbyak::Label full;
        c.test(byte[REGS + offsetof(JitRegisters, rep)], 1);
        c.jz(full, Xbyak::CodeGenerator::T_NEAR);
        c.cmp(word[REGS + offsetof(JitRegisters, repc)], 0);
        c.je(full, Xbyak::CodeGenerator::T_NEAR);
        c.cmp(dword[REGS + offsetof(JitRegisters, budget)], cycles);
        c.jle(full, Xbyak::CodeGenerator::T_NEAR);
        emit();
        c.sub(word[REGS + offsetof(JitRegisters, repc)], 1);
        c.sub(dword[REGS + offsetof(JitRegisters, budget)], cycles);
        c.jmp(*block_body);
        c.L(full);
    }

    void EmitFallbackBlock(Block& blk) {
        blk.func = c.getCurr<BlockFunc>();
        blk.body = nullptr;
        blk.cycles = 0;
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(this));
        CallFarFunction(c, FallbackThunk);
        c.jmp(block_exit);
    }

    // The cycles of the instruction just compiled: its length unless a handler set a total.
    u32 InstructionCycles(const Matcher<EmitX64>& decoder) const {
        return instruction_cycles ? instruction_cycles : (decoder.NeedExpansion() ? 2 : 1);
    }

    // A conditional control flow instruction counts its not taken cycles when compiled. The taken
    // path adds the returned difference with AddCycles.
    u32 ControlFlowCycles(Cond cond, u32 words, u32 taken) {
        if (cond.GetName() == CondValue::True) {
            instruction_cycles = taken;
            return 0;
        }
        instruction_cycles = words + Cycles::NotTaken;
        return taken - instruction_cycles;
    }

    void AddCycles(u32 cycles) {
        if (cycles != 0) {
            c.sub(dword[REGS + offsetof(JitRegisters, budget)], cycles);
        }
    }

    void StorePC(u32 pc) {
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], pc);
        pc_written = true;
    }

    void StorePC(Reg32 pc) {
        c.mov(dword[REGS + offsetof(JitRegisters, pc)], pc);
        pc_written = true;
    }

    // Subtracts the block's cycles from the budget, then jumps to the next block if it is linked
    // and the budget is enough, or returns to the dispatcher.
    void EmitBlockEnd(s32 cycles) {
        Xbyak::Label exit;
        c.sub(dword[REGS + offsetof(JitRegisters, budget)], cycles);
        c.jle(exit);
        if (link_ok) {
            LinkSlot& slot = link_slots.emplace_back();
            slot.key = blk_key;
            slot.key.pc = 0;
            Xbyak::Label second, miss;
            c.mov(rbp, reinterpret_cast<uintptr_t>(&slot));
            c.mov(eax, dword[REGS + offsetof(JitRegisters, pc)]);
            c.cmp(eax, dword[rbp + offsetof(LinkSlot, entries)]);
            c.jne(second);
            c.jmp(ptr[rbp + offsetof(LinkSlot, entries) + offsetof(LinkSlot::Entry, code)]);
            c.L(second);
            c.cmp(eax, dword[rbp + offsetof(LinkSlot, entries) + sizeof(LinkSlot::Entry)]);
            c.jne(miss);
            c.jmp(ptr[rbp + offsetof(LinkSlot, entries) + sizeof(LinkSlot::Entry) +
                      offsetof(LinkSlot::Entry, code)]);
            c.L(miss);
            c.mov(qword[REGS + offsetof(JitRegisters, exit_slot)], rbp);
        }
        c.L(exit);
        EmitBlockExit();
    }

    void EmitBlockExit() {
        c.mov(qword[REGS + offsetof(JitRegisters, r)], R0_1_2_3);
        c.mov(qword[REGS + offsetof(JitRegisters, r) + sizeof(u16) * 4], R4_5_6_7);
        c.mov(qword[REGS + offsetof(JitRegisters, y)], FACTORS);
        c.mov(qword[REGS + offsetof(JitRegisters, a)], A[0]);
        c.mov(qword[REGS + offsetof(JitRegisters, a) + sizeof(u64)], A[1]);
        c.mov(qword[REGS + offsetof(JitRegisters, b)], B[0]);
        c.mov(qword[REGS + offsetof(JitRegisters, b) + sizeof(u64)], B[1]);
        c.mov(word[REGS + offsetof(JitRegisters, flags)], FLAGS.cvt16());
        c.jmp(block_exit);
    }

    FarCode& NewFarCode(std::function<void()> emit) {
        far_code.push_back(std::make_unique<FarCode>());
        far_code.back()->emit = std::move(emit);
        return *far_code.back();
    }

    void EmitFarCode() {
        for (auto& code : far_code) {
            c.L(code->entry);
            code->emit();
            c.jmp(code->ret, Xbyak::CodeGenerator::T_NEAR);
        }
        far_code.clear();
    }

    // Ends the block after a loop end. Either the last instruction of a block repeat or an
    // instruction under rep. The pc defaults to the next instruction and is redirected when
    // the loop continues.
    // Jumps back to the start of the block if the loop restarts there with the same static state
    // and the budget is enough. Falls through otherwise.
    void EmitLoopBack(s32 block_cycles) {
        if (!(blk_key == entry_key)) {
            return;
        }
        Xbyak::Label exit;
        c.sub(dword[REGS + offsetof(JitRegisters, budget)], block_cycles);
        c.jle(exit);
        c.jmp(*block_body);
        c.L(exit);
        // The block end subtracts the cycles of this pass.
        c.add(dword[REGS + offsetof(JitRegisters, budget)], block_cycles);
    }

    void EmitLoopEnd(u32 current_pc, u32 next_pc, bool rep_end, bool bkrep_end,
                     s32 block_cycles) {
        Xbyak::Label end_label;
        StorePC(next_pc);
        if (rep_end) {
            Xbyak::Label rep_done;
            c.test(byte[REGS + offsetof(JitRegisters, rep)], 1);
            c.jz(rep_done);
            c.cmp(word[REGS + offsetof(JitRegisters, repc)], 0);
            Xbyak::Label rep_finished;
            c.je(rep_finished);
            c.sub(word[REGS + offsetof(JitRegisters, repc)], 1);
            if (current_pc == block_start_pc) {
                EmitLoopBack(block_cycles);
            }
            StorePC(current_pc);
            c.jmp(end_label, Xbyak::CodeGenerator::T_NEAR);
            c.L(rep_finished);
            c.mov(byte[REGS + offsetof(JitRegisters, rep)], 0);
            c.L(rep_done);
        }
        if (bkrep_end) {
            using Frame = JitRegisters::BlockRepeatFrame;
            static_assert(sizeof(Frame) == 12);
            Xbyak::Label jump_to_start;
            const Reg64 bcn = rax;
            const Reg64 frame = rbx;
            c.test(word[REGS + offsetof(JitRegisters, lp)], 0x1);
            c.jz(end_label, Xbyak::CodeGenerator::T_NEAR);
            c.movzx(bcn, word[REGS + offsetof(JitRegisters, bcn)]);
            c.sub(bcn, 1);
            c.lea(frame, ptr[bcn + bcn * 2]);
            c.lea(frame, ptr[REGS + offsetof(JitRegisters, bkrep_stack) + frame * 4]);
            c.cmp(dword[frame + offsetof(Frame, end)], next_pc - 1);
            c.jne(end_label, Xbyak::CodeGenerator::T_NEAR);
            c.cmp(word[frame + offsetof(Frame, lc)], 0);
            c.jne(jump_to_start);
            c.mov(word[REGS + offsetof(JitRegisters, bcn)], bcn.cvt16());
            c.test(bcn, bcn);
            c.setnz(byte[REGS + offsetof(JitRegisters, lp)]);
            c.jmp(end_label);
            c.L(jump_to_start);
            c.sub(word[frame + offsetof(Frame, lc)], 1);
            c.mov(frame.cvt32(), dword[frame + offsetof(Frame, start)]);
            Xbyak::Label other_start;
            c.cmp(frame.cvt32(), block_start_pc);
            c.jne(other_start);
            EmitLoopBack(block_cycles);
            c.L(other_start);
            StorePC(frame.cvt32());
        }
        c.L(end_label);
        compiling = false;
    }

    void EmitPushPC() {
        u16 l = (u16)(regs.pc & 0xFFFF);
        u16 h = (u16)(regs.pc >> 16);
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        if (regs.cpc == 1) {
            StoreToMemory(sp, h);
            c.sub(sp, 1);
            StoreToMemory(sp, l);
        } else {
            StoreToMemory(sp, l);
            c.sub(sp, 1);
            StoreToMemory(sp, h);
        }
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    void PushPC() {
        u16 l = (u16)(regs.pc & 0xFFFF);
        u16 h = (u16)(regs.pc >> 16);
        if (regs.cpc == 1) {
            mem.DataWrite(--regs.sp, h);
            mem.DataWrite(--regs.sp, l);
        } else {
            mem.DataWrite(--regs.sp, l);
            mem.DataWrite(--regs.sp, h);
        }
        SyncMemoryInterfaceUnit();
    }

    void EmitPopPC() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 pc = rcx;
        c.xor_(pc, pc);
        if (regs.cpc == 1) {
            LoadFromMemory(pc, sp);
            c.add(sp, 1);
            c.ror(pc.cvt32(), 16);
            LoadFromMemory(pc, sp);
            c.add(sp, 1);
            c.rol(pc.cvt32(), 16);
        } else {
            LoadFromMemory(pc, sp);
            c.add(sp, 1);
            c.shl(pc, 16);
            LoadFromMemory(pc, sp);
            c.add(sp, 1);
        }
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
        StorePC(pc.cvt32());
    }

    void SignalInterrupt(u32 i, s32 source_delay = 0) {
        requests.Signal(i, source_delay);
    }
    void SignalVectoredInterrupt(u32 address, bool context_switch, s32 source_delay = 0) {
        vinterrupt_address = address;
        vinterrupt_context_switch = context_switch;
        requests.Signal(InterruptRequests::Vectored, source_delay);
    }

    using instruction_return_type = void;

    InterruptRequests requests;
    bool vinterrupt_context_switch = false;
    u32 vinterrupt_address = 0;

    void nop() {
        // literally nothing
    }

    void undefined(u16 opcode) {
        NOT_IMPLEMENTED();
    }

    void norm(Ax a, Rn b, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void swap(SwapType swap) {
        RegName s0, d0, s1, d1;
        const Reg64 u = rax, v = rbx;
        switch (swap.GetName()) {
        case SwapTypeValue::a0b0:
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a0b1:
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a1b0:
            s0 = d1 = RegName::a1;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a1b1:
            s0 = d1 = RegName::a1;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a0b0a1b1:
            GetAcc(u, RegName::a1);
            GetAcc(v, RegName::b1);
            SatAndSetAccAndFlag(RegName::a1, v);
            SatAndSetAccAndFlag(RegName::b1, u);
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b0;
            break;
        case SwapTypeValue::a0b1a1b0:
            GetAcc(u, RegName::a1);
            GetAcc(v, RegName::b0);
            SatAndSetAccAndFlag(RegName::a1, v);
            SatAndSetAccAndFlag(RegName::b0, u);
            s0 = d1 = RegName::a0;
            s1 = d0 = RegName::b1;
            break;
        case SwapTypeValue::a0b0a1:
            s0 = RegName::a0;
            d0 = s1 = RegName::b0;
            d1 = RegName::a1;
            break;
        case SwapTypeValue::a0b1a1:
            s0 = RegName::a0;
            d0 = s1 = RegName::b1;
            d1 = RegName::a1;
            break;
        case SwapTypeValue::a1b0a0:
            s0 = RegName::a1;
            d0 = s1 = RegName::b0;
            d1 = RegName::a0;
            break;
        case SwapTypeValue::a1b1a0:
            s0 = RegName::a1;
            d0 = s1 = RegName::b1;
            d1 = RegName::a0;
            break;
        case SwapTypeValue::b0a0b1:
            s0 = d1 = RegName::a0;
            d0 = RegName::b1;
            s1 = RegName::b0;
            break;
        case SwapTypeValue::b0a1b1:
            s0 = d1 = RegName::a1;
            d0 = RegName::b1;
            s1 = RegName::b0;
            break;
        case SwapTypeValue::b1a0b0:
            s0 = d1 = RegName::a0;
            d0 = RegName::b0;
            s1 = RegName::b1;
            break;
        case SwapTypeValue::b1a1b0:
            s0 = d1 = RegName::a1;
            d0 = RegName::b0;
            s1 = RegName::b1;
            break;
        default:
            UNREACHABLE();
        }
        GetAcc(u, s0);
        GetAcc(v, s1);
        SatAndSetAccAndFlag(d0, u);
        SatAndSetAccAndFlag(d1, v); // only this one affects flags (except for fl)
    }
    void trap() {
        NOT_IMPLEMENTED();
    }

    static void ReadThunk(void* this_ptr) {
        auto* self = reinterpret_cast<EmitX64*>(this_ptr);
        self->CatchUpTiming();
        self->regs.tmp_value = self->mem.DataRead(self->regs.tmp_addr);
        self->regs.budget -= self->mem.TakeMmioAccesses() * Cycles::MmioAccess;
        self->UpdateTicksToEvent();
    }

    static void WriteThunk(void* this_ptr) {
        auto* self = reinterpret_cast<EmitX64*>(this_ptr);
        self->CatchUpTiming();
        self->mem.DataWrite(self->regs.tmp_addr, self->regs.tmp_value);
        self->regs.budget -= self->mem.TakeMmioAccesses() * Cycles::MmioAccess;
        // The write can raise an interrupt or change program memory, so return to the dispatcher at
        // the end of the block.
        self->EndChain();
        self->SyncMemoryInterfaceUnit();
        self->UpdateTicksToEvent();
    }

    // Makes the blocks return to the dispatcher at the end of the current one, without changing the
    // cycles run (budget_start - budget).
    void EndChain() {
        if (regs.budget > 0) {
            regs.budget_start -= regs.budget;
            regs.budget = 0;
        }
    }

    // Ticks the hardware up to the instruction making an MMIO access, so
    // that timers and FIFOs that are read inside a block are up to date.
    void CatchUpTiming() {
        const u32 elapsed = static_cast<u32>(regs.budget_start - regs.budget) + regs.tmp_cycles +
                            mem.PeekMmioAccesses() * Cycles::MmioAccess;
        if (elapsed > regs.ticked_cycles) {
            pending_ticks += elapsed - regs.ticked_cycles;
            regs.ticked_cycles = elapsed;
        }
        FlushTicks();
    }

    // Calls out of a block. Block code treats every host register as preserved, so everything the
    // callee may clobber is saved, including rax.
    void EmitSlowCall(void (*func)(void*)) {
        std::bitset<32> saved = ABI_ALL_CALLER_SAVED_GPR;
        saved.set(RegToIndex(rax));
        ABI_PushRegistersAndAdjustStack(c, saved, 0);
        c.mov(ABI_PARAM1, reinterpret_cast<uintptr_t>(this));
        CallFarFunction(c, func);
        ABI_PopRegistersAndAdjustStack(c, saved, 0);
    }

    // Data accesses use rbp as their only scratch register. A register address is zero extended
    // in place from its low 16 bits.
    template <typename T>
    void EmitMmioCheck(T addr, Xbyak::Label& mmio) {
        if constexpr (std::is_integral_v<T>) {
            c.mov(ebp, static_cast<u32>(addr & 0xFFFF));
        } else {
            c.movzx(addr.cvt32(), addr.cvt16());
            c.mov(ebp, addr.cvt32());
        }
        c.sub(ebp, dword[REGS + offsetof(JitRegisters, mmio_base)]);
        c.cmp(ebp, MemoryInterfaceUnit::MMIOSize);
        c.jb(mmio);
    }

    template <typename T>
    Xbyak::Address EmitDataPointer(T addr) {
        c.mov(rbp, qword[REGS + offsetof(JitRegisters, data_lo)]);
        if constexpr (std::is_integral_v<T>) {
            const u32 address = static_cast<u32>(addr & 0xFFFF);
            c.cmp(word[REGS + offsetof(JitRegisters, data_split)], address);
            c.cmovb(rbp, qword[REGS + offsetof(JitRegisters, data_hi)]);
            return word[rbp + address * 2];
        } else {
            c.cmp(addr.cvt16(), word[REGS + offsetof(JitRegisters, data_split)]);
            c.cmova(rbp, qword[REGS + offsetof(JitRegisters, data_hi)]);
            return word[rbp + addr * 2];
        }
    }

    template <typename T>
    void EmitStoreTmpAddr(T addr) {
        if constexpr (std::is_integral_v<T>) {
            c.mov(word[REGS + offsetof(JitRegisters, tmp_addr)], static_cast<u32>(addr & 0xFFFF));
        } else {
            c.mov(word[REGS + offsetof(JitRegisters, tmp_addr)], addr.cvt16());
        }
    }

    void LoadFromMemory(Reg64 out, MemImm8 addr) {
        LoadFromMemory(out, addr.Unsigned16() + (blk_key.curr.mod1.page << 8));
    }

    // Writes the low 16 bits of out and leaves the rest of it untouched.
    template <typename T>
    void LoadFromMemory(Reg64 out, T addr) {
        FarCode& mmio = NewFarCode([this, out, addr, cycles = current_blk->cycles] {
            EmitStoreTmpAddr(addr);
            c.mov(dword[REGS + offsetof(JitRegisters, tmp_cycles)], cycles);
            EmitSlowCall(ReadThunk);
            c.mov(out.cvt16(), word[REGS + offsetof(JitRegisters, tmp_value)]);
        });
        EmitMmioCheck(addr, mmio.entry);
        c.mov(out.cvt16(), EmitDataPointer(addr));
        c.L(mmio.ret);
    }

    void DoMultiplication(u32 unit, Reg32 x, Reg32 y, bool x_sign, bool y_sign) {
        c.mov(x.cvt64(), FACTORS);
        if (unit == 0) {
            c.movzx(y, FACTORS.cvt16());
            c.shr(x.cvt64(), 32);
            c.and_(x, 0xFFFF);
        } else {
            c.shr(x.cvt64(), 16);
            c.movzx(y, x.cvt16());
            c.shr(x.cvt64(), 32);
        }
        if (blk_key.curr.mod0.hwm == 1 || (blk_key.curr.mod0.hwm == 3 && unit == 0)) {
            c.shr(y, 8);
        } else if (blk_key.curr.mod0.hwm == 2 || (blk_key.curr.mod0.hwm == 3 && unit == 1)) {
            c.and_(y, 0xFF);
        }
        if (x_sign) {
            SignExtend(x, 16);
        }
        if (y_sign) {
            SignExtend(y, 16);
        }
        c.imul(x, y);
        c.mov(dword[REGS + offsetof(JitRegisters, p) + sizeof(u32) * unit], x);
        if (x_sign || y_sign) {
            c.shr(x, 31);
            c.mov(word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit], x.cvt16());
        } else {
            c.mov(word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit], 0);
        }
    }

    void AlmGeneric(AlmOp op, Reg64 a, Ax b) {
        const Reg64 value = rax;
        switch (op) {
        case AlmOp::Or: {
            GetAcc(value, b.GetName());
            c.or_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::And: {
            GetAcc(value, b.GetName());
            c.and_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::Xor: {
            GetAcc(value, b.GetName());
            c.xor_(value, a);
            SignExtend(value, 40);
            SetAccAndFlag(b.GetName(), value);
            break;
        }
        case AlmOp::Tst0: {
            GetAcc(value, b.GetName());
            c.and_(value, 0xFFFF);
            c.and_(FLAGS, ~decltype(Flags::fz)::mask); // clear fz
            c.test(value, a);
            c.setz(a.cvt8());
            c.shl(a, 15);
            c.shr(a, 15 - decltype(Flags::fz)::position);
            c.or_(FLAGS.cvt16(), a.cvt16()); // regs.fz = (value & a) == 0;
            break;
        }
        case AlmOp::Tst1: {
            GetAcc(value, b.GetName());
            c.and_(value, 0xFFFF);
            c.and_(FLAGS, ~decltype(Flags::fz)::mask); // clear fz
            c.not_(a);
            c.test(value, a);
            c.setz(a.cvt8());
            c.shl(a, 15);
            c.shr(a, 15 - decltype(Flags::fz)::position);
            c.or_(FLAGS, a.cvt32()); // regs.fz = (value & ~a) == 0;
            break;
        }
        case AlmOp::Cmp:
        case AlmOp::Cmpu:
        case AlmOp::Sub:
        case AlmOp::Subl:
        case AlmOp::Subh:
        case AlmOp::Add:
        case AlmOp::Addl:
        case AlmOp::Addh: {
            GetAcc(value, b.GetName());
            const bool sub = !(op == AlmOp::Add || op == AlmOp::Addl || op == AlmOp::Addh);
            const Reg64 result = rcx;
            AddSub(value, a, result, sub);
            if (op == AlmOp::Cmp || op == AlmOp::Cmpu) {
                SetAccFlag(result);
            } else {
                SatAndSetAccAndFlag(b.GetName(), result);
            }
            break;
        }
        case AlmOp::Msu: {
            NOT_IMPLEMENTED();
            GetAcc(value, b.GetName());
            const Reg64 product = rcx;
            ProductToBus40(product, Px{0});
            const Reg64 result = rdx;
            AddSub(value, product, result, true);
            SatAndSetAccAndFlag(b.GetName(), result);

            c.shl(a, 48);
            c.shr(a, 16);
            c.mov(result, ~decltype(Factors::x0)::mask);
            c.and_(FACTORS, result);
            c.or_(FACTORS, a); // regs.x[0] = a & 0xFFFF;
            DoMultiplication(0, eax, ebx, true, true);
            break;
        }
        case AlmOp::Sqra: {
            GetAcc(value, b.GetName());
            const Reg64 product = rdx;
            ProductToBus40(product, Px{0});
            const Reg64 result = rcx;
            AddSub(value, product, result, false);
            SatAndSetAccAndFlag(b.GetName(), result);
        }
            [[fallthrough]];
        case AlmOp::Sqr: {
            c.and_(a, 0xFFFF);
            c.mov(FACTORS.cvt16(), a.cvt16());
            c.ror(FACTORS, 32);
            c.mov(FACTORS.cvt16(), a.cvt16());
            c.rol(FACTORS, 32);
            DoMultiplication(0, eax, ebx, true, true);
            break;
        }

        default:
            NOT_IMPLEMENTED();
        }
    }

    void ExtendOperandForAlm(AlmOp op, Reg64 a) {
        switch (op) {
        case AlmOp::Cmp:
        case AlmOp::Sub:
        case AlmOp::Add:
            SignExtend(a, 16);
            break;
        case AlmOp::Addh:
        case AlmOp::Subh:
            c.shl(a, 16);
            SignExtend(a, 32);
            break;
        default:
            c.movzx(a.cvt64(), a.cvt16());
            break;
        }
    }

    u64 ExtendOperandForAlm(AlmOp op, u16 a) {
        switch (op) {
        case AlmOp::Cmp:
        case AlmOp::Sub:
        case AlmOp::Add:
            return ::SignExtend<16, u64>(a);
        case AlmOp::Addh:
        case AlmOp::Subh:
            return ::SignExtend<32, u64>((u64)a << 16);
        default:
            return a;
        }
    }

    void alm(Alm op, MemImm8 a, Ax b) {
        const Reg64 value = rbx;
        const Reg64 address = rcx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alm(Alm op, Rn a, StepZIDS as, Ax b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        LoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alm(Alm op, Register a, Ax b) {
        const Reg64 value = rbx;
        auto CheckBus40OperandAllowed = [&] {
            static const std::unordered_set<AlmOp> allowed_instruction{
                AlmOp::Or, AlmOp::And, AlmOp::Xor, AlmOp::Add, AlmOp::Cmp, AlmOp::Sub,
            };
            if (allowed_instruction.count(op.GetName()) == 0)
                NOT_IMPLEMENTED(); // weird effect. probably undefined
        };
        switch (a.GetName()) {
        // need more test
        case RegName::p:
            CheckBus40OperandAllowed();
            ProductToBus40(value, Px{0});
            break;
        case RegName::a0:
        case RegName::a1:
            CheckBus40OperandAllowed();
            GetAcc(value, a.GetName());
            break;
        default:
            RegToBus16(a.GetName(), value);
            ExtendOperandForAlm(op.GetName(), value);
            break;
        }
        AlmGeneric(op.GetName(), value, b);
    }
    void alm_r6(Alm op, Ax b) {
        NOT_IMPLEMENTED();
    }

    void alu(Alu op, MemImm16 a, Ax b) {
        const Reg64 value = rbx;
        const Reg64 address = rax;
        c.mov(address, a.Unsigned16());
        LoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alu(Alu op, MemR7Imm16 a, Ax b) {
        const Reg64 address = rax;
        const Reg64 value = rbx;
        RegToBus16(RegName::r7, address);
        c.movzx(address, address.cvt16());
        c.add(address, a.Unsigned16());
        LoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }
    void alu(Alu op, Imm16 a, Ax b) {
        u16 value = a.Unsigned16();
        c.mov(rbx, ExtendOperandForAlm(op.GetName(), value));
        AlmGeneric(op.GetName(), rbx, b);
    }
    void alu(Alu op, Imm8 a, Ax b) {
        u16 value = a.Unsigned16();
        const Reg64 and_backup = rsi;
        c.xor_(and_backup.cvt32(), and_backup.cvt32());
        if (op.GetName() == AlmOp::And) {
            // AND instruction has a special treatment:
            // bit 8~15 are unaffected in the accumulator, but the flags are set as if they are
            // affected
            GetAcc(and_backup, b.GetName());
            c.and_(and_backup, 0xFF00);
        }
        value = ExtendOperandForAlm(op.GetName(), value);
        c.mov(rbx, value);
        AlmGeneric(op.GetName(), rbx, b);
        if (op.GetName() == AlmOp::And) {
            const Reg64 and_new = rax;
            GetAcc(and_new, b.GetName());
            c.and_(and_new.cvt16(), 0x00FF);
            c.or_(and_new, and_backup);
            SetAcc(b.GetName(), and_new);
        }
    }
    void alu(Alu op, MemR7Imm7s a, Ax b) {
        const Reg64 address = rax;
        RegToBus16(RegName::r7, address);
        c.movzx(address, address.cvt16());
        c.add(address.cvt32(), a.Signed16());
        const Reg64 value = rbx;
        LoadFromMemory(value, address);
        ExtendOperandForAlm(op.GetName(), value);
        AlmGeneric(op.GetName(), value, b);
    }

    void or_(Ab a, Ax b, Ax c) {
        GetAcc(rax, a.GetName());
        GetAcc(rbx, b.GetName());
        this->c.or_(rax, rbx);
        SetAccAndFlag(c.GetName(), rax);
    }
    void or_(Ax a, Bx b, Ax c) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        this->c.or_(value, GetAccDirect(b.GetName()));
        SetAccAndFlag(c.GetName(), value);
    }
    void or_(Bx a, Bx b, Ax c) {
        NOT_IMPLEMENTED();
    }

    u16 GenericAlbConst(Alb op, u16 a, u16 b) {
        u16 result;
        switch (op.GetName()) {
        case AlbOp::Set: {
            result = a | b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Rst: {
            result = ~a & b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Chng: {
            result = a ^ b;
            if (result >> 15) {
                c.bts(FLAGS, decltype(Flags::fm)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fm)::position);
            }
            break;
        }
        case AlbOp::Addv: {
            u32 r = a + b;
            if ((r >> 16) != 0) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            if ((::SignExtend<16, u32>(b) + ::SignExtend<16, u32>(a)) >> 31) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            result = r & 0xFFFF;
            break;
        }
        case AlbOp::Tst0: {
            result = (a & b) != 0;
            break;
        }
        case AlbOp::Tst1: {
            result = (a & ~b) != 0;
            break;
        }
        case AlbOp::Cmpv:
        case AlbOp::Subv: {
            u32 r = b - a;
            if ((r >> 16) != 0) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            if ((::SignExtend<16, u32>(b) + ::SignExtend<16, u32>(a)) >> 31) {
                c.bts(FLAGS, decltype(Flags::fc0)::position);
            } else {
                c.btr(FLAGS, decltype(Flags::fc0)::position);
            }
            result = r & 0xFFFF;
            break;
        }
        default:
            UNREACHABLE();
        }
        if (result == 0) {
            c.bts(FLAGS, decltype(Flags::fz)::position);
        } else {
            c.btr(FLAGS, decltype(Flags::fz)::position);
        }
        return result;
    }

    void GenericAlb(Alb op, u16 a, Reg16 b, Reg16 result) {
        switch (op.GetName()) {
        case AlbOp::Set: {
            c.or_(b.cvt32(), a);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS.cvt16(), b);
            break;
        }
        case AlbOp::Rst: {
            [[maybe_unused]] const u16 a_not = ~a;
            c.and_(b.cvt32(), a_not);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS.cvt16(), b);
            break;
        }
        case AlbOp::Chng: {
            c.xor_(b.cvt32(), a);
            c.mov(result, b);
            c.shr(b, 15);
            c.shl(b, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~decltype(Flags::fm)::mask);
            c.or_(FLAGS.cvt16(), b);
            break;
        }
        case AlbOp::Addv: {
            c.movsx(result.cvt32(), b);
            c.add(result.cvt32(), ::SignExtend<16, u32>(a));
            c.shr(result.cvt32(), 31);
            c.shl(result, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~(decltype(Flags::fm)::mask | decltype(Flags::fc0)::mask));
            c.or_(FLAGS.cvt16(), result);

            c.movzx(result.cvt32(), b);
            c.add(result.cvt32(), static_cast<u32>(a));
            c.mov(b.cvt32(), result.cvt32());
            c.test(b.cvt32(), ~0xFFFF);
            c.setnz(b.cvt8());
            c.shl(b.cvt8(), decltype(Flags::fc0)::position);
            c.or_(FLAGS.cvt8(), b.cvt8());
            break;
        }
        case AlbOp::Tst0: {
            c.xor_(result, result);
            c.and_(b.cvt32(), a);
            c.test(b, b);
            c.setnz(result.cvt8());
            break;
        }
        case AlbOp::Tst1: {
            c.xor_(result, result);
            c.not_(b);
            c.and_(b.cvt32(), a);
            c.test(b, b);
            c.setnz(result.cvt8());
            break;
        }
        case AlbOp::Cmpv:
        case AlbOp::Subv: {
            c.movsx(result.cvt32(), b);
            c.sub(result.cvt32(), ::SignExtend<16, u32>(a));
            c.shr(result.cvt32(), 31);
            c.shl(result, decltype(Flags::fm)::position);
            c.and_(FLAGS, ~(decltype(Flags::fm)::mask | decltype(Flags::fc0)::mask));
            c.or_(FLAGS.cvt16(), result);

            c.movzx(result.cvt32(), b);
            c.sub(result.cvt32(), static_cast<u32>(a));
            c.mov(b.cvt32(), result.cvt32());
            c.test(b.cvt32(), ~0xFFFF);
            c.setnz(b.cvt8());
            c.shl(b.cvt8(), decltype(Flags::fc0)::position);
            c.or_(FLAGS.cvt8(), b.cvt8());
            break;
        }
        default:
            UNREACHABLE();
        }
        c.xor_(b, b);
        c.test(result.cvt16(), result.cvt16());
        c.setz(b.cvt8());
        c.shl(b, decltype(Flags::fz)::position);
        c.and_(FLAGS, ~decltype(Flags::fz)::mask);
        c.or_(FLAGS.cvt16(), b);
    }

    static bool IsAlbModifying(Alb op) {
        switch (op.GetName()) {
        case AlbOp::Set:
        case AlbOp::Rst:
        case AlbOp::Chng:
        case AlbOp::Addv:
        case AlbOp::Subv:
            return true;
        case AlbOp::Tst0:
        case AlbOp::Tst1:
        case AlbOp::Cmpv:
            return false;
        default:
            UNREACHABLE();
        }
    }

    static bool IsAlbConst(SttMod b) {
        switch (b.GetName()) {
        case RegName::mod0:
        case RegName::mod1:
        case RegName::mod2:
            return true;
        default:
            return false;
        }
    }

    void alb(Alb op, Imm16 a, MemImm8 b) {
        const Reg64 bv = rax;
        const Reg64 address = rbx;
        c.mov(address, b.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(bv, address);
        const Reg64 result = rbx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result.cvt16());
        if (IsAlbModifying(op)) {
            StoreToMemory(b.Unsigned16() + (blk_key.curr.mod1.page << 8), result);
        }
    }
    void alb(Alb op, Imm16 a, Rn b, StepZIDS bs) {
        const Reg64 address = rbx;
        RnAddressAndModify(b.Index(), bs.GetName(), address);
        const Reg64 bv = rax;
        LoadFromMemory(bv, address);
        const Reg64 result = rcx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result.cvt16());
        if (IsAlbModifying(op)) {
            StoreToMemory(address, result);
        }
    }
    void alb(Alb op, Imm16 a, Register b) {
        const Reg64 bv = rax;
        if (b.GetName() == RegName::p) {
            ProductToBus40(bv, Px{0});
            c.shr(bv, 16);
        } else if (b.GetName() == RegName::a0 || b.GetName() == RegName::a1) {
            NOT_IMPLEMENTED(); // weird effect;
        } else {
            RegToBus16(b.GetName(), bv);
        }
        const Reg64 result = rbx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result.cvt16());
        if (IsAlbModifying(op)) {
            switch (b.GetName()) {
            case RegName::a0:
            case RegName::a1:
                NOT_IMPLEMENTED();
            // operation on accumulators doesn't go through regular bus with flag and saturation
            case RegName::a0l:
                c.mov(A[0].cvt16(), result.cvt16());
                break;
            case RegName::a1l:
                c.mov(A[1].cvt16(), result.cvt16());
                break;
            case RegName::b0l:
                c.mov(B[0].cvt16(), result.cvt16());
                break;
            case RegName::b1l:
                c.mov(B[1].cvt16(), result.cvt16());
                break;
            case RegName::a0h:
                c.shl(result, 48);
                c.shr(result, 32);
                c.mov(rsi, 0xFFFF'FFFF'0000'FFFF);
                c.and_(A[0], rsi);
                c.or_(A[0], result);
                break;
            case RegName::a1h:
                c.shl(result, 48);
                c.shr(result, 32);
                c.mov(rsi, 0xFFFF'FFFF'0000'FFFF);
                c.and_(A[1], rsi);
                c.or_(A[1], result);
                break;
            case RegName::b0h:
                c.shl(result, 48);
                c.shr(result, 32);
                c.mov(rsi, 0xFFFF'FFFF'0000'FFFF);
                c.and_(B[0], rsi);
                c.or_(B[0], result);
                break;
            case RegName::b1h:
                c.shl(result, 48);
                c.shr(result, 32);
                c.mov(rsi, 0xFFFF'FFFF'0000'FFFF);
                c.and_(B[1], rsi);
                c.or_(B[1], result);
                break;
            default:
                RegFromBus16(b.GetName(), result.cvt64()); // including RegName:p (p0h)
            }
        }
    }
    void alb_r6(Alb op, Imm16 a) {
        NOT_IMPLEMENTED();
    }
    void alb(Alb op, Imm16 a, SttMod b) {
        if (IsAlbConst(b)) {
            u16 bv;
            switch (b.GetName()) {
            case RegName::mod0:
                bv = blk_key.curr.mod0.raw;
                break;
            case RegName::mod1:
                bv = blk_key.curr.mod1.raw;
                break;
            case RegName::mod2:
                bv = blk_key.curr.mod2.raw;
                break;
            default:
                UNREACHABLE();
            }
            u16 result = GenericAlbConst(op, a.Unsigned16(), bv);
            if (IsAlbModifying(op)) {
                RegFromBus16(b.GetName(), result);
            }
            return;
        }
        const Reg64 bv = rax;
        RegToBus16(b.GetName(), bv);
        const Reg16 result = bx;
        GenericAlb(op, a.Unsigned16(), bv.cvt16(), result);
        if (IsAlbModifying(op)) {
            RegFromBus16(b.GetName(), result.cvt64());
        }
    }

    void add(Ab a, Bx b) {
        const Reg64 value_a = rax;
        const Reg64 value_b = rbx;
        GetAcc(value_a, a.GetName());
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add(Bx a, Ax b) {
        const Reg64 value_a = rax;
        GetAcc(value_a, a.GetName());
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add_p1(Ax b) {
        const Reg64 value_a = rax;
        ProductToBus40(value_a, Px{1});
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void add(Px a, Bx b) {
        const Reg64 value_a = rax;
        ProductToBus40(value_a, a);
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, false);
        SatAndSetAccAndFlag(b.GetName(), result);
    }

    void sub(Ab a, Bx b) {
        const Reg64 value_a = rax;
        GetAcc(value_a, a.GetName());
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, true);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void sub(Bx a, Ax b) {
        const Reg64 value_a = rax;
        GetAcc(value_a, a.GetName());
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        const Reg64 result = rcx;
        AddSub(value_b, value_a, result, true);
        SatAndSetAccAndFlag(b.GetName(), result);
    }
    void sub_p1(Ax b) {
        NOT_IMPLEMENTED();
    }
    void sub(Px a, Bx b) {
        NOT_IMPLEMENTED();
    }

    void app(Ab c, SumBase base, bool sub_p0, bool p0_align, bool sub_p1, bool p1_align) {
        ProductSum(base, c.GetName(), sub_p0, p0_align, sub_p1, p1_align);
    }

    void add_add(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void add_sub(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_add(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_sub(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void add_sub_sv(ArRn1 a, ArStep1 as, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_add_sv(ArRn1 a, ArStep1 as, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_add_i_mov_j_sv(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void sub_add_j_mov_i_sv(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void add_sub_i_mov_j(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void add_sub_j_mov_i(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }

    template <typename T>
    constexpr size_t BitSize() {
        if constexpr (std::is_same_v<T, Reg64>) {
            return 64;
        } else if constexpr (std::is_same_v<T, Reg32>) {
            return 32;
        } else if constexpr (std::is_same_v<T, Reg16>) {
            return 16;
        } else {
            return sizeof(std::make_unsigned_t<T>) * CHAR_BIT;
        }
    }

    template <typename T>
    void SignExtend(T value, u32 bit_count) {
        if (bit_count == 16) {
            c.movsx(value, value.cvt16());
            return;
        }
        if (bit_count == 32) {
            if constexpr (std::is_same_v<T, Xbyak::Reg64>) {
                c.movsxd(value, value.cvt32());
                return;
            }
            NOT_IMPLEMENTED();
        }
        c.shl(value, BitSize<T>() - bit_count);
        c.sar(value, BitSize<T>() - bit_count);
    }

    void ShiftBus40(Reg64 value, u16 sv, RegName dest) {
        c.mov(rbx, 0xFF'FFFF'FFFF);
        c.xor_(ecx, ecx); // rcx = 0
        c.and_(value, rbx);
        const Reg64 original_sign = rbx;
        c.mov(original_sign, value);
        c.shr(original_sign, 39);
        if ((sv >> 15) == 0) {
            // left shift
            if (sv >= 40) {
                if (blk_key.curr.mod0.s == 0) {
                    // regs.fv = value != 0;
                    c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
                    c.test(value, value);
                    c.setne(cl); // u32 mask = (value != 0) ? 1 : 0
                    c.shl(cx, decltype(Flags::fv)::position); // mask <<= fv_pos
                    c.or_(FLAGS.cvt16(), cx); // flags |= mask
                    // if (regs.fv) {
                    //    regs.fvl = 1;
                    // }
                    static_assert(decltype(Flags::fv)::position - decltype(Flags::fvl)::position == 3);
                    // mask is either 0 or 1. If it was 0, then the or_ wont have any effect.
                    // If it was 1, it will set fvl, which is what we want.
                    c.shr(cx, 3); // mask >>= 3;
                    c.or_(FLAGS.cvt16(), cx);
                }
                c.xor_(value.cvt32(), value.cvt32()); // value = 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
            } else {
                if (blk_key.curr.mod0.s == 0) {
                    // TODO: This can probably be done better
                    c.mov(rdx, value);
                    c.mov(rcx, value);
                    SignExtend(rcx, 40);
                    SignExtend(rdx, 40 - sv);

                           // regs.fv = SignExtend<40>(value) != SignExtend(value, 40 - sv);
                    c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
                    c.cmp(rcx, rdx);
                    c.setne(cl); // u32 mask = (SignExtend<40>(value) != SignExtend(value, 40 - sv) ? 1 : 0
                    c.shl(cl, decltype(Flags::fv)::position); // mask <<= fv_pos
                    c.or_(FLAGS.cvt8(), cl); // flags |= mask
                    // if (regs.fv) {
                    //     regs.fvl = 1;
                    // }
                    c.shr(cl, 3); // mask >>= 3;
                    c.or_(FLAGS.cvt8(), cl);
                }
                c.shl(value, sv); // value <<= sv;
                // regs.fc0 = (value & ((u64)1 << 40)) != 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
                c.bt(value, 40);
                c.setc(cl); // u32 mask = (value & ((u64)1 << 40));
                c.shl(cl, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
                c.or_(FLAGS.cvt8(), cl);
            }
        } else {
            // right shift
            u16 nsv = ~sv + 1;
            if (nsv >= 40) {
                if (blk_key.curr.mod0.s == 0) {
                    NOT_IMPLEMENTED();
                    // regs.fc0 = (value >> 39) & 1;
                    // value = regs.fc0 ? 0xFF'FFFF'FFFF : 0;
                } else {
                    c.xor_(value.cvt32(), value.cvt32()); // value = 0;
                    c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
                }
            } else {
                // regs.fc0 = (value & ((u64)1 << (nsv - 1))) != 0;
                c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
                c.bt(value, nsv - 1);
                c.setc(cl); // u32 mask = (value & ((u64)1 << (nsv - 1)));
                c.shl(cx, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
                c.or_(FLAGS.cvt16(), cx);
                c.shr(value, nsv); // value >>= nsv;
                if (blk_key.curr.mod0.s == 0) {
                    SignExtend(value, 40 - nsv);
                }
            }

            if (blk_key.curr.mod0.s == 0) {
                c.and_(FLAGS, ~decltype(Flags::fv)::mask); // regs.fv = 0;
            }
        }

        SignExtend(value, 40);
        SetAccFlag(value);
        if (blk_key.curr.mod0.s == 0 && blk_key.curr.mod0.sata == 0) {

            Xbyak::Label end_label, saturate_label;
            c.bt(FLAGS, decltype(Flags::fv)::position);
            c.jc(saturate_label);
            c.movsxd(rcx, value.cvt32());
            c.cmp(rcx, value);
            c.je(end_label);
            c.L(saturate_label);
            c.or_(FLAGS, decltype(Flags::flm)::mask);
            c.mov(value, 0x7FFF'FFFF);
            c.mov(rcx, 0xFFFF'FFFF'8000'0000);
            c.cmp(original_sign, 1);
            c.cmove(value, rcx);
            c.L(end_label);
        }
        SetAcc(dest, value);
    }

    void ShiftBus40(Reg64 value, Reg16 sv, RegName dest) {
        ASSERT(sv.getIdx() == Xbyak::Operand::CL);
        c.xor_(esi, esi);
        c.mov(rbx, 0xFF'FFFF'FFFF);
        c.and_(value, rbx);
        const Reg64 original_sign = rbx;
        c.mov(original_sign, value);
        c.shr(original_sign, 39);
        Xbyak::Label end_label, right_shift;
        c.bt(sv, 15);
        c.jc(right_shift);
        // left shift
        Xbyak::Label normal_shift, end_left_shift;
        c.cmp(sv, 40);
        c.jl(normal_shift);
        if (blk_key.curr.mod0.s == 0) {
            // regs.fv = value != 0;
            c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
            c.test(value, value);
            c.setne(sil);                             // u32 mask = (value != 0) ? 1 : 0
            c.shl(si, decltype(Flags::fv)::position); // mask <<= fv_pos
            c.or_(FLAGS.cvt16(), si);                         // flags |= mask
            // if (regs.fv) {
            //    regs.fvl = 1;
            // }
            static_assert(decltype(Flags::fv)::position - decltype(Flags::fvl)::position == 3);
            // mask is either 0 or 1. If it was 0, then the or_ wont have any effect.
            // If it was 1, it will set fvl, which is what we want.
            c.shr(si, 3); // mask >>= 3;
            c.or_(FLAGS.cvt16(), si);
        }
        c.xor_(value, value);                       // value = 0;
        c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
        c.jmp(end_left_shift);
        c.L(normal_shift);
        if (blk_key.curr.mod0.s == 0) {
            // TODO: This can probably be done better
            // rdx = SignExtend(value, 40 - sv);
            c.mov(rdx, value);
            c.add(sv, 24);
            c.shl(rdx, sv.cvt8());
            c.sar(rdx, sv.cvt8());
            c.sub(sv, 24);

            // rcx = SignExtend(value, 40);
            c.mov(rsi, value);
            SignExtend(rsi, 40);

            // regs.fv = SignExtend<40>(value) != SignExtend(value, 40 - sv);
            c.and_(FLAGS, ~decltype(Flags::fv)::mask); // clear fv
            c.cmp(rsi, rdx);
            c.setne(sil); // u32 mask = (SignExtend<40>(value) != SignExtend(value, 40 - sv) ? 1 : 0
            c.and_(si, 1);
            c.shl(si, decltype(Flags::fv)::position); // mask <<= fv_pos
            c.or_(FLAGS.cvt16(), si);                  // flags |= mask
            // if (regs.fv) {
            //     regs.fvl = 1;
            // }
            c.shr(si, 3); // mask >>= 3;
            c.or_(FLAGS.cvt16(), si);
        }
        c.shl(value, sv.cvt8()); // value <<= sv;
        c.xor_(esi, esi);
        // regs.fc0 = (value & ((u64)1 << 40)) != 0;
        c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
        c.bt(value, 40);
        c.setc(sil);                               // u32 mask = (value & ((u64)1 << 40));
        c.shl(si, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
        c.or_(FLAGS.cvt16(), si);
        c.L(end_left_shift);
        c.jmp(end_label);
        c.L(right_shift);
        // right shift
        const Reg16 nsv = sv;
        c.not_(nsv);
        c.add(nsv, 1);
        Xbyak::Label normal_right_shift, end_right_shift;
        c.cmp(nsv, 40);
        c.jl(normal_right_shift);
        if (blk_key.curr.mod0.s == 0) {
            c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
            c.xor_(esi, esi);
            c.bt(value, 39);
            c.setc(sil);
            c.shl(sil, decltype(Flags::fc0)::position);
            c.or_(FLAGS.cvt8(), sil);
            // regs.fc0 = (value >> 39) & 1;
            c.shl(value, 64 - 40);
            c.sar(value, 63);
            // value = regs.fc0 ? 0xFF'FFFF'FFFF : 0;
        } else {
            c.xor_(value, value);                       // value = 0;
            c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // regs.fc0 = 0;
        }
        c.jmp(end_right_shift);
        c.L(normal_right_shift);
        // regs.fc0 = (value & ((u64)1 << (nsv - 1))) != 0;
        c.and_(FLAGS, ~decltype(Flags::fc0)::mask); // clear fc0
        c.xor_(esi, esi);
        c.sub(nsv, 1);
        c.bt(value, nsv.cvt64());
        c.setc(sil);                               // u32 mask = (value & ((u64)1 << (nsv - 1)));
        c.shl(si, decltype(Flags::fc0)::position); // mask <<= fc0_pos;
        c.or_(FLAGS.cvt16(), si);
        c.add(nsv, 1);
        c.shr(value, nsv.cvt8()); // value >>= nsv;
        if (blk_key.curr.mod0.s == 0) {
            c.add(nsv, 24);
            c.shl(value, nsv.cvt8());
            c.sar(value, nsv.cvt8());
            c.sub(nsv, 24);
        }
        c.L(end_right_shift);

        if (blk_key.curr.mod0.s == 0) {
            c.and_(FLAGS, ~decltype(Flags::fv)::mask); // regs.fv = 0;
        }
        c.L(end_label);

        SignExtend(value, 40);
        SetAccFlag(value);
        if (blk_key.curr.mod0.s == 0 && blk_key.curr.mod0.sata == 0) {
            Xbyak::Label end_label, saturate_label;
            c.bt(FLAGS, decltype(Flags::fv)::position);
            c.jc(saturate_label);
            c.movsxd(rcx, value.cvt32());
            c.cmp(rcx, value);
            c.je(end_label);
            c.L(saturate_label);
            c.or_(FLAGS, decltype(Flags::flm)::mask);
            c.mov(value, 0x7FFF'FFFF);
            c.mov(rcx, 0xFFFF'FFFF'8000'0000);
            c.cmp(original_sign, 1);
            c.cmove(value, rcx);
            c.L(end_label);
        }
        SetAcc(dest, value);
    }

    void AddSub(Reg64 a, Reg64 b, Reg64 result, bool sub) {
        c.mov(rsi, 0xFF'FFFF'FFFF);
        c.and_(a, rsi);
        c.and_(b, rsi);
        c.mov(result, a);
        if (sub) {
            c.sub(result, b);
        } else {
            c.add(result, b);
        }
        if (light_flags) {
            // Only the sticky fvl, set on overflow.
            if (sub) {
                c.not_(b);
            }
            c.xor_(b, a);
            c.not_(b);
            c.xor_(a, result);
            c.and_(a, b);
            c.xor_(esi, esi);
            c.bt(a, 39);
            c.setc(rsi.cvt8());
            c.shl(esi, decltype(Flags::fvl)::position);
            c.or_(FLAGS, esi);
            SignExtend(result, 40);
            return;
        }
        c.and_(FLAGS, ~(decltype(Flags::fc0)::mask | decltype(Flags::fv)::mask)); // clear fc0, fv
        c.xor_(esi, esi);
        c.bt(result, 40);
        c.setc(rsi.cvt8());
        c.shl(rsi, decltype(Flags::fc0)::position);
        c.or_(FLAGS.cvt16(), rsi.cvt16()); // regs.fc0 = (result >> 40) & 1;
        if (sub) {
            c.not_(b);
        }
        c.xor_(b, a);
        c.not_(b);
        c.xor_(a, result);
        c.and_(a, b);
        c.bt(a, 39); // ((~(a ^ b) & (a ^ result)) >> 39) & 1;
        c.setc(rsi.cvt8());
        c.shl(rsi, decltype(Flags::fv)::position);
        c.or_(FLAGS.cvt16(), rsi.cvt16());
        // if (regs.fv) {
        //     regs.fvl = 1;
        // }
        static_assert(decltype(Flags::fv)::position - decltype(Flags::fvl)::position == 3);
        c.shr(rsi, 3);
        c.or_(FLAGS.cvt16(), rsi.cvt16());
        SignExtend(result, 40);
    }

    void Moda(ModaOp op, RegName a, Cond cond) {
        ConditionPass(cond, [&] {
            Xbyak::Reg64 acc = rax;
            switch (op) {
            case ModaOp::Shr: {
                GetAcc(acc, a);
                ShiftBus40(acc, 0xFFFF, a);
                break;
            }
            case ModaOp::Shr4: {
                GetAcc(acc, a);
                ShiftBus40(acc, 0xFFFC, a);
                break;
            }
            case ModaOp::Shl: {
                GetAcc(acc, a);
                ShiftBus40(acc, 1, a);
                break;
            }
            case ModaOp::Shl4: {
                GetAcc(acc, a);
                ShiftBus40(acc, 4, a);
                break;
            }
            case ModaOp::Ror: {
                GetAcc(acc, a);
                c.mov(rbx, 0xFF'FFFF'FFFF); // Maybe use BZHI?
                c.and_(acc, rbx);
                c.xor_(rbx, rbx);
                c.btr(FLAGS, decltype(Flags::fc0)::position); // test and clear fc0
                c.setc(bl); // u16 old_fc = regs.fc0;
                c.shl(rbx, 40);
                c.or_(acc, rbx); // value |= (u64)old_fc << 40;
                c.bt(acc, 0); // u32 mask = value & 1;
                c.setc(bl); // mask <<= fc0_pos;
                c.shl(bl, decltype(Flags::fc0)::position);
                c.or_(FLAGS.cvt8(), bl); // flags |= mask;
                c.shr(acc, 1); // value >>= 1;
                SignExtend(acc, 40);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Rol: {
                GetAcc(acc, a);
                c.xor_(rbx, rbx);
                c.btr(FLAGS, decltype(Flags::fc0)::position); // test and clear fc0
                c.setc(bl); // u16 old_fc = regs.fc0;
                c.shl(acc, 1); // value <<= 1;
                c.or_(acc, rbx); // value |= old_fc;
                c.bt(acc, 40);
                c.setc(bl);
                c.shl(bl, decltype(Flags::fc0)::position);
                c.or_(FLAGS.cvt16(), bx); // regs.fc0 = (value >> 40) & 1;
                SignExtend(acc, 40);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Clr: {
                SatAndSetAccAndFlag(a, 0ULL);
                break;
            }
            case ModaOp::Not: {
                GetAcc(acc, a);
                c.not_(acc);
                SetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Neg: {
                GetAcc(acc, a);
                c.and_(FLAGS, ~(decltype(Flags::fc0)::mask | decltype(Flags::fv)::mask)); // clear fc0, fv
                const Reg64 scratch = rbx;
                c.test(acc, acc);
                c.setnz(scratch.cvt8());
                c.shl(scratch, decltype(Flags::fc0)::position);
                c.or_(FLAGS.cvt8(), scratch.cvt8());
                c.mov(scratch, 0xFFFF'FF80'0000'0000);
                c.cmp(acc, scratch);
                c.sete(scratch.cvt8());
                c.shl(scratch, decltype(Flags::fv)::position);
                c.or_(FLAGS.cvt8(), scratch.cvt8());
                // if (regs.fv)
                //    regs.fvl = 1;
                c.shr(scratch, decltype(Flags::fv)::position - decltype(Flags::fvl)::position);
                c.or_(FLAGS.cvt8(), scratch.cvt8());
                c.not_(acc);
                c.add(acc, 1);
                SignExtend(acc, 40);
                SatAndSetAccAndFlag(a, acc);
                break;
            }
            case ModaOp::Rnd: {
                GetAcc(acc, a);
                c.mov(rbx, 0x8000);
                AddSub(acc, rbx, rcx, false);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Pacr: {
                ProductToBus40(acc, Px{0});
                c.mov(rbx, 0x8000);
                AddSub(acc, rbx, rcx, false);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Clrr: {
                SatAndSetAccAndFlag(a, 0x8000ULL);
                break;
            }
            case ModaOp::Inc: {
                GetAcc(acc, a);
                c.mov(rbx, 1);
                AddSub(acc, rbx, rcx, false);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Dec: {
                GetAcc(acc, a);
                c.mov(rbx, 1);
                AddSub(acc, rbx, rcx, true);
                SatAndSetAccAndFlag(a, rcx);
                break;
            }
            case ModaOp::Copy: {
                // note: bX doesn't support
                GetAcc(acc, a == RegName::a0 ? RegName::a1 : RegName::a0);
                SatAndSetAccAndFlag(a, acc);
                break;
            }
            default:
                UNREACHABLE();
            }
        });
    }

    void moda4(Moda4 op, Ax a, Cond cond) {
        Moda(op.GetName(), a.GetName(), cond);
    }

    void moda3(Moda3 op, Bx a, Cond cond) {
        Moda(op.GetName(), a.GetName(), cond);
    }

    void pacr1(Ax a) {
        NOT_IMPLEMENTED();
    }

    void clr(Ab a, Ab b) {
        NOT_IMPLEMENTED();
    }
    void clrr(Ab a, Ab b) {
        NOT_IMPLEMENTED();
    }

    void BlockRepeat(Reg64 lc, u32 address) {
        using Frame = JitRegisters::BlockRepeatFrame;
        static_assert(sizeof(Frame) == 12);
        c.movzx(rbx, word[REGS + offsetof(JitRegisters, bcn)]);
        c.lea(rcx, ptr[rbx + rbx * 2]);
        c.lea(rcx, ptr[REGS + offsetof(JitRegisters, bkrep_stack) + rcx * 4]);
        c.mov(dword[rcx + offsetof(Frame, start)], regs.pc);
        c.mov(dword[rcx + offsetof(Frame, end)], address);
        c.mov(word[rcx + offsetof(Frame, lc)], lc.cvt16());
        c.add(rbx, 1);
        c.or_(rbx, 1 << 16);
        c.mov(dword[REGS + offsetof(JitRegisters, bcn)], rbx.cvt32());
        static_assert(offsetof(JitRegisters, lp) - offsetof(JitRegisters, bcn) == sizeof(u16));
        bkrep_end_locations.insert(address);
        StorePC(regs.pc);
        compiling = false;
    }

    void bkrep(Imm8 a, Address16 addr) {
        // TODO: Can probably inline a loop for this with some checks
        const Reg64 lc = rax;
        c.mov(lc, a.Unsigned16());
        u32 address = addr.Address32() | (regs.pc & 0x30000);
        BlockRepeat(lc, address);
    }
    void bkrep(Register a, Address18_16 addr_low, Address18_2 addr_high) {
        const Reg64 lc = rax;
        RegToBus16(a.GetName(), lc);
        u32 address = Address32(addr_low, addr_high);
        BlockRepeat(lc, address);
    }
    void bkrep_r6(Address18_16 addr_low, Address18_2 addr_high) {
        const Reg64 lc = rax;
        RegToBus16(RegName::r6, lc);
        u32 address = Address32(addr_low, addr_high);
        BlockRepeat(lc, address);
    }

    // The address register is passed in and out through tmp_addr.
    static void StoreBlockRepeatThunk(void* this_ptr) {
        auto* self = reinterpret_cast<EmitX64*>(this_ptr);
        JitRegisters& regs = self->regs;
        u16 address = regs.tmp_addr;
        self->mem.DataWrite(--address, regs.bkrep_stack[0].lc);
        self->mem.DataWrite(--address, regs.bkrep_stack[0].start & 0xFFFF);
        self->mem.DataWrite(--address, regs.bkrep_stack[0].end & 0xFFFF);
        u16 flag = regs.lp << 15;
        flag |= regs.bkrep_stack[0].start >> 16;
        flag |= (regs.bkrep_stack[0].end >> 16) << 8;
        self->mem.DataWrite(--address, flag);
        if (regs.lp) {
            std::copy(regs.bkrep_stack.begin() + 1, regs.bkrep_stack.begin() + regs.bcn,
                      regs.bkrep_stack.begin());
            --regs.bcn;
            if (regs.bcn == 0)
                regs.lp = 0;
        }
        regs.tmp_addr = address;
        self->SyncMemoryInterfaceUnit();
    }

    static void RestoreBlockRepeatThunk(void* this_ptr) {
        auto* self = reinterpret_cast<EmitX64*>(this_ptr);
        JitRegisters& regs = self->regs;
        u16 address = regs.tmp_addr;
        if (regs.lp) {
            ASSERT(regs.bcn <= 3);
            std::copy_backward(regs.bkrep_stack.begin(), regs.bkrep_stack.begin() + regs.bcn,
                               regs.bkrep_stack.begin() + regs.bcn + 1);
            ++regs.bcn;
        }
        const u32 flag = self->mem.DataRead(address++);
        const u16 valid = flag >> 15;
        if (regs.lp) {
            ASSERT(valid);
        } else {
            if (valid)
                regs.lp = regs.bcn = 1;
        }
        regs.bkrep_stack[0].end = self->mem.DataRead(address++) | (((flag >> 8) & 3) << 16);
        regs.bkrep_stack[0].start = self->mem.DataRead(address++) | ((flag & 3) << 16);
        regs.bkrep_stack[0].lc = self->mem.DataRead(address++);
        regs.tmp_addr = address;
        self->SyncMemoryInterfaceUnit();
        // Blocks only check loop ends that were known when they were compiled.
        self->AddBkrepEnd(regs.bkrep_stack[0].end);
    }

    void BlockRepeatStoreRestore(RegName address_reg, bool restore) {
        const Reg64 address = rax;
        RegToBus16(address_reg, address);
        c.mov(word[REGS + offsetof(JitRegisters, tmp_addr)], address.cvt16());
        EmitSlowCall(restore ? RestoreBlockRepeatThunk : StoreBlockRepeatThunk);
        c.movzx(address, word[REGS + offsetof(JitRegisters, tmp_addr)]);
        RegFromBus16(address_reg, address);
        if (restore) {
            StorePC(regs.pc);
            compiling = false;
            link_ok = false;
        }
    }

    static RegName RnName(u16 unit) {
        return static_cast<RegName>(static_cast<int>(RegName::r0) + unit);
    }

    void bkreprst(ArRn2 a) {
        instruction_cycles = Cycles::BlockRepeatStoreRestore;
        BlockRepeatStoreRestore(RnName(GetArRnUnit(a)), true);
    }
    void bkreprst_memsp() {
        instruction_cycles = Cycles::BlockRepeatStoreRestore;
        BlockRepeatStoreRestore(RegName::sp, true);
    }
    void bkrepsto(ArRn2 a) {
        instruction_cycles = Cycles::BlockRepeatStoreRestore;
        BlockRepeatStoreRestore(RnName(GetArRnUnit(a)), false);
    }
    void bkrepsto_memsp() {
        instruction_cycles = Cycles::BlockRepeatStoreRestore;
        BlockRepeatStoreRestore(RegName::sp, false);
    }

    void banke(BankFlags flags) {
        if (flags.Cfgi()) {
            c.mov(word[REGS + offsetof(JitRegisters, cfgi)], blk_key.cfgib.raw);
            c.mov(word[REGS + offsetof(JitRegisters, cfgib)], blk_key.cfgi.raw);
            std::swap(blk_key.cfgi, blk_key.cfgib);
            if (blk_key.curr.mod1.stp16) {
                c.mov(word[REGS + offsetof(JitRegisters, stepi0)], blk_key.stepi0b);
                c.mov(word[REGS + offsetof(JitRegisters, stepi0b)], blk_key.stepi0);
                std::swap(blk_key.stepi0, blk_key.stepi0b);
            }
        }
        if (flags.R4()) {
            c.xchg(word[REGS + offsetof(JitRegisters, r4b)], R4_5_6_7.cvt16());
        }
        if (flags.R1()) {
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            c.xchg(word[REGS + offsetof(JitRegisters, r1b)], R0_1_2_3.cvt16());
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
        }
        if (flags.R0()) {
            c.xchg(word[REGS + offsetof(JitRegisters, r0b)], R0_1_2_3.cvt16());
        }
        if (flags.R7()) {
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            c.xchg(word[REGS + offsetof(JitRegisters, r7b)], R4_5_6_7.cvt16());
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
        }
        if (flags.Cfgj()) {
            c.mov(word[REGS + offsetof(JitRegisters, cfgj)], blk_key.cfgjb.raw);
            c.mov(word[REGS + offsetof(JitRegisters, cfgjb)], blk_key.cfgj.raw);
            std::swap(blk_key.cfgj, blk_key.cfgjb);
            if (blk_key.curr.mod1.stp16) {
                c.mov(word[REGS + offsetof(JitRegisters, stepj0)], blk_key.stepj0b);
                c.mov(word[REGS + offsetof(JitRegisters, stepj0b)], blk_key.stepj0);
                std::swap(blk_key.stepj0, blk_key.stepj0b);
            }
        }
    }
    void bankr() {
        NOT_IMPLEMENTED();
    }
    void bankr(Ar a) {
        NOT_IMPLEMENTED();
    }
    void bankr(Ar a, Arp b) {
        NOT_IMPLEMENTED();
    }
    void bankr(Arp a) {
        NOT_IMPLEMENTED();
    }

    void bitrev(Rn a) {
        NOT_IMPLEMENTED();
    }
    void bitrev_dbrv(Rn a) {
        NOT_IMPLEMENTED();
    }
    void bitrev_ebrv(Rn a) {
        NOT_IMPLEMENTED();
    }

    void br(Address18_16 addr_low, Address18_2 addr_high, Cond cond) {
        const u32 taken_cycles = ControlFlowCycles(cond, 2, Cycles::BranchTaken);
        StorePC(regs.pc);
        ConditionPass(cond, [&] {
            regs.pc = Address32(addr_low, addr_high);
            StorePC(regs.pc);
            AddCycles(taken_cycles);
        });
        // For static jump we can continue compiling.
        compiling = cond.GetName() == CondValue::True;
    }

    void brr(RelAddr7 addr, Cond cond) {
        const u32 taken_cycles = ControlFlowCycles(cond, 1, Cycles::BranchTaken);
        const u32 next_pc = regs.pc;
        // A loop back can replace the jump (see the interpreter's brr).
        const bool loop_end_next = bkrep_end_locations.contains(next_pc);
        StorePC(regs.pc);
        ConditionPass(cond, [&] {
            // note: pc is the address of the NEXT instruction
            regs.pc += addr.Relative32();
            StorePC(regs.pc);
            if (loop_end_next) {
                EmitLoopBackForBranch(next_pc);
            }
            AddCycles(taken_cycles);
            if (addr.Relative32() == 0xFFFFFFFF) {
                c.mov(dword[REGS + offsetof(JitRegisters, idle)], true);
                compiling = false; // Always end compilation for idle loops.
                link_ok = false;
            } else {
                // For static jump we can continue compiling.
                compiling = cond.GetName() == CondValue::True && !loop_end_next;
            }
        });
    }

    // Goes back to the start of the bkrep block instead of the brr target when the last block
    // instruction is the word after the brr.
    void EmitLoopBackForBranch(u32 next_pc) {
        using Frame = JitRegisters::BlockRepeatFrame;
        static_assert(sizeof(Frame) == 12);
        Xbyak::Label skip;
        const Reg64 frame = rax;
        c.test(word[REGS + offsetof(JitRegisters, lp)], 0x1);
        c.jz(skip);
        c.movzx(frame, word[REGS + offsetof(JitRegisters, bcn)]);
        c.lea(frame, ptr[frame + frame * 2]);
        c.lea(frame, ptr[REGS + offsetof(JitRegisters, bkrep_stack) - sizeof(Frame) + frame * 4]);
        c.cmp(dword[frame + offsetof(Frame, end)], next_pc);
        c.jne(skip);
        c.cmp(word[frame + offsetof(Frame, lc)], 0);
        c.je(skip);
        c.sub(word[frame + offsetof(Frame, lc)], 1);
        c.mov(ecx, dword[frame + offsetof(Frame, start)]);
        StorePC(ecx);
        c.L(skip);
    }

    void break_() {
        c.sub(word[REGS + offsetof(JitRegisters, bcn)], 1);
        c.setnz(al);
        c.movzx(eax, al);
        c.mov(word[REGS + offsetof(JitRegisters, lp)], ax);
    }

    void call(Address18_16 addr_low, Address18_2 addr_high, Cond cond) {
        const u32 taken_cycles = ControlFlowCycles(cond, 2, Cycles::BranchTaken);
        const u32 ret_pc = regs.pc;
        StorePC(ret_pc);
        ConditionPass(cond, [&] {
            EmitPushPC();
            regs.pc = Address32(addr_low, addr_high);
            StorePC(regs.pc);
            AddCycles(taken_cycles);
        });
        // For static jump we can continue compiling.
        compiling = cond.GetName() == CondValue::True;
        if (compiling) {
            call_stack.push(ret_pc);
        }
    }
    void calla(Axl a) {
        NOT_IMPLEMENTED();
    }
    void calla(Ax a) {
        instruction_cycles = Cycles::IndirectBranch;
        EmitPushPC();
        const Reg64 pc = rax;
        GetAcc(pc, a.GetName());
        c.and_(pc, 0x3FFFF);
        StorePC(pc.cvt32());
        compiling = false;
    }
    void callr(RelAddr7 addr, Cond cond) {
        const u32 taken_cycles = ControlFlowCycles(cond, 1, Cycles::BranchTaken);
        const u32 ret_pc = regs.pc;
        StorePC(ret_pc);
        ConditionPass(cond, [&] {
            EmitPushPC();
            regs.pc += addr.Relative32();
            StorePC(regs.pc);
            AddCycles(taken_cycles);
        });
        // For static jump we can continue compiling.
        compiling = cond.GetName() == CondValue::True;
        if (compiling) {
            call_stack.push(ret_pc);
        }
    }

    void cntx_s() {
        ContextStore();
        EndBlockForInterrupts();
    }
    void cntx_r() {
        ContextRestore();
        EndBlockForInterrupts();
    }

    // The interpreter checks for interrupts after every instruction. Ending the block after an
    // instruction that can enable one (ie, im, imv) makes the dispatcher check there too. Otherwise
    // an interrupt window that the block closes again would never be taken.
    void EndBlockForInterrupts() {
        StorePC(regs.pc);
        compiling = false;
        link_ok = false;
    }

    void ContextStore() {
        regs.ShadowStore(c);
        regs.ShadowSwap(c);
        std::swap(blk_key.curr, blk_key.shadow);
        // if (!regs.crep) {
        //     regs.repcs = regs.repc;
        // }
        c.mov(ax, word[REGS + offsetof(JitRegisters, repcs)]);
        c.test(word[REGS + offsetof(JitRegisters, crep)], 0x1);
        c.cmovz(ax, word[REGS + offsetof(JitRegisters, repc)]);
        c.mov(word[REGS + offsetof(JitRegisters, repcs)], ax);

        Xbyak::Label ccnta_label, end_label;
        c.test(word[REGS + offsetof(JitRegisters, ccnta)], 0x1);
        c.jnz(ccnta_label);
        c.mov(qword[REGS + offsetof(JitRegisters, a1s)], A[1]);
        c.mov(qword[REGS + offsetof(JitRegisters, b1s)], B[1]);
        c.jmp(end_label);
        c.L(ccnta_label);
        c.mov(rax, B[1]);
        c.mov(B[1], A[1]);
        SetAccAndFlag(RegName::a1, rax); // Flag set on b1->a1
        c.L(end_label);
    }
    void ContextRestore() {
        regs.ShadowRestore(c);
        regs.ShadowSwap(c);
        std::swap(blk_key.curr, blk_key.shadow);

        // if (!regs.crep) {
        //     regs.repc = regs.repcs;
        // }
        c.mov(ax, word[REGS + offsetof(JitRegisters, repc)]);
        c.test(word[REGS + offsetof(JitRegisters, crep)], 0x1);
        c.cmovz(ax, word[REGS + offsetof(JitRegisters, repcs)]);
        c.mov(word[REGS + offsetof(JitRegisters, repc)], ax);

        Xbyak::Label ccnta_label, end_label;
        c.test(word[REGS + offsetof(JitRegisters, ccnta)], 0x1);
        c.jnz(ccnta_label);
        c.mov(A[1], qword[REGS + offsetof(JitRegisters, a1s)]);
        c.mov(B[1], qword[REGS + offsetof(JitRegisters, b1s)]);
        c.jmp(end_label);
        c.L(ccnta_label);
        c.xchg(A[1], B[1]);
        c.L(end_label);
    }

    void ret(Cond cond) {
        const u32 taken_cycles = ControlFlowCycles(cond, 1, Cycles::ReturnTaken);
        StorePC(regs.pc);
        ConditionPass(cond, [&] {
            EmitPopPC();
            AddCycles(taken_cycles);
        });
        // If the last call instruction had a static target and this instruction
        // always returns, we don't have to stop compiling.
        if (cond.GetName() == CondValue::True && !call_stack.empty()) {
            regs.pc = call_stack.top();
            call_stack.pop();
            return;
        }
        compiling = false;
    }
    void retd() {
        NOT_IMPLEMENTED();
    }
    void reti(Cond cond) {
        const u32 taken_cycles = ControlFlowCycles(cond, 1, Cycles::ReturnTaken);
        StorePC(regs.pc);
        ConditionPass(cond, [&]{
            EmitPopPC();
            c.mov(word[REGS + offsetof(JitRegisters, ie)], 1);
            AddCycles(taken_cycles);
        });
        compiling = false;
        link_ok = false;
    }
    void retic(Cond cond) {
        const u32 taken_cycles = ControlFlowCycles(cond, 1, Cycles::ReturnTaken);
        StorePC(regs.pc);
        ConditionPass(cond, [&] {
            EmitPopPC();
            c.mov(word[REGS + offsetof(JitRegisters, ie)], 1);
            ContextRestore();
            AddCycles(taken_cycles);
        });
        compiling = false;
        link_ok = false;
    }
    void retid() {
        NOT_IMPLEMENTED();
    }
    void retidc() {
        NOT_IMPLEMENTED();
    }
    void rets(Imm8 a) {
        instruction_cycles = Cycles::ReturnTaken;
        EmitPopPC();
        c.add(word[REGS + offsetof(JitRegisters, sp)], a.Unsigned16());
        if (!call_stack.empty()) {
            regs.pc = call_stack.top();
            call_stack.pop();
            return;
        }
        compiling = false;
    }

    void load_ps(Imm2 a) {
        const u16 mask = (a.Unsigned16() & 3) << decltype(Mod0::ps0)::position;
        c.mov(ax, word[REGS + offsetof(JitRegisters, mod0)]);
        c.and_(ax, ~decltype(Mod0::ps0)::mask);
        c.or_(ax, mask);
        c.mov(word[REGS + offsetof(JitRegisters, mod0)], ax);
        blk_key.curr.mod0.ps0.Assign(a.Unsigned16());
    }
    void load_stepi(Imm7s a) {
        // Although this is signed, we still only store the lower 7 bits
        const u8 stepi = static_cast<u8>(a.Signed16() & 0x7F);
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgi)]);
        c.and_(ax, ~decltype(Cfg::step)::mask);
        c.or_(ax, stepi);
        c.mov(word[REGS + offsetof(JitRegisters, cfgi)], ax);
        blk_key.cfgi.step.Assign(stepi);
    }
    void load_stepj(Imm7s a) {
        const u8 stepj = static_cast<u8>(a.Signed16() & 0x7F);
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgj)]);
        c.and_(ax, ~decltype(Cfg::step)::mask);
        c.or_(ax, stepj);
        c.mov(word[REGS + offsetof(JitRegisters, cfgj)], ax);
        blk_key.cfgj.step.Assign(stepj);
    }
    void load_page(Imm8 a) {
        const u8 page = static_cast<u8>(a.Unsigned16());
        c.mov(byte[REGS + offsetof(JitRegisters, mod1)], page);
        blk_key.curr.mod1.page.Assign(page);
    }
    void load_modi(Imm9 a) {
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgi)]);
        c.and_(rax, ~decltype(Cfg::mod)::mask);
        c.or_(rax, a.Unsigned16() << decltype(Cfg::mod)::position);
        c.mov(word[REGS + offsetof(JitRegisters, cfgi)], ax);
        blk_key.cfgi.mod.Assign(a.Unsigned16());
    }
    void load_modj(Imm9 a) {
        c.mov(ax, word[REGS + offsetof(JitRegisters, cfgj)]);
        c.and_(rax, ~decltype(Cfg::mod)::mask);
        c.or_(rax, a.Unsigned16() << decltype(Cfg::mod)::position);
        c.mov(word[REGS + offsetof(JitRegisters, cfgj)], ax);
        blk_key.cfgj.mod.Assign(a.Unsigned16());
    }
    void load_movpd(Imm2 a) {
        c.mov(word[REGS + offsetof(JitRegisters, pcmhi)], a.Unsigned16());
    }
    void load_ps01(Imm4 a) {
        const u16 ps0 = a.Unsigned16() & 3;
        const u16 ps1 = a.Unsigned16() >> 2;
        const u16 mask = (ps0 << decltype(Mod0::ps0)::position) | (ps1 << decltype(Mod0::ps1)::position);
        c.mov(ax, word[REGS + offsetof(JitRegisters, mod0)]);
        c.and_(ax, ~(decltype(Mod0::ps0)::mask | decltype(Mod0::ps1)::mask));
        c.or_(ax, mask);
        c.mov(word[REGS + offsetof(JitRegisters, mod0)], ax);
        blk_key.curr.mod0.ps0.Assign(ps0);
        blk_key.curr.mod0.ps1.Assign(ps1);
    }

    void push(Imm16 a) {
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, a.Unsigned16());
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push(Register a) {
        const Reg64 value = rbx;
        RegToBus16(a.GetName(), value, true);
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push(Abe a) {
        const Reg64 value = rbx;
        GetAndSatAcc(value, a.GetName());
        c.shr(value, 32);
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    std::optional<u16> IsArpArpSttModConst(ArArpSttMod a) {
        switch (a.GetName()) {
        case RegName::ar0:
            return blk_key.curr.ar[0].raw;
        case RegName::ar1:
            return blk_key.curr.ar[1].raw;
        case RegName::arp0:
            return blk_key.curr.arp[0].raw;
        case RegName::arp1:
            return blk_key.curr.arp[1].raw;
        case RegName::arp2:
            return blk_key.curr.arp[2].raw;
        case RegName::arp3:
            return blk_key.curr.arp[3].raw;
        case RegName::mod0:
            return blk_key.curr.mod0.raw;
        case RegName::mod1:
            return blk_key.curr.mod1.raw;
        case RegName::mod2:
            return blk_key.curr.mod2.raw;
        default:
            return std::nullopt;
        }
    }

    void push(ArArpSttMod a) {
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        if (auto value = IsArpArpSttModConst(a); value.has_value()) {
            StoreToMemory(sp, value.value());
            c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
            return;
        }
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_prpage() {
        NOT_IMPLEMENTED();
    }
    void push(Px a) {
        const Reg64 value = rbx;
        ProductToBus40(value, a);
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.sub(sp, 1);
        c.shr(value, 16);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_r6() {
        const Reg64 value = rax;
        RegToBus16(RegName::r6, value);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_repc() {
        const Reg64 value = rax;
        c.mov(value, word[REGS + offsetof(JitRegisters, repc)]);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_x0() {
        const Reg64 value = rax;
        c.rorx(value, FACTORS, 32);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_x1() {
        const Reg64 value = rax;
        c.rorx(value, FACTORS, 48);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void push_y1() {
        const Reg64 value = rax;
        c.rorx(value, FACTORS, 16);
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void pusha(Ax a) {
        const Reg64 value = rbx;
        GetAndSatAcc(value, a.GetName());
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.sub(sp, 1);
        c.shr(value, 16);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }
    void pusha(Bx a) {
        const Reg64 value = rbx;
        GetAndSatAcc(value, a.GetName());
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        c.sub(sp, 1);
        StoreToMemory(sp, value);
        c.sub(sp, 1);
        c.shr(value, 16);
        StoreToMemory(sp, value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    void pop(Register a) {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        RegFromBus16(a.GetName(), value);
    }
    void pop(Abe a) {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        const Reg32 tmp = eax; // Register used just for truncating the accumulator register to 32 bits
        LoadFromMemory(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.movsx(value.cvt32(), value.cvt8());
        c.shl(value, 32);
        // Zero-extend bottom 32 bits of accumulator to tmp.cvt64() and merge it into value
        c.mov(tmp, GetAccDirect(a.GetName()).cvt32());
        c.or_(value, tmp.cvt64());
        SetAccAndFlag(a.GetName(), value);
    }
    void pop(ArArpSttMod a) {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        RegFromBus16(a.GetName(), value);
    }
    void pop(Bx a) {
        NOT_IMPLEMENTED();
    }
    void pop_prpage() {
        NOT_IMPLEMENTED();
    }
    void pop(Px a) {
        const Reg64 sp = rbx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(sp, 1);
        c.shl(value, 16);
        LoadFromMemory(value, sp);
        c.add(sp, 1);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
        ProductFromBus32(a, value.cvt32());
    }
    void pop_r6() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        RegFromBus16(RegName::r6, value);
    }
    void pop_repc() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.mov(word[REGS + offsetof(JitRegisters, repc)], value.cvt16());
    }
    void pop_x0() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.rorx(FACTORS, FACTORS, 32);
        c.mov(FACTORS.cvt16(), value.cvt16());
        c.rorx(FACTORS, FACTORS, 32);
    }
    void pop_x1() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.rorx(FACTORS, FACTORS, 48);
        c.mov(FACTORS.cvt16(), value.cvt16());
        c.rorx(FACTORS, FACTORS, 16);
    }
    void pop_y1() {
        const Reg64 sp = rbx;
        c.movzx(sp, word[REGS + offsetof(JitRegisters, sp)]);
        const Reg64 value = rcx;
        LoadFromMemory(value, sp);
        c.add(word[REGS + offsetof(JitRegisters, sp)], 1);
        c.rorx(FACTORS, FACTORS, 16);
        c.mov(FACTORS.cvt16(), value.cvt16());
        c.rorx(FACTORS, FACTORS, 48);
    }
    void popa(Ab a) {
        const Reg64 value = rbx;
        const Reg64 sp = rcx;
        c.mov(sp, word[REGS + offsetof(JitRegisters, sp)]);
        LoadFromMemory(value, sp);
        c.add(sp, 1);
        c.shl(value, 16);
        LoadFromMemory(value, sp);
        c.add(sp, 1);
        SignExtend(value, 32);
        SetAccAndFlag(a.GetName(), value);
        c.mov(word[REGS + offsetof(JitRegisters, sp)], sp.cvt16());
    }

    void rep(Imm8 a) {
        u16 opcode = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
        auto& decoder = decoders[opcode];
        u16 expand_value = 0;
        if (decoder.NeedExpansion()) {
            expand_value = mem.ProgramRead((regs.pc++) | (regs.prpage << 18));
        }

        for (int i = 0; i <= a.Unsigned16(); i++) {
            instruction_cycles = 0;
            decoder.call(*this, opcode, expand_value);
            current_blk->cycles += InstructionCycles(decoder);
            ASSERT(compiling); // Ensure the instruction doesn't break the block
        }
        instruction_cycles = 0;
    }
    void rep(Register a) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        c.mov(word[REGS + offsetof(JitRegisters, rep)], true);
        c.mov(word[REGS + offsetof(JitRegisters, repc)], value.cvt16());
        StorePC(regs.pc);
        compiling = false;
        rep_end_locations.insert(regs.pc);
    }
    void rep_r6() {
        NOT_IMPLEMENTED();
    }

    void shfc(Ab a, Ab b, Cond cond) {
        ConditionPass(cond, [&] {
            const Reg64 value = rax;
            GetAcc(value, a.GetName());
            const Reg16 sv = cx;
            c.mov(sv, word[REGS + offsetof(JitRegisters, sv)]);
            ShiftBus40(value, sv, b.GetName());
        });
    }
    void shfi(Ab a, Ab b, Imm6s s) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        u16 sv = s.Signed16();
        ShiftBus40(value, sv, b.GetName());
    }

    void tst4b(ArRn2 b, ArStep2 bs) {
        NOT_IMPLEMENTED();
    }
    void tst4b(ArRn2 b, ArStep2 bs, Ax c) {
        NOT_IMPLEMENTED();
    }
    void tstb(MemImm8 a, Imm4 b) {
        const Reg64 value = rbx;
        const Reg64 address = rax;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(value, address);
        c.xor_(eax, eax);
        c.and_(FLAGS, ~decltype(Flags::fz)::mask);
        c.bt(value, b.Unsigned16());
        c.setc(ah);
        c.or_(FLAGS, eax);
    }
    void tstb(Rn a, StepZIDS as, Imm4 b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        LoadFromMemory(value, address);
        c.xor_(eax, eax);
        c.and_(FLAGS, ~decltype(Flags::fz)::mask);
        c.bt(value, b.Unsigned16());
        c.setc(ah);
        c.or_(FLAGS, eax);
    }
    void tstb(Register a, Imm4 b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        const Reg64 mask = rbx;
        c.xor_(mask.cvt32(), mask.cvt32());
        c.and_(FLAGS, ~decltype(Flags::fz)::mask); // clear fz
        c.bt(value, b.Unsigned16());
        c.setc(mask.cvt8());
        c.shl(mask, decltype(Flags::fz)::position);
        c.or_(FLAGS, mask.cvt32());
    }
    void tstb_r6(Imm4 b) {
        NOT_IMPLEMENTED();
    }
    void tstb(SttMod a, Imm16 b) {
        NOT_IMPLEMENTED();
    }

    void and_(Ab a, Ab b, Ax c) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        const Reg64 value_b = rbx;
        GetAcc(value_b, b.GetName());
        this->c.and_(value, value_b);
        SetAccAndFlag(c.GetName(), value);
    }

    void dint() {
        c.mov(word[REGS + offsetof(JitRegisters, ie)], 0);
    }
    void eint() {
        c.mov(word[REGS + offsetof(JitRegisters, ie)], 1);
        EndBlockForInterrupts();
    }

    void MulGeneric(MulOp op, Ax a) {
        if (op != MulOp::Mpy && op != MulOp::Mpysu) {
            const Reg64 value = rax;
            GetAcc(value, a.GetName());
            const Reg64 product = rbx;
            ProductToBus40(product, Px{0});
            if (op == MulOp::Maa || op == MulOp::Maasu) {
                c.shr(product, 16);
                SignExtend(product, 24);
            }
            const Reg64 result = rcx;
            AddSub(value, product, result, false);
            SatAndSetAccAndFlag(a.GetName(), result);
        }

        switch (op) {
        case MulOp::Mpy:
        case MulOp::Mac:
        case MulOp::Maa:
            DoMultiplication(0, eax, ebx, true, true);
            break;
        case MulOp::Mpysu:
        case MulOp::Macsu:
        case MulOp::Maasu:
            // Note: the naming conventin of "mpysu" is "multiply signed *y* by unsigned *x*"
            DoMultiplication(0, eax, ebx, false, true);
            break;
        case MulOp::Macus:
            DoMultiplication(0, eax, ebx, true, false);
            break;
        case MulOp::Macuu:
            DoMultiplication(0, eax, ebx, false, false);
            break;
        }
    }

    void mul(Mul3 op, Rn y, StepZIDS ys, Imm16 x, Ax a) {
        NOT_IMPLEMENTED();
    }
    void mul_y0(Mul3 op, Rn x, StepZIDS xs, Ax a) {
        const Reg64 address = rax;
        RnAddressAndModify(x.Index(), xs.GetName(), address);
        const Reg64 value = rbx;
        LoadFromMemory(value, address);
        c.rorx(FACTORS, FACTORS, 32);
        c.mov(FACTORS.cvt16(), value.cvt16());
        c.rorx(FACTORS, FACTORS, 32);
        MulGeneric(op.GetName(), a);
    }
    void mul_y0(Mul3 op, Register x, Ax a) {
        const Reg64 x0 = rax;
        RegToBus16(x.GetName(), x0);
        c.rorx(FACTORS, FACTORS, 32);
        c.mov(FACTORS.cvt16(), x0.cvt16());
        c.rorx(FACTORS, FACTORS, 32);
        MulGeneric(op.GetName(), a);
    }
    void mul(Mul3 op, R45 y, StepZIDS ys, R0123 x, StepZIDS xs, Ax a) {
        const Reg64 address_y = rax;
        const Reg64 address_x = rbx;
        RnAddressAndModify(y.Index(), ys.GetName(), address_y);
        RnAddressAndModify(x.Index(), xs.GetName(), address_x);
        LoadFromMemory(FACTORS, address_y);
        c.rorx(FACTORS, FACTORS, 32);
        LoadFromMemory(FACTORS, address_x);
        c.rorx(FACTORS, FACTORS, 32);
        MulGeneric(op.GetName(), a);
    }
    void mul_y0_r6(Mul3 op, Ax a) {
        NOT_IMPLEMENTED();
    }
    void mul_y0(Mul2 op, MemImm8 x, Ax a) {
        const Reg64 x0 = rax;
        const Reg64 address = rbx;
        c.mov(address, x.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(x0, address);
        c.rorx(FACTORS, FACTORS, 32);
        c.mov(FACTORS.cvt16(), x0.cvt16());
        c.rorx(FACTORS, FACTORS, 32);
        MulGeneric(op.GetName(), a);
    }

    void mpyi(Imm8s x) {
        c.ror(FACTORS, 32);
        c.mov(FACTORS.cvt16(), x.Signed16());
        c.rol(FACTORS, 32);
        DoMultiplication(0, eax, ebx, true, true);
    }

    void msu(R45 y, StepZIDS ys, R0123 x, StepZIDS xs, Ax a) {
        NOT_IMPLEMENTED();
    }
    void msu(Rn y, StepZIDS ys, Imm16 x, Ax a) {
        NOT_IMPLEMENTED();
    }
    void msusu(ArRn2 x, ArStep2 xs, Ax a) {
        NOT_IMPLEMENTED();
    }
    void mac_x1to0(Ax a) {
        NOT_IMPLEMENTED();
    }
    void mac1(ArpRn1 xy, ArpStep1 xis, ArpStep1 yjs, Ax a) {
        NOT_IMPLEMENTED();
    }

    void modr(Rn a, StepZIDS as) {
        const u32 unit = a.Index();
        const Reg64 reg = rax;
        RnAndModifyNoPreserve(unit, as.GetName(), reg);
        c.and_(FLAGS, ~decltype(Flags::fr)::mask);
        c.test(reg, reg);
        c.setz(reg.cvt8());
        static_assert(decltype(Flags::fr)::position == 0);
        c.or_(FLAGS.cvt8(), reg.cvt8());
    }
    void modr_dmod(Rn a, StepZIDS as) {
        NOT_IMPLEMENTED();
    }
    void modr_i2(Rn a) {
        u32 unit = a.Index();
        const Reg64 reg = rax;
        RnAndModifyNoPreserve(unit, StepValue::Increase2Mode1, reg);
        c.and_(FLAGS, ~decltype(Flags::fr)::mask);
        c.test(reg, reg);
        c.setz(reg.cvt8());
        static_assert(decltype(Flags::fr)::position == 0);
        c.or_(FLAGS.cvt8(), reg.cvt8());
    }
    void modr_i2_dmod(Rn a) {
        NOT_IMPLEMENTED();
    }
    void modr_d2(Rn a) {
        u32 unit = a.Index();
        const Reg64 reg = rax;
        RnAndModifyNoPreserve(unit, StepValue::Decrease2Mode1, reg);
        c.and_(FLAGS, ~decltype(Flags::fr)::mask);
        c.test(reg, reg);
        c.setz(reg.cvt8());
        static_assert(decltype(Flags::fr)::position == 0);
        c.or_(FLAGS.cvt8(), reg.cvt8());
    }
    void modr_d2_dmod(Rn a) {
        NOT_IMPLEMENTED();
    }
    void modr_eemod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        u32 uniti, unitj;
        StepValue stepi, stepj;
        std::tie(uniti, unitj) = GetArpRnUnit(a);
        std::tie(stepi, stepj) = GetArpStep(asi, asj);
        RnAndModifyNoPreserve(uniti, stepi, rax);
        RnAndModifyNoPreserve(unitj, stepj, rax);
    }
    void modr_edmod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        NOT_IMPLEMENTED();
    }
    void modr_demod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        NOT_IMPLEMENTED();
    }
    void modr_ddmod(ArpRn2 a, ArpStep2 asi, ArpStep2 asj) {
        NOT_IMPLEMENTED();
    }

    void ProgramRead(Reg64 out, Reg64 address) {
        c.mov(out, reinterpret_cast<uintptr_t>(mem.GetSharedMemory().raw));
        c.movzx(out, word[out + address * 2]);
    }

    void movd(R0123 a, StepZIDS as, R45 b, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void movp(Axl a, Register b) {
        NOT_IMPLEMENTED();
    }
    void movp(Ax a, Register b) {
        instruction_cycles = Cycles::ProgramReadToRegister;
        const Reg64 address = rax;
        GetAcc(address, a.GetName());
        c.and_(address, 0x3FFFF);
        const Reg64 value = rbx;
        ProgramRead(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void movp(Rn a, StepZIDS as, R0123 b, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void movpdw(Ax a) {
        instruction_cycles = Cycles::ProgramReadJump;
        const Reg64 address = rbx;
        GetAcc(address, a.GetName());
        c.and_(address, 0x3FFFF);
        const Reg64 pc = rax;
        const Reg64 low = rcx;
        // the endianess doesn't seem to be affected by regs.cpc
        ProgramRead(pc, address);
        c.add(address, 1);
        ProgramRead(low, address);
        c.shl(pc, 16);
        c.or_(pc, low);
        StorePC(pc.cvt32());
        compiling = false;
    }

    void mov(Ab a, Ab b) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        SatAndSetAccAndFlag(b.GetName(), value);
    }
    void mov_dvm(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov_x0(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov_x1(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov_y1(Abl a) {
        const Reg64 value16 = rax;
        RegToBus16(a.GetName(), value16, true);
        c.rorx(FACTORS, FACTORS, 16);
        c.mov(FACTORS.cvt16(), value16.cvt16());
        c.rorx(FACTORS, FACTORS, 48);
    }

    void StoreToMemory(MemImm8 addr, Reg64 value) {
        StoreToMemory(addr.Unsigned16() + (blk_key.curr.mod1.page << 8), value);
    }

    template <typename T1, typename T2>
    void StoreToMemory(T1 addr, T2 value) {
        FarCode& mmio = NewFarCode([this, addr, value, cycles = current_blk->cycles] {
            EmitStoreTmpAddr(addr);
            c.mov(dword[REGS + offsetof(JitRegisters, tmp_cycles)], cycles);
            if constexpr (std::is_integral_v<T2>) {
                c.mov(word[REGS + offsetof(JitRegisters, tmp_value)],
                      static_cast<u32>(value & 0xFFFF));
            } else {
                c.mov(word[REGS + offsetof(JitRegisters, tmp_value)], value.cvt16());
            }
            EmitSlowCall(WriteThunk);
        });
        EmitMmioCheck(addr, mmio.entry);
        const Xbyak::Address pointer = EmitDataPointer(addr);
        if constexpr (std::is_integral_v<T2>) {
            c.mov(pointer, static_cast<u32>(value & 0xFFFF));
        } else {
            c.mov(pointer, value.cvt16());
        }
        c.L(mmio.ret);
    }

    void mov(Ablh a, MemImm8 b) {
        const Reg64 value16 = rbx;
        RegToBus16(a.GetName(), value16, true);
        StoreToMemory(b, value16);
    }
    void mov(Axl a, MemImm16 b) {
        const Reg64 value16 = rbx;
        RegToBus16(a.GetName(), value16, true);
        StoreToMemory(b.Unsigned16(), value16);
    }
    void mov(Axl a, MemR7Imm16 b) {
        const Reg64 value16 = rax;
        RegToBus16(a.GetName(), value16, true);
        const Reg64 address = rbx;
        RegToBus16(RegName::r7, address);
        c.add(address, b.Unsigned16());
        StoreToMemory(address, value16);
    }
    void mov(Axl a, MemR7Imm7s b) {
        const Reg64 value16 = rbx;
        RegToBus16(a.GetName(), value16, true);
        const Reg64 address = rcx;
        RegToBus16(RegName::r7, address);
        c.add(address, b.Signed16());
        StoreToMemory(address, value16);
    }

    void mov(MemImm16 a, Ax b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16());
        LoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemImm8 a, Ab b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemImm8 a, Ablh b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov_eu(MemImm8 a, Axh b) {
        NOT_IMPLEMENTED();
    }
    void mov(MemImm8 a, RnOld b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov_sv(MemImm8 a) {
        const Reg64 value = rbx;
        const Reg64 address = rax;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(value, address);
        c.mov(word[REGS + offsetof(JitRegisters, sv)], value.cvt16());
    }
    void mov_dvm_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov_icr_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov(Imm16 a, Bx b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov(Imm16 a, Register b) {
        const u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_icr(Imm5 a) {
        NOT_IMPLEMENTED();
    }
    void mov(Imm8s a, Axh b) {
        NOT_IMPLEMENTED();
    }
    void mov(Imm8s a, RnOld b) {
        u16 value = a.Signed16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_sv(Imm8s a) {
        const u16 value = a.Signed16();
        c.mov(word[REGS + offsetof(JitRegisters, sv)], value);
    }
    void mov(Imm8 a, Axl b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemR7Imm16 a, Ax b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        RegToBus16(RegName::r7, address);
        c.movzx(address, address.cvt16());
        c.add(address, a.Unsigned16());
        LoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(MemR7Imm7s a, Ax b) {
        const Reg64 address = rax;
        RegToBus16(RegName::r7, address);
        c.movzx(address, address.cvt16());
        c.add(address.cvt32(), ::SignExtend<16, u32>(a.Signed16()));
        const Reg64 value = rbx;
        LoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Rn a, StepZIDS as, Bx b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        LoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Rn a, StepZIDS as, Register b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        LoadFromMemory(value, address);
        RegFromBus16(b.GetName(), value);
    }
    void mov_memsp_to(Register b) {
        NOT_IMPLEMENTED();
    }
    void mov_mixp_to(Register b) {
        const Reg64 value = rax;
        c.mov(value, word[REGS + offsetof(JitRegisters, mixp)]);
        RegFromBus16(b.GetName(), value);
    }
    void mov(RnOld a, MemImm8 b) {
        const Reg64 value = rbx;
        RegToBus16(a.GetName(), value);
        StoreToMemory(b.Unsigned16() + (blk_key.curr.mod1.page << 8), value);
    }
    void mov_icr(Register a) {
        NOT_IMPLEMENTED();
    }
    void mov_mixp(Register a) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        c.mov(word[REGS + offsetof(JitRegisters, mixp)], value.cvt16());
    }
    void mov(Register a, Rn b, StepZIDS bs) {
        // a = a0 or a1 is overrided
        const Reg64 value = rbx;
        RegToBus16(a.GetName(), value, true);
        const Reg64 address = rcx;
        RnAddressAndModify(b.Index(), bs.GetName(), address);
        StoreToMemory(address, value);
    }
    void mov(Register a, Bx b) {
        const Reg64 value = rax;
        if (a.GetName() == RegName::p) {
            ProductToBus40(value, Px{0});
            SatAndSetAccAndFlag(b.GetName(), value);
        } else if (a.GetName() == RegName::a0 || a.GetName() == RegName::a1) {
            // Is there any difference from the mov(Ab, Ab) instruction?
            GetAcc(value, a.GetName());
            SatAndSetAccAndFlag(b.GetName(), value);
        } else {
            RegToBus16(a.GetName(), value, true);
            RegFromBus16(b.GetName(), value);
        }
    }
    void mov(Register a, Register b) {
        // a = a0 or a1 is overrided
        if (a.GetName() == RegName::p) {
            // b loses its typical meaning in this case
            const RegName b_name = b.GetNameForMovFromP();
            const Reg64 value = rax;
            ProductToBus40(value, Px{0});
            SatAndSetAccAndFlag(b_name, value);
        } else if (a.GetName() == RegName::pc) {
            if (b.GetName() == RegName::a0 || b.GetName() == RegName::a1) {
                SatAndSetAccAndFlag(b.GetName(), static_cast<u64>(regs.pc));
            } else {
                RegFromBus16(b.GetName(), regs.pc & 0xFFFF);
            }
        } else {
            const Reg64 value = rax;
            RegToBus16(a.GetName(), value, true);
            RegFromBus16(b.GetName(), value);
        }
    }
    void mov_repc_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov_sv_to(MemImm8 b) {
        const Reg64 value = rax;
        c.movzx(value, word[REGS + offsetof(JitRegisters, sv)]);
        StoreToMemory(b, value);
    }
    void mov_x0_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov_x1_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov_y1_to(Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov(Imm16 a, ArArp b) {
        u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_r6(Imm16 a) {
        u16 value = a.Unsigned16();
        RegFromBus16(RegName::r6, value);
    }

    static void ThrowException() {
        throw UnimplementedException();
    }

    void mov_repc(Imm16 a) {
        NOT_IMPLEMENTED();
    }
    void mov_stepi0(Imm16 a) {
        u16 value = a.Unsigned16();
        c.mov(word[REGS + offsetof(JitRegisters, stepi0)], value);
        blk_key.stepi0 = value;
    }
    void mov_stepj0(Imm16 a) {
        u16 value = a.Unsigned16();
        c.mov(word[REGS + offsetof(JitRegisters, stepj0)], value);
        blk_key.stepj0 = value;
    }
    void mov(Imm16 a, SttMod b) {
        const u16 value = a.Unsigned16();
        RegFromBus16(b.GetName(), value);
    }
    void mov_prpage(Imm4 a) {
        NOT_IMPLEMENTED();
    }

    void mov_a0h_stepi0() {
        const Reg64 value = rax;
        RegToBus16(RegName::a0h, value, true);
        c.mov(word[REGS + offsetof(JitRegisters, stepi0)], value.cvt16());
        StorePC(regs.pc);
        compiling = false;
        link_ok = false;
    }
    void mov_a0h_stepj0() {
        const Reg64 value = rax;
        RegToBus16(RegName::a0h, value, true);
        c.mov(word[REGS + offsetof(JitRegisters, stepj0)], value.cvt16());
        StorePC(regs.pc);
        compiling = false;
        link_ok = false;
    }
    void mov_stepi0_a0h() {
        RegFromBus16(RegName::a0h, blk_key.stepi0);
    }
    void mov_stepj0_a0h() {
        RegFromBus16(RegName::a0h, blk_key.stepj0);
    }

    void mov_prpage(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov_repc(Abl a) {
        NOT_IMPLEMENTED();
    }
    void mov(Abl a, ArArp b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        RegFromBus16(b.GetName(), value);
    }
    void mov(Abl a, SttMod b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        RegFromBus16(b.GetName(), value);
    }

    void mov_prpage_to(Abl b) {
        NOT_IMPLEMENTED();
    }
    void mov_repc_to(Abl b) {
        NOT_IMPLEMENTED();
    }
    void mov(ArArp a, Abl b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        RegFromBus16(b.GetName(), value);
    }
    void mov(SttMod a, Abl b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        RegFromBus16(b.GetName(), value);
    }

    void mov_repc_to(ArRn1 b, ArStep1 bs) {
        NOT_IMPLEMENTED();
    }
    void mov(ArArp a, ArRn1 b, ArStep1 bs) {
        NOT_IMPLEMENTED();
    }
    void mov(SttMod a, ArRn1 b, ArStep1 bs) {
        NOT_IMPLEMENTED();
    }

    void mov_repc(ArRn1 a, ArStep1 as) {
        NOT_IMPLEMENTED();
    }
    void mov(ArRn1 a, ArStep1 as, ArArp b) {
        NOT_IMPLEMENTED();
    }
    void mov(ArRn1 a, ArStep1 as, SttMod b) {
        NOT_IMPLEMENTED();
    }

    void mov_repc_to(MemR7Imm16 b) {
        NOT_IMPLEMENTED();
    }
    void mov(ArArpSttMod a, MemR7Imm16 b) {
        NOT_IMPLEMENTED();
    }

    void mov_repc(MemR7Imm16 a) {
        NOT_IMPLEMENTED();
    }
    void mov(MemR7Imm16 a, ArArpSttMod b) {
        NOT_IMPLEMENTED();
    }

    void mov_pc(Ax a) {
        instruction_cycles = Cycles::IndirectBranch;
        StorePC(GetAccDirect(a.GetName()).cvt32());
        compiling = false;
    }
    void mov_pc(Bx a) {
        instruction_cycles = Cycles::IndirectBranch;
        StorePC(GetAccDirect(a.GetName()).cvt32());
        compiling = false;
    }

    void mov_mixp_to(Bx b) {
        NOT_IMPLEMENTED();
    }
    void mov_mixp_r6() {
        NOT_IMPLEMENTED();
    }
    void mov_p0h_to(Bx b) {
        NOT_IMPLEMENTED();
    }
    void mov_p0h_r6() {
        NOT_IMPLEMENTED();
    }
    void mov_p0h_to(Register b) {
        NOT_IMPLEMENTED();
    }
    void mov_p0(Ab a) {
        const Reg64 value = rax;
        GetAndSatAcc(value, a.GetName());
        ProductFromBus32(Px{0}, value.cvt32());
    }
    void mov_p1_to(Ab b) {
        const Reg64 value = rax;
        ProductToBus40(value, Px{1});
        SatAndSetAccAndFlag(b.GetName(), value);
    }

    void mov2(Px a, ArRn2 b, ArStep2 bs) {
        const Reg64 value = rbx;
        c.mov(value, dword[REGS + offsetof(JitRegisters, p) + sizeof(u32) * a.Index()]);
        u16 unit = GetArRnUnit(b);
        const Reg64 address = rcx;
        RnAddressAndModify(unit, GetArStep(bs), address);
        const Reg64 address2 = rax;
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(bs));
        // NOTE: keep the write order exactly like this.
        StoreToMemory(address2, value);
        c.shr(value, 16);
        StoreToMemory(address, value);
    }
    void mov2s(Px a, ArRn2 b, ArStep2 bs) {
        NOT_IMPLEMENTED();
    }
    void mov2(ArRn2 a, ArStep2 as, Px b) {
        u16 unit = GetArRnUnit(a);
        const Reg64 address = rax;
        RnAddressAndModify(unit, GetArStep(as), address);
        const Reg64 address2 = rbx;
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(as));
        const Reg64 value = rcx;
        LoadFromMemory(value, address);
        c.shl(value, 16);
        LoadFromMemory(value, address2);
        ProductFromBus32(b, value.cvt32());
    }
    void mova(Ab a, ArRn2 b, ArStep2 bs) {
        u16 unit = GetArRnUnit(b);
        const Reg64 value = rcx;
        GetAndSatAcc(value, a.GetName());
        const Reg64 address = rbx;
        const Reg64 address2 = rax;
        RnAddressAndModify(unit, GetArStep(bs), address);
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(bs));
        // NOTE: keep the write order exactly like this. The second one overrides the first one if
        // the offset is zero.
        StoreToMemory(address2, value);
        c.shr(value, 16);
        StoreToMemory(address, value);
    }
    void mova(ArRn2 a, ArStep2 as, Ab b) {
        u16 unit = GetArRnUnit(a);
        const Reg64 address = rax;
        RnAddressAndModify(unit, GetArStep(as), address);
        const Reg64 address2 = rbx;
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(as));
        const Reg64 value = rcx;
        LoadFromMemory(value, address);
        c.ror(value.cvt32(), 16);
        LoadFromMemory(value, address2);
        c.movsxd(value, value.cvt32());
        SatAndSetAccAndFlag(b.GetName(), value);
    }

    void mov_r6_to(Bx b) {
        NOT_IMPLEMENTED();
    }
    void mov_r6_mixp() {
        NOT_IMPLEMENTED();
    }
    void mov_r6_to(Register b) {
        const Reg64 value = rax;
        RegToBus16(RegName::r6, value);
        RegFromBus16(b.GetName(), value);
    }
    void mov_r6(Register a) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value, true);
        RegFromBus16(RegName::r6, value);
    }
    void mov_memsp_r6() {
        NOT_IMPLEMENTED();
    }
    void mov_r6_to(Rn b, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void mov_r6(Rn a, StepZIDS as) {
        NOT_IMPLEMENTED();
    }

    void mov2_axh_m_y0_m(Axh a, ArRn2 b, ArStep2 bs) {
        NOT_IMPLEMENTED();
    }

    void mov2_ax_mij(Ab a, ArpRn1 b, ArpStep1 bsi, ArpStep1 bsj) {
        NOT_IMPLEMENTED();
    }
    void mov2_ax_mji(Ab a, ArpRn1 b, ArpStep1 bsi, ArpStep1 bsj) {
        NOT_IMPLEMENTED();
    }
    void mov2_mij_ax(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov2_mji_ax(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, Ab b) {
        NOT_IMPLEMENTED();
    }
    void mov2_abh_m(Abh ax, Abh ay, ArRn1 b, ArStep1 bs) {
        const Reg64 u = rbx;
        const Reg64 v = rax;
        GetAndSatAccNoFlag(u, ax.GetName());
        c.shr(u, 16);
        GetAndSatAccNoFlag(v, ay.GetName());
        c.shr(v, 16);
        u16 unit = GetArRnUnit(b);
        const Reg64 ua = rcx;
        RnAddressAndModify(unit, GetArStep(bs), ua);
        const Reg64 va = rdx;
        c.mov(va, ua);
        OffsetAddress(unit, va.cvt16(), GetArOffset(bs));
        // keep the order
        StoreToMemory(va, v);
        StoreToMemory(ua, u);
    }
    void exchange_iaj(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        NOT_IMPLEMENTED();
    }
    void exchange_riaj(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        NOT_IMPLEMENTED();
    }
    void exchange_jai(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        auto [ui, uj] = GetArpRnUnit(b);
        auto [si, sj] = GetArpStep(bsi, bsj);
        const Reg64 i = rax;
        const Reg64 j = rbx;
        RnAddressAndModify(ui, si, i);
        RnAddressAndModify(uj, sj, j);
        const Reg64 value = rcx;
        GetAndSatAccNoFlag(value, a.GetName());
        c.shr(value, 16);
        StoreToMemory(i, value);
        LoadFromMemory(value, j);
        c.shl(value, 16);
        SignExtend(value, 32);
        SetAcc(a.GetName(), value);
    }
    void exchange_rjai(Axh a, ArpRn2 b, ArpStep2 bsi, ArpStep2 bsj) {
        NOT_IMPLEMENTED();
    }

    void movs(MemImm8 a, Ab b) {
        const Reg64 value = rax;
        const Reg64 address = rbx;
        c.mov(address, a.Unsigned16() + (blk_key.curr.mod1.page << 8));
        LoadFromMemory(value, address);
        SignExtend(value, 16);
        const Reg16 sv = cx;
        c.mov(sv, word[REGS + offsetof(JitRegisters, sv)]);
        ShiftBus40(value, sv, b.GetName());
    }
    void movs(Rn a, StepZIDS as, Ab b) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rax;
        LoadFromMemory(value, address);
        SignExtend(value, 16);
        const Reg16 sv = cx;
        c.mov(sv, word[REGS + offsetof(JitRegisters, sv)]);
        ShiftBus40(value, sv, b.GetName());
    }
    void movs(Register a, Ab b) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        SignExtend(value, 16);
        const Reg16 sv = cx;
        c.mov(sv, word[REGS + offsetof(JitRegisters, sv)]);
        ShiftBus40(value, sv, b.GetName());
    }
    void movs_r6_to(Ax b) {
        NOT_IMPLEMENTED();
    }
    void movsi(RnOld a, Ab b, Imm5s s) {
        const Reg64 value = rax;
        RegToBus16(a.GetName(), value);
        SignExtend(value, 16);
        u16 sv = s.Signed16();
        ShiftBus40(value, sv, b.GetName());
    }

    void movr(ArRn2 a, ArStep2 as, Abh b) {
        NOT_IMPLEMENTED();
    }
    void movr(Rn a, StepZIDS as, Ax b) {
        NOT_IMPLEMENTED();
    }
    void movr(Register a, Ax b) {
        NOT_IMPLEMENTED();
    }
    void movr(Bx a, Ax b) {
        NOT_IMPLEMENTED();
    }
    void movr_r6_to(Ax b) {
        NOT_IMPLEMENTED();
    }

    template <typename T>
    void Exp(Reg64 value, T count) {
        c.mov(rsi, value);
        c.not_(rsi);
        c.bt(value, 39);
        c.cmovc(value, rsi);
        c.shl(value, 64 - 39);
        c.or_(value, 0x1FFFFFF);
        c.lzcnt(count, value);
        c.sub(count, 8);
    }

    void ExpToSv(Reg64 value) {
        const Reg64 count = rcx;
        Exp(value, count);
        c.mov(word[REGS + offsetof(JitRegisters, sv)], count.cvt16());
    }

    void ExpStore(Ax b) {
        const Reg64 value = rax;
        c.movsx(value, word[REGS + offsetof(JitRegisters, sv)]);
        SetAcc(b.GetName(), value);
    }

    void exp(Bx a) {
        const Reg64 value = rbx;
        GetAcc(value, a.GetName());
        ExpToSv(value);
    }
    void exp(Bx a, Ax b) {
        exp(a);
        ExpStore(b);
    }
    void exp(Rn a, StepZIDS as) {
        const Reg64 address = rax;
        RnAddressAndModify(a.Index(), as.GetName(), address);
        const Reg64 value = rbx;
        LoadFromMemory(value, address);
        c.movsx(value, value.cvt16());
        c.shl(value, 16);
        ExpToSv(value);
    }
    void exp(Rn a, StepZIDS as, Ax b) {
        exp(a, as);
        ExpStore(b);
    }
    void exp(Register a) {
        const Reg64 value = rax;
        if (a.GetName() == RegName::a0 || a.GetName() == RegName::a1) {
            GetAcc(value, a.GetName());
        } else {
            // RegName::p follows the usual rule
            RegToBus16(a.GetName(), value);
            c.shl(value, 16);
            SignExtend(value, 32);
        }
        const Reg64 count = rbx;
        Exp(value, count);
        c.mov(word[REGS + offsetof(JitRegisters, sv)], count.cvt16());
    }
    void exp(Register a, Ax b) {
        exp(a);
        ExpStore(b);
    }
    void exp_r6() {
        const Reg64 value = rbx;
        RegToBus16(RegName::r6, value);
        c.shl(value, 16);
        SignExtend(value, 32);
        ExpToSv(value);
    }
    void exp_r6(Ax b) {
        exp_r6();
        ExpStore(b);
    }

    void lim(Ax a, Ax b) {
        const Reg64 value = rax;
        GetAcc(value, a.GetName());
        SaturateAcc<true>(value);
        SetAccAndFlag(b.GetName(), value);
    }

    void vtrclr0() {
        NOT_IMPLEMENTED();
    }
    void vtrclr1() {
        NOT_IMPLEMENTED();
    }
    void vtrclr() {
        NOT_IMPLEMENTED();
    }
    void vtrmov0(Axl a) {
        NOT_IMPLEMENTED();
    }
    void vtrmov1(Axl a) {
        NOT_IMPLEMENTED();
    }
    void vtrmov(Axl a) {
        NOT_IMPLEMENTED();
    }
    void vtrshr() {
        NOT_IMPLEMENTED();
    }

    void clrp0() {
        const Reg32 tmp = eax;
        c.xor_(tmp, tmp);
        ProductFromBus32(Px{0}, tmp);
    }
    void clrp1() {
        const Reg32 tmp = eax;
        c.xor_(tmp, tmp);
        ProductFromBus32(Px{1}, tmp);
    }
    void clrp() {
        const Reg32 tmp = eax;
        c.xor_(tmp, tmp);
        ProductFromBus32(Px{0}, tmp);
        ProductFromBus32(Px{1}, tmp);
    }

    void max_ge(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void max_gt(Ax a, StepZIDS bs) {
        const Reg64 u = rax;
        GetAcc(u, a.GetName());
        const Reg64 v = rbx;
        GetAcc(v, CounterAcc(a.GetName()));
        const Reg64 d = rcx;
        c.mov(d, v);
        c.sub(d, u);
        const Reg64 r0 = u;
        RnAndModify(0, bs.GetName(), r0);

        Xbyak::Label end_label, zero_fm;
        c.test(d, d);
        c.jle(zero_fm);
        c.bts(FLAGS, decltype(Flags::fm)::position);
        c.mov(word[REGS + offsetof(JitRegisters, mixp)], r0.cvt16());
        SetAcc(a.GetName(), v);
        c.jmp(end_label);
        c.L(zero_fm);
        c.btr(FLAGS, decltype(Flags::fm)::position);
        c.L(end_label);
    }
    void min_le(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void min_lt(Ax a, StepZIDS bs) {
        const Reg64 u = rax;
        const Reg64 v = rbx;
        GetAcc(u, a.GetName());
        GetAcc(v, CounterAcc(a.GetName()));
        const Reg64 d = rcx;
        c.mov(d, v);
        c.sub(d, u);
        const Reg64 r0 = u;
        RnAndModify(0, bs.GetName(), r0);

        Xbyak::Label end_label, false_label;
        c.bt(d, 63);
        c.jnc(false_label);
        c.or_(FLAGS, decltype(Flags::fm)::mask);
        c.mov(word[REGS + offsetof(JitRegisters, mixp)], r0.cvt16());
        SetAcc(a.GetName(), v);
        c.jmp(end_label);
        c.L(false_label);
        c.and_(FLAGS, ~decltype(Flags::fm)::mask);
        c.L(end_label);
    }

    void max_ge_r0(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void max_gt_r0(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void min_le_r0(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }
    void min_lt_r0(Ax a, StepZIDS bs) {
        NOT_IMPLEMENTED();
    }

    void divs(MemImm8 a, Ax b) {
        NOT_IMPLEMENTED();
    }

    void sqr_sqr_add3(Ab a, Ab b) {
        NOT_IMPLEMENTED();
    }

    void sqr_sqr_add3(ArRn2 a, ArStep2 as, Ab b) {
        NOT_IMPLEMENTED();
    }

    void sqr_mpysu_add3a(Ab a, Ab b) {
        NOT_IMPLEMENTED();
    }

    void cmp(Ax a, Bx b) {
        NOT_IMPLEMENTED();
    }
    void cmp_b0_b1() {
        NOT_IMPLEMENTED();
    }
    void cmp_b1_b0() {
        NOT_IMPLEMENTED();
    }
    void cmp(Bx a, Ax b) {
        NOT_IMPLEMENTED();
    }
    void cmp_p1_to(Ax b) {
        NOT_IMPLEMENTED();
    }

    void max2_vtr(Ax a) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr(Ax a) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr(Ax a, Bx b) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr(Ax a, Bx b) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movl(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movh(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movl(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movh(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movl(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movh(Ax a, Bx b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movl(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movh(Bx a, Ax b, ArRn1 c, ArStep1 cs) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movij(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        NOT_IMPLEMENTED();
    }
    void max2_vtr_movji(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movij(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        NOT_IMPLEMENTED();
    }
    void min2_vtr_movji(Ax a, Bx b, ArpRn1 c, ArpStep1 csi, ArpStep1 csj) {
        NOT_IMPLEMENTED();
    }

    template <typename ArpStepX>
    void mov_sv_app(ArRn1 a, ArpStepX as, Bx b, SumBase base, bool sub_p0, bool p0_align,
                    bool sub_p1, bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void cbs(Axh a, CbsCond c) {
        NOT_IMPLEMENTED();
    }
    void cbs(Axh a, Bxh b, CbsCond c) {
        NOT_IMPLEMENTED();
    }
    void cbs(ArpRn1 a, ArpStep1 asi, ArpStep1 asj, CbsCond c) {
        NOT_IMPLEMENTED();
    }

    void ProductSum(SumBase base, RegName acc, bool sub_p0, bool p0_align, bool sub_p1,
                    bool p1_align) {
        const Reg64 value_a = rax;
        const Reg64 value_b = rbx;
        ProductToBus40(value_a, Px{0});
        ProductToBus40(value_b, Px{1});
        if (p0_align) {
            c.shr(value_a, 16);
            SignExtend(value_a, 24);
        }
        if (p1_align) {
            c.shr(value_b, 16);
            SignExtend(value_b, 24);
        }
        const Reg64 value_c = rcx;
        switch (base) {
        case SumBase::Zero:
            c.xor_(value_c, value_c);
            break;
        case SumBase::Acc:
            GetAcc(value_c, acc);
            break;
        case SumBase::Sv:
            c.movzx(value_c, word[REGS + offsetof(JitRegisters, sv)]);
            c.shl(value_c, 16);
            SignExtend(value_c, 32);
            break;
        case SumBase::SvRnd:
            c.movzx(value_c, word[REGS + offsetof(JitRegisters, sv)]);
            c.shl(value_c, 16);
            SignExtend(value_c, 32);
            c.or_(value_c, 0x8000);
            break;
        default:
            UNREACHABLE();
        }
        const Reg64 result = rdx;
        AddSub(value_c, value_a, result, sub_p0);
        const Reg32 temp = value_a.cvt32();
        if (!light_flags) {
            c.mov(temp, FLAGS);
            c.and_(temp, decltype(Flags::fc0)::mask | decltype(Flags::fv)::mask); // Keep fc0 and fv
        }
        const Reg64 result2 = value_c;
        AddSub(result, value_b, result2, sub_p1);
        // Is this correct?
        if (!light_flags) {
            if (sub_p0 == sub_p1) {
                c.or_(FLAGS, temp);
            } else {
                c.xor_(FLAGS, temp);
            }
        }
        SatAndSetAccAndFlag(acc, result2);
    }

    void mma(RegName a, bool x0_sign, bool y0_sign, bool x1_sign, bool y1_sign, SumBase base,
             bool sub_p0, bool p0_align, bool sub_p1, bool p1_align) {
        NOT_IMPLEMENTED();
    }

    template <typename ArpRnX, typename ArpStepX>
    void mma(ArpRnX xy, ArpStepX i, ArpStepX j, bool dmodi, bool dmodj, RegName a, bool x0_sign,
             bool y0_sign, bool x1_sign, bool y1_sign, SumBase base, bool sub_p0, bool p0_align,
             bool sub_p1, bool p1_align) {
        light_ok = true;
        ProductSum(base, a, sub_p0, p0_align, sub_p1, p1_align);
        auto [ui, uj] = GetArpRnUnit(xy);
        auto [si, sj] = GetArpStep(i, j);
        auto [oi, oj] = GetArpOffset(i, j);
        const Reg64 x = rbx;
        RnAddressAndModify(ui, si, x, dmodi);
        const Reg64 x2 = rax;
        c.mov(x2, x);
        OffsetAddress(ui, x2.cvt16(), oi, dmodi);
        const Reg64 factors = rcx;
        LoadFromMemory(factors, x2);
        c.shl(factors, 16);
        LoadFromMemory(factors, x);
        c.shl(factors, 16);
        const Reg64 y = x;
        RnAddressAndModify(uj, sj, y, dmodj);
        const Reg64 y2 = x2;
        c.mov(y2, y);
        OffsetAddress(uj, y2.cvt16(), oj, dmodj);
        LoadFromMemory(factors, y2);
        c.shl(factors, 16);
        LoadFromMemory(factors, y);
        c.mov(FACTORS, factors);
        DoMultiplication(0, eax, ebx, x0_sign, y0_sign);
        DoMultiplication(1, eax, ebx, x1_sign, y1_sign);
    }

    void mma_mx_xy(ArRn1 y, ArStep1 ys, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void mma_xy_mx(ArRn1 y, ArStep1 ys, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void mma_my_my(ArRn1 x, ArStep1 xs, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                   bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                   bool p1_align) {
        ProductSum(base, a, sub_p0, p0_align, sub_p1, p1_align);
        u16 unit = GetArRnUnit(x);
        const Reg64 address = rax;
        RnAddressAndModify(unit, GetArStep(xs), address);
        const Reg64 address2 = rbx;
        c.mov(address2, address);
        OffsetAddress(unit, address2.cvt16(), GetArOffset(xs));
        c.rorx(FACTORS, FACTORS, 32);
        LoadFromMemory(FACTORS, address);
        c.rorx(FACTORS, FACTORS, 16);
        LoadFromMemory(FACTORS, address2);
        c.rorx(FACTORS, FACTORS, 16);
        DoMultiplication(0, eax, ebx, x0_sign, y0_sign);
        DoMultiplication(1, eax, ebx, x1_sign, y1_sign);
    }

    void mma_mov(Axh u, Bxh v, ArRn1 w, ArStep1 ws, RegName a, bool x0_sign, bool y0_sign,
                 bool x1_sign, bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                 bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void mma_mov(ArRn2 w, ArStep1 ws, RegName a, bool x0_sign, bool y0_sign, bool x1_sign,
                 bool y1_sign, SumBase base, bool sub_p0, bool p0_align, bool sub_p1,
                 bool p1_align) {
        NOT_IMPLEMENTED();
    }

    void addhp(ArRn2 a, ArStep2 as, Px b, Ax c) {
        NOT_IMPLEMENTED();
    }

    void mov_ext0(Imm8s a) {
        NOT_IMPLEMENTED();
    }
    void mov_ext1(Imm8s a) {
        NOT_IMPLEMENTED();
    }
    void mov_ext2(Imm8s a) {
        NOT_IMPLEMENTED();
    }
    void mov_ext3(Imm8s a) {
        NOT_IMPLEMENTED();
    }

private:
    void ConditionPass(Cond cond, auto&& func) {
        Xbyak::Label end_cond;
        switch (cond.GetName()) {
        case CondValue::True:
            func();
            return;
        case CondValue::Eq:
            // return fz == 1;
            c.test(FLAGS, decltype(Flags::fz)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::Neq:
            // return fz == 0;
            c.test(FLAGS, decltype(Flags::fz)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::Gt:
            // return fz == 0 && fm == 0;
            c.test(FLAGS, decltype(Flags::fz)::mask | decltype(Flags::fm)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::Ge:
            // return fm == 0;
            c.test(FLAGS, decltype(Flags::fm)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::Lt:
            // return fm == 1;
            c.test(FLAGS, decltype(Flags::fm)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::Le:
            // return fm == 1 || fz == 1;
            c.test(FLAGS, decltype(Flags::fm)::mask | decltype(Flags::fz)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::Nn:
            // return fn == 0;
            c.test(FLAGS, decltype(Flags::fn)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::C:
            // return fc0 == 1;
            c.test(FLAGS, decltype(Flags::fc0)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::V:
            // return fv == 1;
            c.test(FLAGS, decltype(Flags::fv)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::E:
            // return fe == 1;
            c.test(FLAGS, decltype(Flags::fe)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::L:
            // return flm == 1 || fvl == 1;
            c.test(FLAGS, decltype(Flags::flm)::mask | decltype(Flags::fvl)::mask);
            c.jz(end_cond, c.T_NEAR);
            break;
        case CondValue::Nr:
            // return fr == 0;
            c.test(FLAGS, decltype(Flags::fr)::mask);
            c.jnz(end_cond, c.T_NEAR);
            break;
        case CondValue::Niu0:
        case CondValue::Iu0:
        case CondValue::Iu1:
        default:
            NOT_IMPLEMENTED();
            return;
        }
        func();
        c.L(end_cond);
    }

    void RegToBus16(RegName reg, Reg64 out, bool enable_sat_for_mov = false) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            GetAcc(out, reg);
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            if (enable_sat_for_mov) {
                GetAndSatAcc(out, reg);
            } else {
                GetAcc(out, reg);
            }
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            if (enable_sat_for_mov) {
                GetAndSatAcc(out, reg);
            } else {
                GetAcc(out, reg);
            }
            c.shr(out, 16);
            c.and_(out, 0xFFFF);
            break;
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            NOT_IMPLEMENTED();

        case RegName::r0:
            c.movzx(out, R0_1_2_3.cvt16());
            break;
        case RegName::r1:
            c.rorx(out, R0_1_2_3, 16);
            break;
        case RegName::r2:
            c.rorx(out, R0_1_2_3, 32);
            break;
        case RegName::r3:
            c.rorx(out, R0_1_2_3, 48);
            break;
        case RegName::r4:
            c.movzx(out, R4_5_6_7.cvt16());
            break;
        case RegName::r5:
            c.rorx(out, R4_5_6_7, 16);
            break;
        case RegName::r6:
            c.rorx(out, R4_5_6_7, 32);
            break;
        case RegName::r7:
            c.rorx(out, R4_5_6_7, 48);
            break;
        case RegName::y0:
            c.movzx(out, FACTORS.cvt16());
            break;

        case RegName::p:
            ProductToBus40(out, Px{0});
            c.shr(out, 16);
            c.and_(out, 0xFFFF);
            break;

        case RegName::sp:
            c.movzx(out, word[REGS + offsetof(JitRegisters, sp)]);
            break;
        case RegName::sv:
            c.movzx(out, word[REGS + offsetof(JitRegisters, sv)]);
            break;
        case RegName::lc:
            regs.GetLc(c, out);
            break;
        case RegName::ext0:
        case RegName::ext1:
        case RegName::ext2:
        case RegName::ext3:
            c.movzx(out, word[REGS + offsetof(JitRegisters, ext) +
                              sizeof(u16) * (static_cast<int>(reg) - static_cast<int>(RegName::ext0))]);
            break;
        case RegName::stt0:
            regs.GetStt0(c, out.cvt16());
            break;
        case RegName::stt1:
            regs.GetStt1(c, out.cvt16());
            break;
        case RegName::stt2:
            regs.GetStt2(c, out.cvt16());
            break;
        case RegName::st0:
            regs.GetSt0(c, out.cvt16(), blk_key.curr.mod0);
            break;
        case RegName::st1:
            regs.GetSt1(c, out.cvt16(), blk_key.curr.mod0, blk_key.curr.mod1);
            break;
        case RegName::st2:
            regs.GetSt2(c, out.cvt16(), blk_key.curr.mod0, blk_key.curr.mod2);
            break;

        case RegName::cfgi:
            c.mov(out, blk_key.cfgi.raw);
            break;
        case RegName::cfgj:
            c.mov(out, blk_key.cfgj.raw);
            break;

        case RegName::mod0:
            c.mov(out, blk_key.curr.mod0.raw & Mod0::Mask());
            break;
        case RegName::mod1:
            c.mov(out, blk_key.curr.mod1.raw & Mod1::Mask());
            break;
        case RegName::mod2:
            c.mov(out, blk_key.curr.mod2.raw);
            break;
        case RegName::mod3:
            regs.GetMod3(c, out);
            break;

        case RegName::ar0:
            c.mov(out, blk_key.curr.ar[0].raw & ArU::Mask());
            break;
        case RegName::ar1:
            c.mov(out, blk_key.curr.ar[1].raw & ArU::Mask());
            break;
        case RegName::arp0:
            c.mov(out, blk_key.curr.arp[0].raw & ArpU::Mask());
            break;
        case RegName::arp1:
            c.mov(out, blk_key.curr.arp[1].raw & ArpU::Mask());
            break;
        case RegName::arp2:
            c.mov(out, blk_key.curr.arp[2].raw & ArpU::Mask());
            break;
        case RegName::arp3:
            c.mov(out, blk_key.curr.arp[3].raw & ArpU::Mask());
            break;

        default:
            NOT_IMPLEMENTED();
        }
    }

    void GetAcc(Xbyak::Reg64 out, RegName name) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            c.mov(out, A[0]);
            break;
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            c.mov(out, A[1]);
            break;
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            c.mov(out, B[0]);
            break;
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            c.mov(out, B[1]);
            break;
        default:
            NOT_IMPLEMENTED();
        }
    }

    Reg64 GetAccDirect(RegName name) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            return A[0];
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            return A[1];
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            return B[0];
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            return B[1];
        default:
            UNREACHABLE();
        }
    }

    void RegFromBus16(RegName reg, Reg64 value) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            SignExtend(value, 16);
            SatAndSetAccAndFlag(reg, value);
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            c.movzx(value, value.cvt16());
            SatAndSetAccAndFlag(reg, value);
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            c.shl(value, 16);
            SignExtend(value, 32);
            SatAndSetAccAndFlag(reg, value);
            break;

        case RegName::r0:
            c.mov(R0_1_2_3.cvt16(), value.cvt16());
            break;
        case RegName::r1:
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            c.mov(R0_1_2_3.cvt16(), value.cvt16());
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
            break;
        case RegName::r2:
            c.rorx(R0_1_2_3, R0_1_2_3, 32);
            c.mov(R0_1_2_3.cvt16(), value.cvt16());
            c.rorx(R0_1_2_3, R0_1_2_3, 32);
            break;
        case RegName::r3:
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
            c.mov(R0_1_2_3.cvt16(), value.cvt16());
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            break;
        case RegName::r4:
            c.mov(R4_5_6_7.cvt16(), value.cvt16());
            break;
        case RegName::r5:
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            c.mov(R4_5_6_7.cvt16(), value.cvt16());
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            break;
        case RegName::r6:
            c.rorx(R4_5_6_7, R4_5_6_7, 32);
            c.mov(R4_5_6_7.cvt16(), value.cvt16());
            c.rorx(R4_5_6_7, R4_5_6_7, 32);
            break;
        case RegName::r7:
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            c.mov(R4_5_6_7.cvt16(), value.cvt16());
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            break;

        case RegName::st1:
            regs.SetSt1(c, value, blk_key.curr.mod0, blk_key.curr.mod1);
            StorePC(regs.pc);
            compiling = false; // Modifies static state, end block
            link_ok = false;
            break;
        case RegName::sp:
            c.mov(word[REGS + offsetof(JitRegisters, sp)], value.cvt16());
            break;
        case RegName::y0:
            c.mov(FACTORS.cvt16(), value.cvt16());
            break;
        case RegName::p:
            c.cmp(value.cvt16(), 0x7FFF);
            c.seta(byte[REGS + offsetof(JitRegisters, pe)]);
            c.mov(word[REGS + offsetof(JitRegisters, p) + sizeof(u16)], value.cvt16());
            break;

        case RegName::sv:
            c.mov(word[REGS + offsetof(JitRegisters, sv)], value.cvt16());
            break;

        case RegName::lc:
            regs.SetLc(c, value);
            break;

        case RegName::cfgi:
            c.mov(word[REGS + offsetof(JitRegisters, cfgi)], value.cvt16());
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;
        case RegName::cfgj:
            c.mov(word[REGS + offsetof(JitRegisters, cfgj)], value.cvt16());
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;

        case RegName::st0:
            regs.SetSt0(c, value, blk_key.curr.mod0);
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;

        case RegName::stt0:
            regs.SetStt0(c, value);
            break;
        case RegName::stt1:
            regs.SetStt1(c, value);
            break;
        case RegName::stt2:
            regs.SetStt2(c, value);
            break;

        case RegName::mod0:
            c.mov(word[REGS + offsetof(JitRegisters, mod0)], value.cvt16());
            StorePC(regs.pc);
            compiling = false;
            link_ok = false;
            break;
        case RegName::mod1:
            c.mov(word[REGS + offsetof(JitRegisters, mod1)], value.cvt16());
            StorePC(regs.pc);
            compiling = false;
            link_ok = false;
            break;
        case RegName::mod2:
            c.mov(word[REGS + offsetof(JitRegisters, mod2)], value.cvt16());
            StorePC(regs.pc);
            compiling = false;
            link_ok = false;
            break;
        case RegName::mod3:
            regs.SetMod3(c, value);
            StorePC(regs.pc);
            compiling = false; // cpc is compiled into calls and returns
            link_ok = false;
            break;

        case RegName::st2:
            regs.SetSt2(c, value, blk_key.curr.mod0, blk_key.curr.mod2);
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;

        case RegName::ext0:
        case RegName::ext1:
        case RegName::ext2:
        case RegName::ext3:
            c.mov(word[REGS + offsetof(JitRegisters, ext) +
                       sizeof(u16) * (static_cast<int>(reg) - static_cast<int>(RegName::ext0))],
                  value.cvt16());
            break;

        case RegName::ar0:
            c.mov(word[REGS + offsetof(JitRegisters, ar)], value.cvt16());
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;
        case RegName::ar1:
            c.mov(word[REGS + offsetof(JitRegisters, ar) + sizeof(u16)], value.cvt16());
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;
        case RegName::arp0:
            c.mov(word[REGS + offsetof(JitRegisters, arp)], value.cvt16());
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;
        case RegName::arp1:
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16)], value.cvt16());
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;
        case RegName::arp2:
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 2], value.cvt16());
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;
        case RegName::arp3:
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 3], value.cvt16());
            StorePC(regs.pc);
            compiling = false; // Static state changed, end block
            link_ok = false;
            break;

        default:
            NOT_IMPLEMENTED();
        }
    }

    void RegFromBus16(RegName reg, u16 value) {
        switch (reg) {
        case RegName::a0:
        case RegName::a1:
        case RegName::b0:
        case RegName::b1:
            SatAndSetAccAndFlag(reg, ::SignExtend<16, u64>(value));
            break;
        case RegName::a0l:
        case RegName::a1l:
        case RegName::b0l:
        case RegName::b1l:
            SatAndSetAccAndFlag(reg, (u64)value);
            break;
        case RegName::a0h:
        case RegName::a1h:
        case RegName::b0h:
        case RegName::b1h:
            SatAndSetAccAndFlag(reg, ::SignExtend<32, u64>(u32(value) << 16));
            break;
        case RegName::a0e:
        case RegName::a1e:
        case RegName::b0e:
        case RegName::b1e:
            NOT_IMPLEMENTED();

        case RegName::r0:
            c.mov(R0_1_2_3.cvt16(), value);
            break;
        case RegName::r1:
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            c.mov(R0_1_2_3.cvt16(), value);
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
            break;
        case RegName::r2:
            c.rorx(R0_1_2_3, R0_1_2_3, 32);
            c.mov(R0_1_2_3.cvt16(), value);
            c.rorx(R0_1_2_3, R0_1_2_3, 32);
            break;
        case RegName::r3:
            c.rorx(R0_1_2_3, R0_1_2_3, 48);
            c.mov(R0_1_2_3.cvt16(), value);
            c.rorx(R0_1_2_3, R0_1_2_3, 16);
            break;
        case RegName::r4:
            c.mov(R4_5_6_7.cvt16(), value);
            break;
        case RegName::r5:
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            c.mov(R4_5_6_7.cvt16(), value);
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            break;
        case RegName::r6:
            c.rorx(R4_5_6_7, R4_5_6_7, 32);
            c.mov(R4_5_6_7.cvt16(), value);
            c.rorx(R4_5_6_7, R4_5_6_7, 32);
            break;
        case RegName::r7:
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            c.mov(R4_5_6_7.cvt16(), value);
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            break;

        case RegName::y0:
            c.mov(FACTORS.cvt16(), value);
            break;
        case RegName::p:
            c.mov(word[REGS + offsetof(JitRegisters, pe)], value > 0x7FFF);
            c.mov(word[REGS + offsetof(JitRegisters, p) + sizeof(u16)], value);
            break;

        case RegName::sp:
            c.mov(word[REGS + offsetof(JitRegisters, sp)], value);
            break;
        case RegName::mod0:
            blk_key.curr.mod0.raw = value;
            regs.SetMod0(c, value);
            break;
        case RegName::mod1:
            blk_key.curr.mod1.raw = value;
            regs.SetMod1(c, value);
            break;
        case RegName::mod2:
            blk_key.curr.mod2.raw = value;
            regs.SetMod2(c, value);
            break;
        case RegName::mod3:
            regs.SetMod3(c, value);
            StorePC(regs.pc);
            compiling = false; // cpc is compiled into calls and returns
            link_ok = false;
            break;

        case RegName::ext0:
        case RegName::ext1:
        case RegName::ext2:
        case RegName::ext3:
            c.mov(word[REGS + offsetof(JitRegisters, ext) +
                       sizeof(u16) * (static_cast<int>(reg) - static_cast<int>(RegName::ext0))],
                  value);
            break;

        case RegName::cfgi:
            blk_key.cfgi.raw = value;
            regs.SetCfgi(c, value);
            break;
        case RegName::cfgj:
            blk_key.cfgj.raw = value;
            regs.SetCfgj(c, value);
            break;

        case RegName::sv:
            c.mov(word[REGS + offsetof(JitRegisters, sv)], value);
            break;

        case RegName::st0:
            regs.SetSt0(c, value, blk_key.curr.mod0);
            EndBlockForInterrupts();
            break;
        case RegName::st1:
            regs.SetSt1(c, value, blk_key.curr.mod0, blk_key.curr.mod1);
            break;
        case RegName::st2:
            regs.SetSt2(c, value, blk_key.curr.mod0, blk_key.curr.mod2);
            EndBlockForInterrupts();
            break;

        case RegName::ar0:
            blk_key.curr.ar[0].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, ar)], value);
            break;
        case RegName::ar1:
            blk_key.curr.ar[1].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, ar) + sizeof(u16)], value);
            break;

        case RegName::arp0:
            blk_key.curr.arp[0].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp)], value);
            break;
        case RegName::arp1:
            blk_key.curr.arp[1].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16)], value);
            break;
        case RegName::arp2:
            blk_key.curr.arp[2].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 2], value);
            break;
        case RegName::arp3:
            blk_key.curr.arp[3].raw = value;
            c.mov(word[REGS + offsetof(JitRegisters, arp) + sizeof(u16) * 3], value);
            break;

        default:
            NOT_IMPLEMENTED();
        }
    }

    template <typename T>
    void SetAccFlag(T value) {
        if (light_flags) {
            return;
        }
        constexpr u16 ACC_MASK = ~(decltype(Flags::fz)::mask | decltype(Flags::fm)::mask |
                                   decltype(Flags::fe)::mask | decltype(Flags::fn)::mask);
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            Xbyak::Label fe_set;
            c.and_(FLAGS, ACC_MASK);
            // fe = value != SignExtend<32>(value)
            c.movsxd(rbp, value.cvt32());
            c.cmp(rbp, value);
            c.setne(bpl);
            c.movzx(ebp, bpl);
            c.shl(ebp, decltype(Flags::fe)::position);
            c.or_(FLAGS, ebp);
            // fm = (value >> 39) != 0
            c.mov(rbp, value);
            c.shr(rbp, 39);
            c.setnz(bpl);
            c.movzx(ebp, bpl);
            c.shl(ebp, decltype(Flags::fm)::position);
            c.or_(FLAGS, ebp);
            // fn = !fe && bit31 != bit30, which is when doubling the low word overflows
            c.test(FLAGS, decltype(Flags::fe)::mask);
            c.jnz(fe_set);
            c.mov(ebp, value.cvt32());
            c.add(ebp, ebp);
            c.seto(bpl);
            c.movzx(ebp, bpl);
            c.shl(ebp, decltype(Flags::fn)::position);
            c.or_(FLAGS, ebp);
            c.L(fe_set);
            // fz = value == 0, which also sets fn
            c.test(value, value);
            c.setz(bpl);
            c.movzx(ebp, bpl);
            c.imul(ebp, ebp, decltype(Flags::fz)::mask | decltype(Flags::fn)::mask);
            c.or_(FLAGS, ebp);
        } else {
            const u64 bit31 = (value >> 31) & 1;
            const u64 bit30 = (value >> 30) & 1;
            const u16 fz = value == 0;
            const u16 fm = (value >> 39) != 0;
            const u16 fe = value != ::SignExtend<32>(value);
            const u16 fn = fz || (!fe && (bit31 ^ bit30) != 0);

            u16 mask = 0;
            mask |= fz << decltype(Flags::fz)::position;
            mask |= fm << decltype(Flags::fm)::position;
            mask |= fe << decltype(Flags::fe)::position;
            mask |= fn << decltype(Flags::fn)::position;

            c.and_(FLAGS, ACC_MASK); // clear fz, fm, fe, fn
            c.or_(FLAGS, mask);
        }
    }

    template <typename T>
    void SetAcc(RegName name, T value) {
        switch (name) {
        case RegName::a0:
        case RegName::a0h:
        case RegName::a0l:
        case RegName::a0e:
            c.mov(A[0], value);
            break;
        case RegName::a1:
        case RegName::a1h:
        case RegName::a1l:
        case RegName::a1e:
            c.mov(A[1], value);
            break;
        case RegName::b0:
        case RegName::b0h:
        case RegName::b0l:
        case RegName::b0e:
            c.mov(B[0], value);
            break;
        case RegName::b1:
        case RegName::b1h:
        case RegName::b1l:
        case RegName::b1e:
            c.mov(B[1], value);
            break;
        default:
            UNREACHABLE();
        }
    }

    template <bool flag, typename T>
    void SaturateAcc(T& value) {
        if constexpr (std::is_base_of_v<Xbyak::Reg, T>) {
            Xbyak::Label end_saturate;
            c.movsxd(rsi, value.cvt32()); // rbx = SignExtend<32>(value);
            c.cmp(value, rsi);
            c.je(end_saturate);
            if constexpr (flag) {
                c.or_(FLAGS, decltype(Flags::flm)::mask); // regs.flm = 1;
            }
            c.shr(value, 39);
            c.mov(rsi, 0x0000'0000'7FFF'FFFF);
            c.test(value, value);
            c.mov(value, 0xFFFF'FFFF'8000'0000);
            c.cmovz(value, rsi);
            // note: flm doesn't change value otherwise
            c.L(end_saturate);
        } else {
            if (value != ::SignExtend<32>(value)) {
                if constexpr (flag) {
                    c.or_(FLAGS, decltype(Flags::flm)::mask); // regs.flm = 1;
                }
                if ((value >> 39) != 0)
                    value = 0xFFFF'FFFF'8000'0000;
                else
                    value = 0x0000'0000'7FFF'FFFF;
            }
            // note: flm doesn't change value otherwise
        }
    }

    template <typename T>
    void SatAndSetAccAndFlag(RegName name, T value) {
        SetAccFlag(value);
        if (!blk_key.curr.mod0.sata) {
            SaturateAcc<true>(value);
        }
        SetAcc(name, value);
    }

    void SetAccAndFlag(RegName name, Reg64 value) {
        SetAccFlag(value);
        SetAcc(name, value);
    }

    void GetAndSatAcc(Reg64 out, RegName name) {
        GetAcc(out, name);
        if (!blk_key.curr.mod0.sat) {
            SaturateAcc<true>(out);
        }
    }

    void GetAndSatAccNoFlag(Reg64 out, RegName name) {
        GetAcc(out, name);
        if (!blk_key.curr.mod0.sat) {
            SaturateAcc<false>(out);
        }
    }

    void ProductToBus40(Reg64 value, Px reg) {
        const u16 unit = reg.Index();
        c.mov(value, word[REGS + offsetof(JitRegisters, pe) + sizeof(u16) * unit]);
        c.shl(value, 32);
        c.xor_(rsi, rsi);
        c.mov(esi, dword[REGS + offsetof(JitRegisters, p) + sizeof(u32) * unit]);
        c.or_(value, rsi);
        switch (unit == 0 ? blk_key.curr.mod0.ps0.Value() : blk_key.curr.mod0.ps1.Value()) {
        case 0:
            SignExtend(value, 33);
            break;
        case 1:
            c.shr(value, 1);
            SignExtend(value, 32);
            break;
        case 2:
            c.shl(value, 1);
            SignExtend(value, 34);
            break;
        case 3:
            c.shl(value, 2);
            SignExtend(value, 35);
            break;
        }
    }

    void RnAddress(u32 unit, Reg64 value) {
        if (blk_key.curr.mod2.IsBr(unit) && !blk_key.curr.mod2.IsM(unit)) {
            c.rol(value.cvt16(), 8);
            c.mov(esi, value.cvt32());
            c.and_(esi, 3855);
            c.shl(esi, 4);
            c.shr(value.cvt32(), 4);
            c.and_(value.cvt32(), 3855);
            c.or_(value.cvt32(), esi);
            c.mov(esi, value.cvt32());
            c.and_(esi, 13107);
            c.shr(value.cvt32(), 2);
            c.and_(value.cvt32(), 13107);
            c.lea(esi, ptr[value + 4*rsi]);
            c.mov(edx, esi);
            c.and_(edx, 21845);
            c.shr(esi, 1);
            c.and_(esi, 21845);
            c.lea(esi, ptr[rsi + 2*rdx]);
            c.mov(value.cvt32(), esi);
        }
    }

    enum class OffsetValue : u16 {
        Zero = 0,
        PlusOne = 1,
        MinusOne = 2,
        MinusOneDmod = 3,
    };

    void OffsetAddress(u32 unit, Reg16 address, OffsetValue offset, bool dmod = false) {
        if (offset == OffsetValue::Zero)
            return;
        if (offset == OffsetValue::MinusOneDmod) {
            c.sub(address, 1);
            return;
        }
        const bool emod = blk_key.curr.mod2.IsM(unit) && !blk_key.curr.mod2.IsBr(unit) && !dmod;
        u16 mod = unit < 4 ? blk_key.cfgi.mod : blk_key.cfgj.mod;
        [[maybe_unused]] u16 mask = 1; // mod = 0 still have one bit mask
        for (u32 i = 0; i < 9; ++i) {
            mask |= mod >> i;
        }
        if (offset == OffsetValue::PlusOne) {
            if (!emod) {
                c.add(address, 1);
                return;
            }
            // if ((address & mask) == mod)
            //    return address & ~mask;
            // return address + 1;
            Xbyak::Label addressDoesNotEqualMask;

            // Note: This section has a LOT of register pressure!
            const Reg16 tmp = si;
            c.mov(tmp, address);
            c.add(address, 1);
            c.and_(tmp, mask);
            c.cmp(tmp, mod);
            c.jne(addressDoesNotEqualMask);
            // Subtract 1 to get the original value
            c.mov(tmp, ~mask);
            c.sub(address, 1);
            // Xbyak seems to have issues with AND r, imm16, so we need to store the value to a register
            c.and_(address, tmp);
            c.L(addressDoesNotEqualMask);
        } else { // OffsetValue::MinusOne
            if (!emod) {
                c.sub(address, 1);
                return;
            }
            NOT_IMPLEMENTED();
            // TODO: sometimes this would return two addresses,
            // neither of which is the original Rn value.
            // This only happens for memory writing, but not for memory reading.
            // Might be some undefined behaviour.
            // if ((address & mask) == 0)
            //    return address | mod;
            // return address - 1;
        }
    }

    void StepAddress(u32 unit, Reg16 address, StepValue step, bool dmod = false) {
        u16 s;
        bool legacy = blk_key.curr.mod1.cmd;
        bool step2_mode1 = false;
        bool step2_mode2 = false;
        switch (step) {
        case StepValue::Zero:
            s = 0;
            break;
        case StepValue::Increase:
            s = 1;
            break;
        case StepValue::Decrease:
            s = 0xFFFF;
            break;
        // TODO: Increase/Decrease2Mode1/2 sometimes have wrong result if Offset=+/-1.
        // This however never happens with modr instruction.
        case StepValue::Increase2Mode1:
            s = 2;
            step2_mode1 = !legacy;
            break;
        case StepValue::Decrease2Mode1:
            s = 0xFFFE;
            step2_mode1 = !legacy;
            break;
        case StepValue::Increase2Mode2:
            s = 2;
            step2_mode2 = !legacy;
            break;
        case StepValue::Decrease2Mode2:
            s = 0xFFFE;
            step2_mode2 = !legacy;
            break;
        case StepValue::PlusStep: {
            if (blk_key.curr.mod2.IsBr(unit) && !blk_key.curr.mod2.IsM(unit)) {
                s = unit < 4 ? blk_key.stepi0 : blk_key.stepj0;
            } else {
                s = unit < 4 ? blk_key.cfgi.step : blk_key.cfgj.step;
                s = ::SignExtend<7>(s);
            }
            if (blk_key.curr.mod1.stp16 == 1 && !legacy) {
                s = unit < 4 ? regs.stepi0 : regs.stepj0;
                if (blk_key.curr.mod2.IsM(unit)) {
                    s = ::SignExtend<9>(s);
                }
            }
            break;
        }
        default:
            UNREACHABLE();
        }

        if (s == 0)
            return;
        if (!dmod && !blk_key.curr.mod2.IsBr(unit) && blk_key.curr.mod2.IsM(unit)) {
            u16 mod = unit < 4 ? blk_key.cfgi.mod : blk_key.cfgj.mod;

            if (mod == 0) {
                return;
            }

            if (mod == 1 && step2_mode2) {
                return;
            }

            u32 iteration = 1;
            if (step2_mode1) {
                iteration = 2;
                s = ::SignExtend<15, u16>(s >> 1);
            }

            for (u32 i = 0; i < iteration; ++i) {
                if (legacy || step2_mode2) {
                    bool negative = false;
                    u16 m = mod;
                    if (s >> 15) {
                        negative = true;
                        m |= ~s;
                    } else {
                        m |= s;
                    }

                    u16 mask = (1 << std20::log2p1(m)) - 1;
                    const Reg16 scratch = rsi.cvt16();
                    const Reg16 next = rdx.cvt16();
                    if (!negative) {
                        if (!((!step2_mode2 || mod != mask))) {
                            c.mov(next, address);
                            c.add(next, s);
                            c.and_(next, mask);
                        } else {
                            c.mov(next, address);
                            c.add(next, s);
                            c.and_(next, mask);
                            c.mov(scratch, address);
                            c.and_(scratch, mask);
                            c.cmp(scratch, mod);
                            c.setne(scratch.cvt8());
                            c.shl(scratch, 15);
                            c.sar(scratch, 15);
                            c.and_(next, scratch);
                        }
                    } else {
                        if (!((!step2_mode2 || mod != mask))) {
                            c.mov(next, address);
                            c.add(next.cvt32(), s);
                            c.and_(next, mask);
                        } else {
                            c.mov(next, address);
                            c.add(next.cvt32(), s);
                            c.and_(next, mask);
                            c.mov(scratch, mod);
                            c.test(address, mask);
                            c.cmovz(next, scratch);
                        }
                    }
                    c.and_(address, ~mask);
                    c.or_(address, next);
                } else {
                    u16 mask = (1 << std20::log2p1(mod)) - 1;
                    const Reg16 scratch = rsi.cvt16();
                    const Reg16 next = rdx.cvt16();
                    if (s < 0x8000) {
                        c.mov(next, address);
                        c.add(next, s);
                        c.and_(next, mask);
                        c.xor_(scratch, scratch);
                        c.cmp(next, (mod + 1) & mask);
                        c.cmove(next, scratch);
                    } else {
                        c.mov(next, address);
                        c.and_(next, mask);
                        c.mov(scratch, mod + 1);
                        c.test(next, next);
                        c.cmovz(next, scratch);
                        c.add(next.cvt32(), s);
                        c.and_(next, mask);
                    }
                    c.and_(address, ~mask);
                    c.or_(address, next);
                }
            }
        } else {
            const s16 s_s = static_cast<s16>(s);
            if (s_s < 0) {
                c.sub(address, -s_s);
            } else {
                c.add(address, s);
            }
        }
    }

    void RnAndModifyNoPreserve(u32 unit, StepValue step, Reg64 out, bool dmod = false) {
        if ((unit == 3 && blk_key.curr.mod1.epi) || (unit == 7 && blk_key.curr.mod1.epj)) {
            if (step != StepValue::Increase2Mode1 && step != StepValue::Decrease2Mode1 &&
                step != StepValue::Increase2Mode2 && step != StepValue::Decrease2Mode2) {
                switch (unit) {
                case 0:
                    c.xor_(R0_1_2_3.cvt16(), R0_1_2_3.cvt16());
                    break;
                case 1:
                    c.mov(rsi, 0xFFFFFFFF0000FFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 2:
                    c.mov(rsi, 0xFFFF0000FFFFFFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 3:
                    c.mov(rsi, 0x0000FFFFFFFFFFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 4:
                    c.xor_(R4_5_6_7.cvt16(), R4_5_6_7.cvt16());
                    break;
                case 5:
                    c.mov(rsi, 0xFFFFFFFF0000FFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                case 6:
                    c.mov(rsi, 0xFFFF0000FFFFFFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                case 7:
                    c.mov(rsi, 0x0000FFFFFFFFFFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                default:
                    UNREACHABLE();
                }
                c.xor_(out, out);
                return;
            }
        }
        switch (unit) {
        case 0:
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            break;
        case 1:
            c.ror(R0_1_2_3, 16);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R0_1_2_3, 16);
            break;
        case 2:
            c.ror(R0_1_2_3, 32);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R0_1_2_3, 32);
            break;
        case 3:
            c.ror(R0_1_2_3, 48);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R0_1_2_3, 48);
            break;
        case 4:
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.movzx(out, R4_5_6_7.cvt16());
            break;
        case 5:
            c.ror(R4_5_6_7, 16);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.movzx(out, R4_5_6_7.cvt16());
            c.rol(R4_5_6_7, 16);
            break;
        case 6:
            c.ror(R4_5_6_7, 32);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R4_5_6_7, 32);
            break;
        case 7:
            c.ror(R4_5_6_7, 48);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.movzx(out, R0_1_2_3.cvt16());
            c.rol(R4_5_6_7, 48);
            break;
        default:
            UNREACHABLE();
        }
    }

    void RnAndModify(u32 unit, StepValue step, Reg64 out, bool dmod = false) {
        switch (unit) {
        case 0:
            c.movzx(out, R0_1_2_3.cvt16());
            break;
        case 1:
            c.rorx(out, R0_1_2_3, 16);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 2:
            c.rorx(out, R0_1_2_3, 32);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 3:
            c.rorx(out, R0_1_2_3, 48);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 4:
            c.movzx(out, R4_5_6_7.cvt16());
            break;
        case 5:
            c.rorx(out, R4_5_6_7, 16);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 6:
            c.rorx(out, R4_5_6_7, 32);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        case 7:
            c.rorx(out, R4_5_6_7, 48);
            c.movzx(out, out.cvt16()); // Needed?
            break;
        default:
            UNREACHABLE();
        }

        if ((unit == 3 && blk_key.curr.mod1.epi) || (unit == 7 && blk_key.curr.mod1.epj)) {
            if (step != StepValue::Increase2Mode1 && step != StepValue::Decrease2Mode1 &&
                step != StepValue::Increase2Mode2 && step != StepValue::Decrease2Mode2) {
                switch (unit) {
                case 0:
                    c.xor_(R0_1_2_3.cvt16(), R0_1_2_3.cvt16());
                    break;
                case 1:
                    c.mov(rsi, 0xFFFFFFFF0000FFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 2:
                    c.mov(rsi, 0xFFFF0000FFFFFFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 3:
                    c.mov(rsi, 0x0000FFFFFFFFFFFFULL);
                    c.and_(R0_1_2_3, rsi);
                    break;
                case 4:
                    c.xor_(R4_5_6_7.cvt16(), R4_5_6_7.cvt16());
                    break;
                case 5:
                    c.mov(rsi, 0xFFFFFFFF0000FFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                case 6:
                    c.mov(rsi, 0xFFFF0000FFFFFFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                case 7:
                    c.mov(rsi, 0x0000FFFFFFFFFFFFULL);
                    c.and_(R4_5_6_7, rsi);
                    break;
                default:
                    UNREACHABLE();
                }
                return;
            }
        }
        switch (unit) {
        case 0:
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            break;
        case 1:
            c.ror(R0_1_2_3, 16);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.rol(R0_1_2_3, 16);
            break;
        case 2:
            c.ror(R0_1_2_3, 32);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.rol(R0_1_2_3, 32);
            break;
        case 3:
            c.ror(R0_1_2_3, 48);
            StepAddress(unit, R0_1_2_3.cvt16(), step, dmod);
            c.rol(R0_1_2_3, 48);
            break;
        case 4:
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            break;
        case 5:
            c.rorx(R4_5_6_7, R4_5_6_7, 16);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.rorx(R4_5_6_7, R4_5_6_7, 48);
            break;
        case 6:
            c.ror(R4_5_6_7, 32);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.rol(R4_5_6_7, 32);
            break;
        case 7:
            c.ror(R4_5_6_7, 48);
            StepAddress(unit, R4_5_6_7.cvt16(), step, dmod);
            c.rol(R4_5_6_7, 48);
            break;
        default:
            UNREACHABLE();
        }
    }

    void RnAddressAndModify(u32 unit, StepValue step, Reg64 out, bool dmod = false) {
        RnAndModify(unit, step, out, dmod);
        RnAddress(unit, out);
    }

    template <typename ArRnX>
    u16 GetArRnUnit(ArRnX arrn) const {
        static_assert(std::is_same_v<ArRnX, ArRn1> || std::is_same_v<ArRnX, ArRn2>);
        switch (arrn.Index()) {
        case 0:
            return blk_key.curr.ar[0].arrn0.Value();
        case 1:
            return blk_key.curr.ar[0].arrn1.Value();
        case 2:
            return blk_key.curr.ar[1].arrn0.Value();
        case 3:
            return blk_key.curr.ar[1].arrn1.Value();
        default:
            UNREACHABLE();
        }
    }

    static StepValue ConvertArStep(u16 arvalue) {
        switch (arvalue) {
        case 0:
            return StepValue::Zero;
        case 1:
            return StepValue::Increase;
        case 2:
            return StepValue::Decrease;
        case 3:
            return StepValue::PlusStep;
        case 4:
            return StepValue::Increase2Mode1;
        case 5:
            return StepValue::Decrease2Mode1;
        case 6:
            return StepValue::Increase2Mode2;
        case 7:
            return StepValue::Decrease2Mode2;
        default:
            UNREACHABLE();
        }
    }

    template <typename ArStepX>
    StepValue GetArStep(ArStepX arstep) const {
        static_assert(std::is_same_v<ArStepX, ArStep1> || std::is_same_v<ArStepX, ArStep1Alt> ||
                      std::is_same_v<ArStepX, ArStep2>);
        const u16 value = [&] {
            switch (arstep.Index()) {
            case 0:
                return blk_key.curr.ar[0].arstep0.Value();
            case 1:
                return blk_key.curr.ar[0].arstep1.Value();
            case 2:
                return blk_key.curr.ar[1].arstep0.Value();
            case 3:
                return blk_key.curr.ar[1].arstep1.Value();
            default:
                UNREACHABLE();
            }
        }();
        return ConvertArStep(value);
    }

    template <typename ArpStepX>
    std::tuple<OffsetValue, OffsetValue> GetArpOffset(ArpStepX arpstepi, ArpStepX arpstepj) const {
        static_assert(std::is_same_v<ArpStepX, ArpStep1> || std::is_same_v<ArpStepX, ArpStep2>);
        return std::make_tuple((OffsetValue)blk_key.curr.arp[arpstepi.Index()].arpoffseti.Value(),
                               (OffsetValue)blk_key.curr.arp[arpstepj.Index()].arpoffsetj.Value());
    }

    template <typename ArpRnX>
    std::tuple<u16, u16> GetArpRnUnit(ArpRnX arprn) const {
        static_assert(std::is_same_v<ArpRnX, ArpRn1> || std::is_same_v<ArpRnX, ArpRn2>);
        return std::make_tuple(blk_key.curr.arp[arprn.Index()].arprni, blk_key.curr.arp[arprn.Index()].arprnj + 4);
    }

    template <typename ArpStepX>
    std::tuple<StepValue, StepValue> GetArpStep(ArpStepX arpstepi, ArpStepX arpstepj) const {
        static_assert(std::is_same_v<ArpStepX, ArpStep1> || std::is_same_v<ArpStepX, ArpStep2>);
        return std::make_tuple(ConvertArStep(blk_key.curr.arp[arpstepi.Index()].arpstepi),
                               ConvertArStep(blk_key.curr.arp[arpstepj.Index()].arpstepj));
    }

    template <typename ArStepX>
    OffsetValue GetArOffset(ArStepX arstep) const {
        static_assert(std::is_same_v<ArStepX, ArStep1> || std::is_same_v<ArStepX, ArStep2>);
        const u16 value = [&] {
            switch (arstep.Index()) {
            case 0:
                return blk_key.curr.ar[0].aroffset0.Value();
            case 1:
                return blk_key.curr.ar[0].aroffset1.Value();
            case 2:
                return blk_key.curr.ar[1].aroffset0.Value();
            case 3:
                return blk_key.curr.ar[1].aroffset1.Value();
            default:
                UNREACHABLE();
            }
        }();
        return static_cast<OffsetValue>(value);
    }

    void ProductFromBus32(Px reg, Reg32 value) {
        u16 unit = reg.Index();
        c.mov(dword[REGS + offsetof(JitRegisters, p) + unit * sizeof(u32)], value);
        c.bt(value, 31);
        c.setc(word[REGS + offsetof(JitRegisters, pe) + unit * sizeof(u16)]);
    }

    static RegName CounterAcc(RegName in) {
        static std::unordered_map<RegName, RegName> map{
                                                        {RegName::a0, RegName::a1},   {RegName::a1, RegName::a0},
                                                        {RegName::b0, RegName::b1},   {RegName::b1, RegName::b0},
                                                        {RegName::a0l, RegName::a1l}, {RegName::a1l, RegName::a0l},
                                                        {RegName::b0l, RegName::b1l}, {RegName::b1l, RegName::b0l},
                                                        {RegName::a0h, RegName::a1h}, {RegName::a1h, RegName::a0h},
                                                        {RegName::b0h, RegName::b1h}, {RegName::b1h, RegName::b0h},
                                                        {RegName::a0e, RegName::a1e}, {RegName::a1e, RegName::a0e},
                                                        {RegName::b0e, RegName::b1e}, {RegName::b1e, RegName::b0e},
                                                        };
        return map.at(in);
    }
};

} // namespace Teakra
