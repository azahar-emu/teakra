#include "interpreter.h"
#include "processor.h"
#include "register.h"
#ifdef TEAKRA_ENABLE_JIT
#include <xbyak/xbyak_util.h>
#include "jit_x64.h"
#endif

namespace Teakra {

struct Processor::Impl {
    Impl(CoreTiming& core_timing, MemoryInterface& memory_interface, bool use_jit)
        : core_timing(core_timing), interpreter(core_timing, regs, memory_interface) {
#ifdef TEAKRA_ENABLE_JIT
        if (use_jit && IsJitSupported()) {
            jit = std::make_unique<EmitX64>(core_timing, memory_interface, interpreter, regs);
        }
#endif
    }
    CoreTiming& core_timing;
    RegisterState regs;
    Interpreter interpreter;
#ifdef TEAKRA_ENABLE_JIT
    std::unique_ptr<EmitX64> jit;
#endif
};

Processor::Processor(CoreTiming& core_timing, MemoryInterface& memory_interface, bool use_jit)
    : impl(new Impl(core_timing, memory_interface, use_jit)) {}
Processor::~Processor() = default;

void Processor::Reset() {
    impl->regs = RegisterState();
    impl->interpreter.ClearOverrun();
    impl->interpreter.ResetInterruptRequests();
#ifdef TEAKRA_ENABLE_JIT
    if (impl->jit) {
        impl->jit->Reset();
    }
#endif
}

void Processor::Run(unsigned cycles) {
#ifdef TEAKRA_ENABLE_JIT
    if (impl->jit) {
        impl->jit->Run(cycles);
        return;
    }
#endif
    impl->interpreter.Run(cycles);
}

void Processor::SignalInterrupt(u32 i, s32 source_delay) {
#ifdef TEAKRA_ENABLE_JIT
    if (impl->jit) {
        impl->jit->SignalInterrupt(i, source_delay);
        return;
    }
#endif
    impl->interpreter.SignalInterrupt(i, source_delay);
}
void Processor::SignalVectoredInterrupt(u32 address, bool context_switch, s32 source_delay) {
#ifdef TEAKRA_ENABLE_JIT
    if (impl->jit) {
        impl->jit->SignalVectoredInterrupt(address, context_switch, source_delay);
        return;
    }
#endif
    impl->interpreter.SignalVectoredInterrupt(address, context_switch, source_delay);
}

Teakra::RegisterState& Processor::GetRegisterState() {
#ifdef TEAKRA_ENABLE_JIT
    if (impl->jit) {
        // The JIT keeps its own register layout. Its state is copied here and loaded back before
        // the next Run, so changes made through the reference are kept.
        impl->jit->SaveState();
    }
#endif
    return impl->regs;
}

const Teakra::RegisterState& Processor::GetRegisterState() const {
#ifdef TEAKRA_ENABLE_JIT
    if (impl->jit) {
        impl->jit->SaveState();
    }
#endif
    return impl->regs;
}

bool Processor::IsJitSupported() {
#ifdef TEAKRA_ENABLE_JIT
    static const bool supported = [] {
        const Xbyak::util::Cpu cpu;
        return cpu.has(Xbyak::util::Cpu::tBMI2) && cpu.has(Xbyak::util::Cpu::tLZCNT);
    }();
    return supported;
#else
    return false;
#endif
}

} // namespace Teakra
