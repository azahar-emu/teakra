#pragma once

#include <memory>
#include "common_types.h"
#include "core_timing.h"

namespace Teakra {

class MemoryInterface;
struct RegisterState;

class Processor {
public:
    Processor(CoreTiming& core_timing, MemoryInterface& memory_interface, bool use_jit);
    ~Processor();
    void Reset();
    void Run(unsigned cycles);
    void SignalInterrupt(u32 i, u32 source_delay = 0);
    void SignalVectoredInterrupt(u32 address, bool context_switch, u32 source_delay = 0);

    RegisterState& GetRegisterState();
    const RegisterState& GetRegisterState() const;

    static bool IsJitSupported();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Teakra
