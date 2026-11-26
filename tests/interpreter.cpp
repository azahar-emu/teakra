#include <memory>
#include <catch.hpp>
#include "../src/core_timing.h"
#include "../src/interpreter.h"
#include "../src/memory_interface.h"
#include "../src/shared_memory.h"

TEST_CASE("Cycle accuracy", "[interpreter]") {
    std::array<Teakra::Timer, 2> timer{};
    std::array<Teakra::Btdmp, 2> btdmp{};
    Teakra::CoreTiming core_timing{timer, btdmp};
    Teakra::SharedMemory shared_memory;
    Teakra::MemoryInterfaceUnit miu;
    Teakra::MemoryInterface memory_interface{shared_memory, miu};
    Teakra::RegisterState regs;
    Teakra::Interpreter interpreter(core_timing, regs, memory_interface);

    regs.pc = 0;
    regs.a[0] = 0;
    regs.a[1] = 0;
    regs.ie = 1;
    regs.im[0] = 1;

    memory_interface.ProgramWrite(0x0000, 0x4180); // br (RESET vector)

    memory_interface.ProgramWrite(0x0006, 0x4180); // br (INT0 vector)
    memory_interface.ProgramWrite(0x0007, 0x2000); // br 0x2000

    for (u32 i = 0; i < 16; ++i) {
        memory_interface.ProgramWrite(0x1000 + i, 0x67D0); // inc a0
        memory_interface.ProgramWrite(0x2000 + i, 0x77D0); // inc a1
    }

    memory_interface.ProgramWrite(0x1010, 0x57F0); // brr -1

    // Fires interrupt 0 nine cycles in, which is the reset branch (3 cycles) and six more.
    timer[0].SetInterruptHandler([&interpreter](u64) { interpreter.SignalInterrupt(0); });
    timer[0].count_mode = Teakra::Timer::CountMode::Single;
    timer[0].start_low = 9;
    timer[0].start_high = 0;
    timer[0].Restart();

    // Cycles: a taken br or brr costs 3, inc 1. The timer reaches 0 at cycle 9, at the end of an
    // instruction. The test wires the timer to the core directly, so the request reaches the core
    // 6 cycles later (InterruptRequest), at cycle 15. It is taken after the instruction during
    // which it arrives, for 1 cycle (InterruptEntry). Cycles that a Run goes over are taken from
    // the next Run.
    SECTION("Simple counting") {
        memory_interface.ProgramWrite(0x0001, 0x1000);
        interpreter.Run(3 + 5);
        REQUIRE(regs.a[0] == 5);
        REQUIRE(regs.a[1] == 0);
    }

    SECTION("Counting with interrupt") {
        memory_interface.ProgramWrite(0x0001, 0x1000);
        interpreter.Run(3 + 12 + 1 + 3 + 3);
        REQUIRE(regs.a[0] == 12);
        REQUIRE(regs.a[1] == 3);
    }

    SECTION("Counting with idle and interrupt") {
        memory_interface.ProgramWrite(0x0001, 0x100D);
        interpreter.Run(3 + 3 + 3);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 0);

        interpreter.Run(3 + 3 + 1 + 3 + 1);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 1);
    }

    SECTION("Counting with idle and interrupt 2") {
        memory_interface.ProgramWrite(0x0001, 0x100D);
        interpreter.Run(3 + 7);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 0);

        // The last Run went 2 cycles over, to cycle 12.
        interpreter.Run(2 + 3 + 1 + 3 + 2);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 2);
    }

    SECTION("Counting with idle and interrupt 3") {
        memory_interface.ProgramWrite(0x0001, 0x100D);
        interpreter.Run(3 + 3 + 9 + 1 + 3 + 2);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 2);
    }

    SECTION("Idle with interrupt 1") {
        memory_interface.ProgramWrite(0x0001, 0x1010);
        interpreter.Run(3 + 12 + 1 + 3 + 2);
        REQUIRE(regs.a[0] == 0);
        REQUIRE(regs.a[1] == 2);
    }

    SECTION("Idle with interrupt 2") {
        memory_interface.ProgramWrite(0x0001, 0x1010);
        interpreter.Run(3 + 7);
        REQUIRE(regs.a[0] == 0);
        REQUIRE(regs.a[1] == 0);

        // The last Run went 2 cycles over, to cycle 12.
        interpreter.Run(2 + 3 + 1 + 3 + 2);
        REQUIRE(regs.a[0] == 0);
        REQUIRE(regs.a[1] == 2);
    }

    // The idle skip stops a cycle before the timer reaches 0, and the brr that follows ends at
    // cycle 11, 2 cycles over. The request then arrives at cycle 17.
    SECTION("Idle with interrupt 3") {
        memory_interface.ProgramWrite(0x0001, 0x1010);
        interpreter.Run(3 + 6);
        REQUIRE(regs.a[0] == 0);
        REQUIRE(regs.a[1] == 0);

        SECTION("Single step on the interrupt") {
            interpreter.Run(1);
            REQUIRE(regs.a[0] == 0);
            REQUIRE(regs.a[1] == 0);

            interpreter.Run(1 + 3 + 3 + 1 + 3 + 2);
            REQUIRE(regs.a[0] == 0);
            REQUIRE(regs.a[1] == 2);
        }

        SECTION("Run over the interrupt") {
            interpreter.Run(2 + 3 + 3 + 1 + 3 + 2);
            REQUIRE(regs.a[0] == 0);
            REQUIRE(regs.a[1] == 2);
        }
    }
}
