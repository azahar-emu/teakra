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

    // Fires interrupt 0 seven cycles in.
    timer[0].SetInterruptHandler([&interpreter]() { interpreter.SignalInterrupt(0); });
    timer[0].count_mode = Teakra::Timer::CountMode::Single;
    timer[0].start_low = 7;
    timer[0].start_high = 0;
    timer[0].Restart();

    SECTION("Simple counting") {
        memory_interface.ProgramWrite(0x0001, 0x1000);
        interpreter.Run(1 + 5);
        REQUIRE(regs.a[0] == 5);
        REQUIRE(regs.a[1] == 0);
    }

    SECTION("Counting with interrupt") {
        memory_interface.ProgramWrite(0x0001, 0x1000);
        interpreter.Run(1 + 7 + 1 + 3);
        REQUIRE(regs.a[0] == 7);
        REQUIRE(regs.a[1] == 3);
    }

    SECTION("Counting with idle and interrupt") {
        memory_interface.ProgramWrite(0x0001, 0x100D);
        interpreter.Run(1 + 6);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 0);

        interpreter.Run(3);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 1);
    }

    SECTION("Counting with idle and interrupt 2") {
        memory_interface.ProgramWrite(0x0001, 0x100D);
        interpreter.Run(1 + 7);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 0);

        interpreter.Run(3);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 2);
    }

    SECTION("Counting with idle and interrupt 3") {
        memory_interface.ProgramWrite(0x0001, 0x100D);
        interpreter.Run(1 + 7 + 1 + 2);
        REQUIRE(regs.a[0] == 3);
        REQUIRE(regs.a[1] == 2);
    }

    SECTION("Idle with interrupt 1") {
        memory_interface.ProgramWrite(0x0001, 0x1010);
        interpreter.Run(1 + 7 + 1 + 2);
        REQUIRE(regs.a[0] == 0);
        REQUIRE(regs.a[1] == 2);
    }

    SECTION("Idle with interrupt 2") {
        memory_interface.ProgramWrite(0x0001, 0x1010);
        interpreter.Run(1 + 7);
        REQUIRE(regs.a[0] == 0);
        REQUIRE(regs.a[1] == 0);

        interpreter.Run(1 + 2);
        REQUIRE(regs.a[0] == 0);
        REQUIRE(regs.a[1] == 2);
    }

    SECTION("Idle with interrupt 3") {
        memory_interface.ProgramWrite(0x0001, 0x1010);
        interpreter.Run(1 + 6);
        REQUIRE(regs.a[0] == 0);
        REQUIRE(regs.a[1] == 0);

        SECTION("Single step on the interrupt") {
            interpreter.Run(1);
            REQUIRE(regs.a[0] == 0);
            REQUIRE(regs.a[1] == 0);

            interpreter.Run(1 + 2);
            REQUIRE(regs.a[0] == 0);
            REQUIRE(regs.a[1] == 2);
        }

        SECTION("Run over the interrupt") {
            interpreter.Run(1 + 1 + 2);
            REQUIRE(regs.a[0] == 0);
            REQUIRE(regs.a[1] == 2);
        }
    }
}
