#include <algorithm>
#include <limits>
#include "crash.h"
#include "timer.h"

namespace Teakra {

void Timer::Reset() {
    update_mmio = 0;
    pause = 0;
    count_mode = CountMode::Single;
    scale = 0;

    start_high = 0;
    start_low = 0;
    counter = 0;
    counter_high = 0;
    counter_low = 0;
}

void Timer::Restart() {
    ASSERT(static_cast<u16>(count_mode) < 4);
    if (count_mode != CountMode::FreeRunning) {
        counter = ((u32)start_high << 16) | start_low;
        UpdateMMIO();
    }
}

void Timer::Tick(u64 ticks) {
    ASSERT(static_cast<u16>(count_mode) < 4);
    ASSERT(scale == 0);
    if (pause)
        return;
    if (count_mode == CountMode::EventCount)
        return;
    while (ticks > 0) {
        if (counter == 0) {
            if (count_mode == CountMode::AutoRestart) {
                counter = ((u32)start_high << 16) | start_low;
            } else if (count_mode == CountMode::FreeRunning) {
                counter = 0xFFFFFFFF;
            } else {
                break;
            }
            --ticks;
            continue;
        }
        const u64 step = std::min<u64>(ticks, counter);
        counter -= static_cast<u32>(step);
        ticks -= step;
        if (counter == 0) {
            interrupt_handler(ticks);
        }
    }
    UpdateMMIO();
}

void Timer::TickEvent() {
    if (pause)
        return;
    if (count_mode != CountMode::EventCount)
        return;
    if (counter == 0)
        return;
    --counter;
    UpdateMMIO();
    if (counter == 0)
        interrupt_handler(0);
}

void Timer::UpdateMMIO() {
    if (!update_mmio)
        return;
    counter_high = counter >> 16;
    counter_low = counter & 0xFFFF;
}

u64 Timer::GetMaxSkip() const {
    if (pause || count_mode == CountMode::EventCount)
        return std::numeric_limits<u64>::max();

    if (counter == 0) {
        if (count_mode == CountMode::AutoRestart) {
            return ((u32)start_high << 16) | start_low;
        } else if (count_mode == CountMode::FreeRunning) {
            return std::numeric_limits<u32>::max();
        } else /*Single*/ {
            return std::numeric_limits<u64>::max();
        }
    }

    return counter - 1;
}

void Timer::Skip(u64 ticks) {
    if (pause || count_mode == CountMode::EventCount)
        return;

    if (counter == 0) {
        u32 reset;
        if (count_mode == CountMode::AutoRestart) {
            reset = ((u32)start_high << 16) | start_low;
        } else if (count_mode == CountMode::FreeRunning) {
            reset = std::numeric_limits<u32>::max();
        } else {
            return;
        }
        ASSERT(reset >= ticks);
        counter = reset - ((u32)ticks - 1);
    } else {
        ASSERT(counter > ticks);
        counter -= (u32)ticks;
    }

    UpdateMMIO();
}

} // namespace Teakra
