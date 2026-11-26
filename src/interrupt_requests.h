#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include "common_types.h"
#include "cycles.h"

namespace Teakra {

// Handles interrupt requests on their way from the ICU to the core. A request sets ip or ipv
// Cycles::InterruptRequest cycles after it is signalled, plus a delay given by its source.
class InterruptRequests {
public:
    static constexpr u32 Vectored = 3;

    void Signal(u32 line, u32 source_delay) {
        pending[line] = Cycles::InterruptRequest + source_delay + 1;
    }

    void Collect() {
        for (u32 line = 0; line < countdown.size(); ++line) {
            if (const u32 value = pending[line].exchange(0)) {
                const s64 delay = static_cast<s64>(value) - 1;
                countdown[line] = countdown[line] < 0 ? delay : std::min(countdown[line], delay);
            }
        }
    }

    // Advances the counted requests by the given cycles. Returns a bit for each line whose request
    // reached the core.
    u32 Advance(u64 cycles) {
        u32 arrived = 0;
        for (u32 line = 0; line < countdown.size(); ++line) {
            if (countdown[line] < 0) {
                continue;
            }
            countdown[line] -= static_cast<s64>(cycles);
            if (countdown[line] <= 0) {
                countdown[line] = -1;
                arrived |= 1u << line;
            }
        }
        return arrived;
    }

    bool Any() const {
        return std::any_of(countdown.begin(), countdown.end(), [](s64 c) { return c >= 0; }) ||
               std::any_of(pending.begin(), pending.end(), [](const auto& p) { return p != 0; });
    }

    u64 NextArrival() const {
        s64 next = std::numeric_limits<s64>::max();
        for (const s64 c : countdown) {
            if (c >= 0) {
                next = std::min(next, std::max<s64>(c, 1));
            }
        }
        return static_cast<u64>(next);
    }

    void Reset() {
        for (auto& p : pending) {
            p = 0;
        }
        countdown.fill(-1);
    }

private:
    std::array<std::atomic<u32>, 4> pending{};
    std::array<s64, 4> countdown{{-1, -1, -1, -1}};
};

} // namespace Teakra
