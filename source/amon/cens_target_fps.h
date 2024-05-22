// Copyright (c) 2018-2050 CoreContext - Arikan.Li

#include <utility>

#include "cens_statistics_base.h"

namespace onnx {
namespace sd {
namespace amon {
using namespace std;

#if defined(OS_IOS)
#define LOCAL_ATOMIC_ALIGNMENT alignas(16)
#else
#define LOCAL_ATOMIC_ALIGNMENT
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <Windows.h>

static int64_t timer_freq, timer_start;

void timing_init(void) {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
}

int64_t timing_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart - timer_start) * 1000) / timer_freq;
}

int64_t timing_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart - timer_start) * 1000000) / timer_freq;
}
#else
void timing_init() {}

int64_t timing_ms() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000 + (int64_t) ts.tv_nsec / 1000000;
}

int64_t timing_us() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000 + (int64_t) ts.tv_nsec / 1000;
}
#endif

#define NOTIFIER_DURATION_THRESHOLD_MSEC 60000000      // 1 min

typedef struct timing_notifier_s {
    std::string from_source;
    LOCAL_ATOMIC_ALIGNMENT uint64_t timing_last_cost;
    LOCAL_ATOMIC_ALIGNMENT uint64_t timing_average_dur;
    LOCAL_ATOMIC_ALIGNMENT float timing_average_fps;
} timing_notifier_t;

class target_fps_statistics : public statistics_base {
private:
    timing_notifier_t timing_cost_meta = {"", 0, 0, 0.0f};
    LOCAL_ATOMIC_ALIGNMENT uint64_t timing_last_at = 0;
    LOCAL_ATOMIC_ALIGNMENT uint64_t timing_start_at = 0;
    LOCAL_ATOMIC_ALIGNMENT uint64_t timing_end_when = 0;
    LOCAL_ATOMIC_ALIGNMENT uint64_t timing_maintain = 0;
    LOCAL_ATOMIC_ALIGNMENT uint64_t timing_duration = 0;
    LOCAL_ATOMIC_ALIGNMENT uint64_t notify_duration = 0;

private:
    void do_statistics(StatisticsLevel level_, const char *msg_) override {
        if (notify_duration > NOTIFIER_DURATION_THRESHOLD_MSEC) {
            notify_duration = 0;
            auto msg_average_fps = float(timing_cost_meta.timing_average_fps);
            long msg_average_dur = long(timing_cost_meta.timing_average_dur);
            sd_log(((loglevel_e) level_)) << "ORT Statistics: "
                                          << "target " << timing_cost_meta.from_source.c_str()
                                          << "avg_fps " << msg_average_fps
                                          << "avg_dur " << msg_average_dur << " us"
                                          << " with " << msg_;
        }
    }

public:
    void mark_start_at(std::string target_) {
        timing_cost_meta.from_source = std::move(target_);
        timing_last_at = timing_start_at;
        timing_start_at = timing_us();

        timing_maintain = timing_start_at - timing_last_at;
        notify_duration = notify_duration + timing_maintain;

        timing_cost_meta.timing_average_fps =
            (timing_maintain > 0)
            ? float(0.5 * timing_cost_meta.timing_average_fps + 0.5 * (1 * 1000000) / float(timing_maintain))
            : timing_cost_meta.timing_average_fps;
    }

    void mark_end_when() {
        if (timing_start_at == 0) {
            return;
        }

        timing_end_when = timing_us();
        timing_duration = timing_end_when - timing_start_at;

        timing_cost_meta.timing_last_cost = timing_duration;
        timing_cost_meta.timing_average_dur =
            (timing_duration > 0) ?
            uint64_t(0.9f * float(timing_cost_meta.timing_average_dur) + 0.1f * float(timing_duration)) :
            timing_cost_meta.timing_average_dur;
    }

public:
    explicit target_fps_statistics(StatisticsLevel level_ = CENS_LOG_INFO, const char *message_ = "") : statistics_base(
        level_, message_) {
        // no-action
    }

    ~target_fps_statistics() = default;
};

} // namespace amon
} // namespace sd
} // namespace onnx
