/*
 * BasicTools
 * Definition: put simple tools we used in here
 * Created by Arikan.Li on 2022/03/11.
 */
#include "onnxsd_defs.h"

namespace onnx {
namespace sd {
namespace base {
using namespace amon;

class RandomGenerator {
protected:
    std::default_random_engine random_generator;
    std::normal_distribution<float> random_style;

public:
    explicit RandomGenerator(float mean_ = 0.0f, float stddev_ = 1.0f) {
        random_style = std::normal_distribution<float>(mean_, stddev_);
    }

    ~RandomGenerator() {
        random_style.reset();
    }

    void seed(uint64_t seed_) {
        if (seed_ == 0) return;
        this->random_generator.seed(seed_);
    }

    float random_at(float mark_) {
        SD_UNUSED(mark_);
        return random_style(random_generator);
    }
};

} // namespace base
} // namespace sd
} // namespace onnx
