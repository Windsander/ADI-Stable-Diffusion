/*
 * Copyright (c) 2018-2050 SD_Clip - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_CLIP_H
#define MODEL_CLIP_H

#include <utility>

#include "model_base.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace scheduler;
using namespace Ort;
using namespace detail;

typedef struct TokenizerConfig {
    float sd_scale_decode_strength = 0.18215f;
    int32_t blank_token_size = 49408;           // blank token generated for unconditional input
    int32_t model_max_length = 77;              // max token length
    int32_t model_hidden_dim = 768;             // out token length

} TokenizerConfig;

class Tokenizer : public ModelBase {
private:
    TokenizerConfig sd_tokenizer_config;

public:
    explicit Tokenizer(std::string model_path_, TokenizerConfig vae_config_ = {});
    ~Tokenizer() override;

    Tensor generate(std::string positive_prompts_, std::string negative_prompts_);

private:
    Tensor tokenize(std::string prompts_);
};

Tokenizer::Tokenizer(std::string model_path_, TokenizerConfig vae_config_) : ModelBase(std::move(model_path_)){
    sd_tokenizer_config = vae_config_;
}

Tokenizer::~Tokenizer(){
    sd_tokenizer_config.~TokenizerConfig();
}

Tensor Tokenizer::tokenize(std::string prompts_) {

    // tokenizer: text tokenized & encoding
    vector<int32_t> timestep_value_{prompts_};
    TensorShape timestep_shape_{1};
    return TensorHelper::create(timestep_shape_, timestep_value_);


    std::vector<Tensor> input_tensors;
    input_tensors.push_back(TensorHelper::multiple(latents_, (1.0f / sd_vae_config.sd_scale_decode_strength)));
    std::vector<Tensor> output_tensors;
    execute(input_tensors, output_tensors);
    std::vector<Tensor> model_result_ ;

    Ort::AllocatorWithDefaultOptions ort_alloc;
    Tensor result_ = output_tensors.front().GetValue(0, ort_alloc);
    return result_;
}


} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_CLIP_H

