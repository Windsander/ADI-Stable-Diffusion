/*
 * Copyright (c) 2018-2050 SD_Clip - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_CLIP_H
#define MODEL_CLIP_H

#include "model_base.cc"
#include "tokenizer_register.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace tokenizer;
using namespace Ort;
using namespace detail;

typedef struct ModelUNetConfig {
    TokenizerConfig sd_tokenizer_config;
    TokenizerType sd_tokenizer_type;
    float sd_scale_decode_strength = 0.18215f;
    int32_t blank_token_size = 49408;           // blank token generated for unconditional input
    int32_t model_max_length = 77;              // max token length
    int32_t model_hidden_dim = 768;             // out token length
} ModelUNetConfig ;

class Clip : public ModelBase {
private:
    typedef std::vector<std::pair<std::string, float>> PromptWeight_map;

private:
    ModelUNetConfig sd_clip_config;
    TokenizerEntity_ptr sd_tokenizer_p;

public:
    explicit Clip(const std::string &model_path_,  const ModelUNetConfig &clip_config_ = {});
    ~Clip() override;

    Tensor embedding(const std::string& prompts_);
};

Clip::Clip(const std::string &model_path_, const ModelUNetConfig &clip_config_) : ModelBase(model_path_){
    sd_clip_config = clip_config_;
    sd_tokenizer_p = TokenizerRegister::request_tokenizer(
        clip_config_.sd_tokenizer_type,
        clip_config_.sd_tokenizer_config
    );
    sd_tokenizer_p->init();
}

Clip::~Clip(){
    sd_tokenizer_p->uninit();
    sd_tokenizer_p = TokenizerRegister::recycle_tokenizer(sd_tokenizer_p);
    sd_clip_config.~ModelUNetConfig();
}

Tensor Clip::embedding(const std::string& prompts_) {

    PairedTokenWeight tokenizer_output_ = sd_tokenizer_p->tokenize(prompts_);

    std::vector<Tensor> merged_hidden_;
    for (auto &tw_pair_: tokenizer_output_) {
        const Tensor &tokens_ = tw_pair_.first;        // [1, 77]
        const Tensor &weight_ = tw_pair_.second;       // [1, 77]

        std::vector<Tensor> input_tensors;
        Ort::AllocatorWithDefaultOptions ort_alloc;
        input_tensors.push_back(tokens_.GetValue(0, ort_alloc));
        std::vector<Tensor> output_tensors;           // [77, 768]
        execute(input_tensors, output_tensors);

        merged_hidden_.push_back(
            TensorHelper::weight(output_tensors[0], weight_, 1, true)  // [1, 77, 768]
        );
    }
    Tensor hidden_state_ = TensorHelper::merge(merged_hidden_);  // [N, 77, 768]

    return hidden_state_;
}

} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_CLIP_H

