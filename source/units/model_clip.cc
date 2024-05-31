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

#define DEFAULT_CLIP_CONFIG                                          \
    {                                                                \
        /*sd_tokenizer_config*/ DEFAULT_TOKENIZER_CONFIG             \
    }                                                                \

typedef struct ModelClipConfig {
    TokenizerConfig sd_tokenizer_config;
} ModelClipConfig ;

class Clip : public ModelBase {
private:
    ModelClipConfig sd_clip_config;
    TokenizerEntity_ptr sd_tokenizer_p;

protected:
    void generate_output(std::vector<Tensor>& output_tensors_) override;
    Tensor tokenizing(const std::string& prompts_);

public:
    explicit Clip(const std::string &model_path_,  const ModelClipConfig &clip_config_ = DEFAULT_CLIP_CONFIG);
    ~Clip() override;

    Tensor embedding(const std::string &positive_prompts_, const std::string &negative_prompts_);
};

Clip::Clip(const std::string &model_path_, const ModelClipConfig &clip_config_) : ModelBase(model_path_){
    sd_clip_config = clip_config_;
    sd_tokenizer_p = TokenizerRegister::request_tokenizer(clip_config_.sd_tokenizer_config);
    sd_tokenizer_p->init();
}

Clip::~Clip(){
    sd_tokenizer_p->uninit();
    sd_tokenizer_p = TokenizerRegister::recycle_tokenizer(sd_tokenizer_p);
    sd_clip_config.~ModelClipConfig();
}

void Clip::generate_output(std::vector<Tensor> &output_tensors_) {
    {
        std::vector<float> output_hidden_(
            sd_clip_config.sd_tokenizer_config.avail_token_size *
            sd_clip_config.sd_tokenizer_config.major_hidden_dim
        );
        TensorShape hidden_shape_ = {
            1,
            sd_clip_config.sd_tokenizer_config.avail_token_size,
            sd_clip_config.sd_tokenizer_config.major_hidden_dim
        };
        output_tensors_.emplace_back(TensorHelper::create(hidden_shape_, output_hidden_));
    }
    {
        std::vector<float> output_pooler_(
            sd_clip_config.sd_tokenizer_config.avail_token_size *
            sd_clip_config.sd_tokenizer_config.major_hidden_dim
        );
        TensorShape pooler_shape_ = {
            1,
            sd_clip_config.sd_tokenizer_config.major_hidden_dim
        };
        output_tensors_.emplace_back(TensorHelper::create(pooler_shape_, output_pooler_));
    }
}

Tensor Clip::tokenizing(const std::string& prompts_) {
    if (prompts_.empty()) return TensorHelper::create(TensorShape{0}, std::vector<float>{});

    PairedTokenWeight tokenizer_output_ = sd_tokenizer_p->tokenize(prompts_);

    std::vector<Tensor> merged_hidden_;
    for (auto &tw_pair_: tokenizer_output_) {           // major_hidden_dim = 768 in SD, 1280 in SDXL
        Tensor &tokens_ = tw_pair_.first;               // [1, 77]
        Tensor &weight_ = tw_pair_.second;              // [1, 77]

        std::vector<Tensor> input_tensors;
        input_tensors.emplace_back(std::move(tokens_)); // [vocab_size, major_hidden_dim]
        std::vector<Tensor> output_tensors;             // [1, 77, major_hidden_dim]
        generate_output(output_tensors);
        execute(input_tensors, output_tensors);

        merged_hidden_.push_back(                       // [1, 77, major_hidden_dim]
            TensorHelper::weight(output_tensors[0], weight_, 1, true)
        );
    }
    // seems not right
    Tensor hidden_state_ = TensorHelper::merge(merged_hidden_, 1);  // [1, 77 * N, major_hidden_dim]

    return hidden_state_;
}

Tensor Clip::embedding(const std::string &positive_prompts_, const std::string &negative_prompts_) {
    Tensor positive_ = tokenizing(positive_prompts_);           // [1, 77 * N_p, major_hidden_dim]
    Tensor negative_ = tokenizing(negative_prompts_);           // [1, 77 * N_n, major_hidden_dim]
    return  sd_tokenizer_p->embedding(positive_, negative_);    // [2, 77 * max(N_p, N_n), major_hidden_dim]
}

} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_CLIP_H

