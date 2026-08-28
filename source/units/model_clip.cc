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
        /*sd_tokenizer_config*/ DEFAULT_TOKENIZER_CONFIG,            \
        /*use_penultimate*/     false,                               \
    }

typedef struct ModelClipConfig {
    TokenizerConfig sd_tokenizer_config;
    // SDXL-style encoders condition on the penultimate hidden state
    // (diffusers hidden_states[-2]); legacy SD v1.x/v2.x use last_hidden_state
    bool use_penultimate;
} ModelClipConfig ;

// hidden: [1, 77 * N, hidden_dim] prompt-weighted sequence embedding;
// pooled: [1, projection_dim] when the encoder provides one (SDXL text_encoder_2), empty otherwise
typedef struct ClipEmbedResult {
    Tensor hidden = TensorHelper::create(TensorShape{0}, std::vector<float>{});
    Tensor pooled = TensorHelper::create(TensorShape{0}, std::vector<float>{});
} ClipEmbedResult;

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

    ClipEmbedResult embedding(const std::string& prompts_);
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
            size_t(sd_clip_config.sd_tokenizer_config.avail_token_size) *
            size_t(sd_clip_config.sd_tokenizer_config.major_hidden_dim)
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
            sd_clip_config.sd_tokenizer_config.major_hidden_dim
        );
        TensorShape pooler_shape_ = {
            1,
            sd_clip_config.sd_tokenizer_config.major_hidden_dim
        };
        output_tensors_.emplace_back(TensorHelper::create(pooler_shape_, output_pooler_));
    }
}

ClipEmbedResult Clip::embedding(const std::string& prompts_) {
    // tokenize prompts
    PairedTokenWeight tokenizer_output_ = sd_tokenizer_p->tokenize(prompts_);

    // adapt token tensor dtype to the text encoder's declared input:
    // tokenizer emits int32 (legacy exports expect that), newer exports
    // (e.g. SD v2.x via optimum) declare int64 input_ids
    ONNXTensorElementDataType ids_type_ = model_input_element_type(0);

    // legacy exports expose exactly [last_hidden_state, pooler_output];
    // SDXL-style exports additionally dump every hidden_states.N layer and
    // must be run with ORT-allocated outputs, selecting layers by name
    const bool legacy_outputs_ = (model_output_count() <= 2);

    std::string hidden_pick_ = "last_hidden_state";
    if (!legacy_outputs_ && sd_clip_config.use_penultimate) {
        long layer_count_ = 0;
        for (size_t o_ = 0; o_ < model_output_count(); ++o_) {
            if (model_output_name(o_).rfind("hidden_states.", 0) == 0) layer_count_++;
        }
        hidden_pick_ = "hidden_states." + std::to_string(std::max(0L, layer_count_ - 2));
    }

    std::vector<Tensor> merged_hidden_;
    Tensor pooled_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
    for (auto &tw_pair_: tokenizer_output_) {           // major_hidden_dim = 768 in SD, 1280 in SDXL
        Tensor &tokens_ = tw_pair_.first;               // [1, 77]
        Tensor &weight_ = tw_pair_.second;              // [1, 77]

        std::vector<Tensor> input_tensors;
        if (ids_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            input_tensors.emplace_back(TensorHelper::cast<int64_t, int32_t>(tokens_));
        } else {
            input_tensors.emplace_back(TensorHelper::clone<int32_t>(tokens_)); // [vocab_size, major_hidden_dim]
        }

        Tensor hidden_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        if (legacy_outputs_) {
            std::vector<Tensor> output_tensors;         // [1, 77, major_hidden_dim]
            generate_output(output_tensors);
            execute(input_tensors, output_tensors);
            hidden_ = std::move(output_tensors[0]);
            if (!TensorHelper::have_data(pooled_) && output_tensors.size() > 1) {
                pooled_ = TensorHelper::clone<float>(output_tensors[1]);
            }
        } else {
            std::vector<Tensor> output_tensors = execute_alloc(input_tensors);
            for (size_t o_ = 0; o_ < output_tensors.size(); ++o_) {
                const std::string name_ = model_output_name(o_);
                if (name_ == hidden_pick_) {
                    hidden_ = TensorHelper::clone<float>(output_tensors[o_]);
                } else if (name_ == "pooler_output" || name_ == "text_embeds") {
                    pooled_ = TensorHelper::clone<float>(output_tensors[o_]);
                }
            }
        }
        if (!TensorHelper::have_data(hidden_)) {
            amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: clip hidden output not found"));
        }

        merged_hidden_.push_back(                       // [1, 77, major_hidden_dim]
            TensorHelper::weight<float>(hidden_, weight_, 1, true)
        );
    }
    // seems not right
    Tensor hidden_state_ = TensorHelper::merge<float>(merged_hidden_, 1);  // [1, 77 * N, major_hidden_dim]

    return {std::move(hidden_state_), std::move(pooled_)};
}

} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_CLIP_H
