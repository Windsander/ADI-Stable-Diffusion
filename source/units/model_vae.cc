/*
 * Copyright (c) 2018-2050 SD_VAE - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_VAE_H
#define MODEL_VAE_H

#include "model_base.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace Ort;
using namespace detail;

#define DEFAULT_VAEs_CONFIG                                          \
    {                                                                \
        /*sd_decode_scale_strength*/  0.18215f,                      \
        /*sd_input_width*/            512,                           \
        /*sd_input_height*/           512,                           \
        /*sd_input_channel*/          4,                             \
    }                                                                \

typedef struct ModelVAEsConfig {
    float sd_decode_scale_strength;
    uint64_t sd_input_width;
    uint64_t sd_input_height;
    uint64_t sd_input_channel;
} ModelVAEsConfig;

class VAE : public ModelBase {
private:
    ModelVAEsConfig sd_vae_config = DEFAULT_VAEs_CONFIG;

protected:
    void generate_output(std::vector<Tensor> &output_tensors_) override;

public:
    explicit VAE(const std::string &model_path_, const ModelVAEsConfig &vae_config_ = DEFAULT_VAEs_CONFIG);
    ~VAE() override;

    Tensor encode(const Tensor &inimage_);
    Tensor decode(const Tensor &latents_);
};

VAE::VAE(const std::string &model_path_, const ModelVAEsConfig &vae_config_) : ModelBase(model_path_){
    sd_vae_config = vae_config_;
}

VAE::~VAE(){
    sd_vae_config.~ModelVAEsConfig();
}

void VAE::generate_output(std::vector<Tensor> &output_tensors_) {
    std::vector<float> output_hidden_(
        sd_vae_config.sd_input_width *
        sd_vae_config.sd_input_height *
        sd_vae_config.sd_input_channel
    );
    TensorShape hidden_shape_ = {
        1,
        int64_t(sd_vae_config.sd_input_channel),
        int64_t(sd_vae_config.sd_input_height),
        int64_t(sd_vae_config.sd_input_width)
    };
    output_tensors_.emplace_back(TensorHelper::create(hidden_shape_, output_hidden_));
}

Tensor VAE::encode(const Tensor &inimage_) {
    if (!TensorHelper::have_data(inimage_)) { return TensorHelper::empty<float>(); }
    std::vector<Tensor> input_tensors;
    input_tensors.push_back(TensorHelper::multiple<float>(inimage_, 2.0f, -1.0f));
    std::vector<Tensor> output_tensors;
    generate_output(output_tensors);
    execute(input_tensors, output_tensors);

    Tensor result_ = TensorHelper::multiple<float>(output_tensors.front(), sd_vae_config.sd_decode_scale_strength);
    return result_;
}

Tensor VAE::decode(const Tensor &latents_) {
    if (!TensorHelper::have_data(latents_)) { return TensorHelper::empty<float>(); }
    std::vector<Tensor> input_tensors;
    input_tensors.push_back(TensorHelper::multiple<float>(latents_, (1.0f / sd_vae_config.sd_decode_scale_strength)));
    std::vector<Tensor> output_tensors;
    generate_output(output_tensors);
    execute(input_tensors, output_tensors);

    Tensor result_ = TensorHelper::divide<float>(output_tensors.front(), 2.0f, +0.5f, true);
    return result_;
}


} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_VAE_H

