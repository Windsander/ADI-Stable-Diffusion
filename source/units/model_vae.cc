﻿/*
 * Copyright (c) 2018-2050 SD_VAE - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_VAE_H
#define MODEL_VAE_H

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


#define DEFAULT_VAEs_CONDIG                                          \
    {                                                                \
        /*sd_decode_scale_strength*/ 0.18215f                        \
    }                                                                \

typedef struct ModelVAEsConfig {
    float sd_decode_scale_strength;
} ModelVAEsConfig;

class VAE : public ModelBase {
private:
    ModelVAEsConfig sd_vae_config = DEFAULT_VAEs_CONDIG;

public:
    explicit VAE(std::string model_path_, const ModelVAEsConfig &vae_config_ = DEFAULT_VAEs_CONDIG);
    ~VAE() override;

    Tensor encode(const Tensor &inimage_);
    Tensor decode(const Tensor &latents_);
};

VAE::VAE(std::string model_path_, const ModelVAEsConfig &vae_config_) : ModelBase(std::move(model_path_)){
    sd_vae_config = vae_config_;
}

VAE::~VAE(){
    sd_vae_config.~ModelVAEsConfig();
}

Tensor VAE::encode(const Tensor &inimage_) {
    Ort::AllocatorWithDefaultOptions ort_alloc;

    std::vector<Tensor> input_tensors;
    input_tensors.push_back(TensorHelper::multiple(inimage_, 2.0f, -1.0f));
    std::vector<Tensor> output_tensors;
    execute(input_tensors, output_tensors);

    Tensor result_ = output_tensors.front().GetValue(0, ort_alloc);
    return result_;
}

Tensor VAE::decode(const Tensor &latents_) {
    Ort::AllocatorWithDefaultOptions ort_alloc;

    std::vector<Tensor> input_tensors;
    input_tensors.push_back(TensorHelper::multiple(latents_, (1.0f / sd_vae_config.sd_decode_scale_strength)));
    std::vector<Tensor> output_tensors;
    execute(input_tensors, output_tensors);

    Tensor result_ = output_tensors.front().GetValue(0, ort_alloc);
    return result_;
}


} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_VAE_H
