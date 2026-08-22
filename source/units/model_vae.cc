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
        /*sd_decode_shift_strength*/  0.0f,                          \
        /*sd_input_width*/            512,                           \
        /*sd_input_height*/           512,                           \
        /*sd_input_channel*/          4,                             \
    }

typedef struct ModelVAEsConfig {
    float sd_decode_scale_strength;
    float sd_decode_shift_strength;         // SD3 VAE shift_factor (default 0.0; SD3.5 = 0.0609)
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
    Tensor sample(const Tensor &latent_params_);
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

    Tensor result_ = TensorHelper::multiple<float>(
        output_tensors.front(),
        sd_vae_config.sd_decode_scale_strength,
        -sd_vae_config.sd_decode_shift_strength * sd_vae_config.sd_decode_scale_strength
    );
    return result_;
}

Tensor VAE::sample(const Tensor &latent_params_) {
    // SD3.5/FLUX VAE encoder outputs 32 channels: [1, 16 (mean), 16 (logvar), H, W].
    // reparameterize to a 16-channel latent for the MMDiT backbone.
    if (!TensorHelper::have_data(latent_params_)) {
        return TensorHelper::empty<float>();
    }
    auto info = latent_params_.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    if (shape.size() != 4) {
        return TensorHelper::empty<float>();
    }
    int64_t c = shape[1];
    int64_t half_c = c / 2;
    int64_t h = shape[2];
    int64_t w = shape[3];
    int64_t spatial = h * w;
    int64_t total = c * spatial;
    const float* src = latent_params_.GetTensorData<float>();

    std::vector<float> mean_(half_c * spatial);
    std::vector<float> logvar_(half_c * spatial);
    for (int64_t i = 0; i < half_c * spatial; ++i) {
        mean_[i]  = src[i];
        logvar_[i] = src[half_c * spatial + i];
    }

    std::vector<float> latent_(half_c * spatial);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < half_c * spatial; ++i) {
        float stddev = std::exp(0.5f * logvar_[i]);
        latent_[i] = mean_[i] + stddev * dist(gen);
    }

    TensorShape out_shape_{1, half_c, h, w};
    return TensorHelper::create(out_shape_, latent_);
}

Tensor VAE::decode(const Tensor &latents_) {
    if (!TensorHelper::have_data(latents_)) { return TensorHelper::empty<float>(); }
    std::vector<Tensor> input_tensors;
    // diffusers AutoencoderKL: latents / scaling_factor + shift_factor
    input_tensors.push_back(TensorHelper::multiple<float>(
        latents_,
        (1.0f / sd_vae_config.sd_decode_scale_strength),
        sd_vae_config.sd_decode_shift_strength
    ));
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

