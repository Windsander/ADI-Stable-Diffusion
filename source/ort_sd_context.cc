/*
 * Copyright (c) 2018-2050 ORT_SD_Context - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef ORT_SD_CONTEXT_ONCE
#define ORT_SD_CONTEXT_ONCE

#include "model_register.cc"

namespace onnx {
namespace sd {
namespace context {

using namespace base;
using namespace amon;
using namespace units;

typedef struct ModelPathConfig {
    std::string onnx_tokenizer_path;
    std::string onnx_text_encoder_path;
    std::string onnx_unet_path;
    std::string onnx_vae_path;
    std::string onnx_safty_path;
} ModelPathConfig;

typedef struct OrtSD_Config {
    ORTBasicsConfig sd_ort_basic_config = {};
    ModelPathConfig sd_modelpath_config = {};
    SchedulerConfig sd_scheduler_config = {};
    SchedulerType sd_scheduler_type = SchedulerType::SCHEDULER_EULAR_A;
    uint64_t sd_inference_steps = 3;
    uint64_t sd_input_width = 512;
    uint64_t sd_input_height = 512;
    uint64_t sd_input_channel = 4;
    float sd_scale_guidance = 0.9f;
    float sd_decode_scale_strength = 0.18215f;
} OrtSD_Config;

typedef struct OrtSD_Remain {
    Tensor* prompts_embeddings_txt = nullptr;
    Tensor* prompts_embeddings_img = nullptr;
} OrtSD_Remain;

class OrtSD_Context {
private:
    ONNXRuntimeExecutor ort_executor;
    OrtSD_Config ort_config;

    Tokenizer ort_sd_clip;
    UNet ort_sd_unet;
    VAE ort_sd_vae_encoder;
    VAE ort_sd_vae_decoder;

private:
    Tensor convert_images(const IMAGE_DATA & image_data_);
    IMAGE_DATA convert_result(const Tensor &infer_output_);

public:
    explicit OrtSD_Context();
    ~OrtSD_Context() ;

    void init();
    void prepare(std::string prompt);
    IMAGE_DATA inference(IMAGE_DATA image_data_);
    void release();
};


Tensor OrtSD_Context::convert_images(const IMAGE_DATA &image_data_) {
    auto* input_data_ = image_data_.data_;
    vector<float> convert_value_;

    for (int h = 0; h < ort_config.sd_input_height; ++h) {
        for (int w = 0; w < ort_config.sd_input_width; ++w) {
            for (int c = 0; c < ort_config.sd_input_channel; ++c) {
                if (c >= 3) { continue; }
                int cur_pixel_ = int(
                    h * ort_config.sd_input_width +
                    w * ort_config.sd_input_channel +
                    c
                );
                convert_value_.push_back(float(input_data_[cur_pixel_]) / 255.0f);
            }
        }
    }

    int w_ = int(ort_config.sd_input_width);
    int h_ = int(ort_config.sd_input_height);
    TensorShape convert_shape_{1, 3, h_, w_};
    return TensorHelper::create(convert_shape_, convert_value_);
}

IMAGE_DATA OrtSD_Context::convert_result(const Tensor &infer_output_) {
    long data_size_ = TensorHelper::get_data_size(infer_output_);
    auto* infer_data_ = infer_output_.GetTensorData<float>();

    IMAGE_BYTE image_data_[data_size_];
    auto image_size_ = uint64_t(
        ort_config.sd_input_height *
        ort_config.sd_input_width *
        ort_config.sd_input_channel
    );

    for (int h = 0; h < ort_config.sd_input_height; ++h) {
        for (int w = 0; w < ort_config.sd_input_width; ++w) {
            for (int c = 0; c < ort_config.sd_input_channel; ++c) {
                int cur_pixel_ = int(
                    h * ort_config.sd_input_width +
                    w * ort_config.sd_input_channel +
                    c
                );
                image_data_[cur_pixel_] = (c < 3) ? (IMAGE_BYTE) (uint8_t(std::round(
                    std::min(std::max((infer_data_[cur_pixel_] / 2 + 0.5), 0.0), 1.0) * 255)
                )) : (IMAGE_BYTE) (uint8_t(1));
            }
        }
    }

    IMAGE_DATA image_result_ = {
        image_data_, image_size_
    };

    return image_result_;
}

void OrtSD_Context::init() {
    ort_sd_clip = Tokenizer(
        ort_config.sd_modelpath_config.onnx_tokenizer_path,
        {

        }
    );

    ort_sd_unet = UNet(
        ort_config.sd_modelpath_config.onnx_unet_path,
        {
            ort_config.sd_scheduler_config,
            ort_config.sd_scheduler_type,
            ort_config.sd_inference_steps,
            ort_config.sd_input_width,
            ort_config.sd_input_height,
            ort_config.sd_input_channel,
            ort_config.sd_scale_guidance
        }
    );

    ort_sd_vae_encoder = VAE(
        ort_config.sd_modelpath_config.onnx_vae_path,
        {
            ort_config.sd_decode_scale_strength
        }
    );

    ort_sd_vae_decoder = VAE(
        ort_config.sd_modelpath_config.onnx_vae_path,
        {
            ort_config.sd_decode_scale_strength
        }
    );

    ort_sd_clip.init(ort_executor);
    ort_sd_unet.init(ort_executor);
    ort_sd_vae_encoder.init(ort_executor);
    ort_sd_vae_decoder.init(ort_executor);
}

void OrtSD_Context::prepare(std::string prompt){
}

IMAGE_DATA OrtSD_Context::inference(IMAGE_DATA image_data_) {

    // input_image [1, 3, h_, w_]
    Tensor sample_image_ = convert_images(image_data_);

    Tensor encoded_prompt_img_ = ort_sd_vae_encoder.encode(convert_images(image_data_));

    Tensor embeded_prompt_txt_ = ort_sd_clip.inference();

    Tensor infered_latent_ = ort_sd_unet.inference(embeded_inputs_, sample_image_);

    Tensor decoded_tensor_ = ort_sd_vae_decoder.decode(infered_latent_);

    return convert_result(decoded_tensor_);
}

void OrtSD_Context::release(){
    ort_sd_vae_decoder.release(ort_executor);
    ort_sd_vae_encoder.release(ort_executor);
    ort_sd_unet.release(ort_executor);
    ort_sd_clip.release(ort_executor);
}

} // namespace context
} // namespace sd
} // namespace onnx

#endif  // ORT_SD_CONTEXT_ONCE