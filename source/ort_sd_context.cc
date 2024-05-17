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

class OrtSD_Context {
private:
    typedef struct OrtSD_Remain {
        Tensor embeded_positive;
        Tensor embeded_negative;
    } OrtSD_Remain;

private:
    ONNXRuntimeExecutor* ort_executor = nullptr;
    OrtSD_Config ort_config;
    OrtSD_Remain ort_remain;

    Tokenizer ort_sd_clip;
    UNet ort_sd_unet;
    VAE ort_sd_vae_encoder;
    VAE ort_sd_vae_decoder;

private:
    Tensor convert_images(const IMAGE_DATA & image_data_) const;
    IMAGE_DATA convert_result(const Tensor &infer_output_) const;

public:
    explicit OrtSD_Context(const OrtSD_Config& ort_config_);
    ~OrtSD_Context() ;

    void init();
    void prepare(std::string positive_prompts_, std::string negative_prompts_);
    IMAGE_DATA inference(IMAGE_DATA image_data_);
    void release();
};

OrtSD_Context::OrtSD_Context(const OrtSD_Config& ort_config_){
    this->ort_config = ort_config_;
    ort_executor = new ONNXRuntimeExecutor(ort_config_.sd_ort_basic_config);
}

OrtSD_Context::~OrtSD_Context(){
    if (ort_executor) {
        ort_executor->~ONNXRuntimeExecutor();
        delete ort_executor;
    }
    this->ort_config = {};
}


Tensor OrtSD_Context::convert_images(const IMAGE_DATA &image_data_) const {
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

IMAGE_DATA OrtSD_Context::convert_result(const Tensor &infer_output_) const {
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
            ort_config.sd_decode_scale_strength   // it's unused for encoding
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

void OrtSD_Context::prepare(std::string positive_prompts_, std::string negative_prompts_){
    // embeded_positive_ [p_count_, 4, 64, 64] TODO move to prepare
    ort_remain.embeded_positive = ort_sd_clip.tokenize(positive_prompts_);

    // embeded_negative_ [n_count_, 4, 64, 64] TODO move to prepare
    ort_remain.embeded_negative = ort_sd_clip.tokenize(negative_prompts_);
}

IMAGE_DATA OrtSD_Context::inference(IMAGE_DATA image_data_) {

    // input_image [1, 3, 512, 512]
    Tensor sample_image_ = convert_images(image_data_);

    // encoded_image [1, 4, 64, 64]
    Tensor encoded_sample_ = ort_sd_vae_encoder.encode(convert_images(image_data_));

    // infered_latent_ [1, 4, 64, 64]
    Tensor infered_latent_ = ort_sd_unet.inference(ort_remain.embeded_positive, ort_remain.embeded_negative, encoded_sample_);

    // infered_latent_ [1, 3, 512, 512]
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