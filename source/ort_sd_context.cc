/*
 * Copyright (c) 2018-2050 ORT_SD_Context - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef ORT_SD_CONTEXT_ONCE
#define ORT_SD_CONTEXT_ONCE

#include "model_wrapper.cc"

namespace onnx {
namespace sd {
namespace context {

using namespace base;
using namespace amon;
using namespace units;

typedef struct ModelPathConfig {
    std::string onnx_clip_path;     // text_encoder
    std::string onnx_unet_path;
    std::string onnx_vae_encoder_path;
    std::string onnx_vae_decoder_path;
    std::string onnx_control_net_path;
    std::string onnx_safty_path;
} ModelPathConfig;

typedef struct OrtSD_Config {
    ORTBasicsConfig sd_ort_basic_config; //= {};
    ModelPathConfig sd_modelpath_config; //= {};
    SchedulerConfig sd_scheduler_config; //= {};
    TokenizerConfig sd_tokenizer_config; //= {};
    uint64_t sd_inference_steps        ; //= 3;
    uint64_t sd_input_width            ; //= 512;
    uint64_t sd_input_height           ; //= 512;
    uint64_t sd_input_channel          ; //= 4;
    float sd_scale_guidance            ; //= 0.9f;
    float sd_decode_scale_strength     ; //= 0.18215f;
} OrtSD_Config;

class OrtSD_Context {
private:
    typedef struct OrtSD_Remain {
        Tensor embeded_positive = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        Tensor embeded_negative = TensorHelper::create(TensorShape{0}, std::vector<float>{});
    } OrtSD_Remain;

private:
    ONNXRuntimeExecutor* ort_executor = nullptr;
    OrtSD_Config ort_config;
    OrtSD_Remain ort_remain;

    Clip *ort_sd_clip = nullptr;
    UNet *ort_sd_unet = nullptr;
    VAE *ort_sd_vae_encoder = nullptr;
    VAE *ort_sd_vae_decoder = nullptr;

private:
    Tensor convert_images(const IMAGE_DATA &image_data_) const;
    IMAGE_DATA convert_tensor(const Tensor &tensor_) const;
    IMAGE_DATA convert_result(const Tensor &infer_output_) const;

public:
    explicit OrtSD_Context(const OrtSD_Config& ort_config_);
    ~OrtSD_Context() ;

    void init();
    void prepare(const std::string &positive_prompts_, const std::string &negative_prompts_);
    IMAGE_DATA inference(IMAGE_DATA image_data_);
    void release();
};

OrtSD_Context::OrtSD_Context(const OrtSD_Config& ort_config_){
    this->ort_config = ort_config_;
    ort_executor = new ONNXRuntimeExecutor(ort_config_.sd_ort_basic_config);
}

OrtSD_Context::~OrtSD_Context(){
    if (ort_executor != nullptr) {
        delete ort_executor;
        ort_executor = nullptr;
    }
    this->ort_remain.embeded_negative.release();
    this->ort_remain.embeded_positive.release();
    this->ort_config = {};
}

Tensor OrtSD_Context::convert_images(const IMAGE_DATA &image_data_) const {
    IMAGE_BYTE* input_data_ = image_data_.data_;
    vector<float> convert_value_(image_data_.size_);

    for (int w = 0; w < ort_config.sd_input_width; ++w) {
        for (int h = 0; h < ort_config.sd_input_height; ++h) {
            for (int c = 0; c < ort_config.sd_input_channel; ++c) {
                if (c >= 3) { continue; }
                int cur_pixel_ = int(w + h * ort_config.sd_input_width) * int(ort_config.sd_input_channel) + c;
                int tensor_at_ = int(h + c * ort_config.sd_input_height) * int(ort_config.sd_input_width) + w;
                convert_value_[tensor_at_] = (float(input_data_[cur_pixel_]) / 255.0f);
            }
        }
    }

    int w_ = int(ort_config.sd_input_width);
    int h_ = int(ort_config.sd_input_height);
    TensorShape convert_shape_{1, 3, h_, w_};
    return TensorHelper::create(convert_shape_, convert_value_);
}

IMAGE_DATA OrtSD_Context::convert_tensor(const onnx::sd::base::Tensor &tensor_) const {
    auto tensor_info = tensor_.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    if (shape.size() != 4) {
        throw std::runtime_error("Expected 4D tensor (N, C, H, W)");
    }

    int batch_size = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];

    if (batch_size != 1) {
        throw std::runtime_error("Batch size > 1 is not supported");
    }

    uint64_t image_size_ = height * width * channels;
    auto tensor_data_ = tensor_.GetTensorData<float>();
    auto image_data_ = new IMAGE_BYTE[image_size_];

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                int tensor_index = (c * height + h) * width + w;
                int image_index = (h * width + w) * channels + c;
                image_data_[image_index] = static_cast<IMAGE_BYTE>(std::round(
                    std::min(std::max(tensor_data_[tensor_index], 0.0f), 1.0f) * 255
                ));
            }
        }
    }

    return IMAGE_DATA{image_data_, image_size_};
}

IMAGE_DATA OrtSD_Context::convert_result(const Tensor &infer_output_) const {
    uint64_t image_size_ = TensorHelper::get_data_size(infer_output_);
    auto infer_data_ = infer_output_.GetTensorData<float>();
    auto image_data_ = new IMAGE_BYTE[image_size_];

    for (int w = 0; w < ort_config.sd_input_width; ++w) {
        for (int h = 0; h < ort_config.sd_input_height; ++h) {
            for (int c = 0; c < ort_config.sd_input_channel; ++c) {
                int cur_pixel_ = int(w + h * ort_config.sd_input_width) * int(ort_config.sd_input_channel) + c;
                int tensor_at_ = int(h + c * ort_config.sd_input_height) * int(ort_config.sd_input_width) + w;
                if (c < 3) {
                    image_data_[cur_pixel_] = IMAGE_BYTE(infer_data_[tensor_at_] * 255);
                } else {
                    image_data_[cur_pixel_] = 255;  // Alpha 通道设为 255
                }
            }
        }
    }

    return IMAGE_DATA{image_data_, image_size_};
}

void OrtSD_Context::init() {
    ort_sd_clip = new Clip(
        ort_config.sd_modelpath_config.onnx_clip_path,
        {
            ort_config.sd_tokenizer_config
        }
    );

    ort_sd_unet = new UNet(
        ort_config.sd_modelpath_config.onnx_unet_path,
        {
            ort_config.sd_scheduler_config,
            ort_config.sd_inference_steps,
            ort_config.sd_input_width / 8,
            ort_config.sd_input_height / 8,
            4,
            ort_config.sd_scale_guidance
        }
    );

    ort_sd_vae_encoder = new VAE(
        ort_config.sd_modelpath_config.onnx_vae_encoder_path,
        {
            ort_config.sd_decode_scale_strength,
            ort_config.sd_input_width / 8,
            ort_config.sd_input_height / 8,
            4,
        }
    );

    ort_sd_vae_decoder = new VAE(
        ort_config.sd_modelpath_config.onnx_vae_decoder_path,
        {
            ort_config.sd_decode_scale_strength,
            ort_config.sd_input_width,
            ort_config.sd_input_height,
            ort_config.sd_input_channel,
        }
    );

    ort_sd_clip->init(*ort_executor);
    ort_sd_unet->init(*ort_executor);
    ort_sd_vae_encoder->init(*ort_executor);
    ort_sd_vae_decoder->init(*ort_executor);
}

void OrtSD_Context::prepare(const std::string &positive_prompts_, const std::string &negative_prompts_){
    // embeded_positive_ [1, 77 * pos_N, 768], txt_encoder_1
    ort_remain.embeded_positive = ort_sd_clip->embedding(positive_prompts_);

    // embeded_negative_ [1, 77 * neg_N, 768], txt_encoder_1
    ort_remain.embeded_negative = ort_sd_clip->embedding(negative_prompts_);
}

IMAGE_DATA OrtSD_Context::inference(IMAGE_DATA image_data_) {

    // input_image [1, 3, 512, 512]
    Tensor sample_image_ = convert_images(image_data_);

    // encoded_image [1, 4, 64, 64]
    Tensor encoded_sample_ = ort_sd_vae_encoder->encode(sample_image_);

    // infered_latent_ [1, 4, 64, 64]
    Tensor infered_latent_ = ort_sd_unet->inference(ort_remain.embeded_positive, ort_remain.embeded_negative, encoded_sample_);

    // infered_latent_ [1, 3, 512, 512]
    Tensor decoded_tensor_ = ort_sd_vae_decoder->decode(encoded_sample_);

    return convert_tensor(decoded_tensor_);
}

void OrtSD_Context::release(){
    ort_sd_vae_decoder->release(*ort_executor);
    ort_sd_vae_encoder->release(*ort_executor);
    ort_sd_unet->release(*ort_executor);
    ort_sd_clip->release(*ort_executor);

    delete ort_sd_vae_decoder;
    delete ort_sd_vae_encoder;
    delete ort_sd_unet;
    delete ort_sd_clip;
}

} // namespace context
} // namespace sd
} // namespace onnx

#endif  // ORT_SD_CONTEXT_ONCE