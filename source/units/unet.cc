/*
 * Copyright (c) 2018-2050 SD_UNet - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_UNET_H
#define MODEL_UNET_H

#include "model_base.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace scheduler;
using namespace Ort;
using namespace detail;

typedef struct ModelPathConfig {
    std::string onnx_tokenizer_path;
    std::string onnx_text_encoder_path;
    std::string onnx_unet_path;
    std::string onnx_vae_path;
    std::string onnx_safty_path;
} ModelPathConfig;

typedef struct StableDiffusionConfig {
    ORTBasicsConfig sd_ort_basic_config = {};
    ModelPathConfig sd_modelpath_config = {};
    SchedulerConfig sd_scheduler_config = {};
    SchedulerType sd_scheduler_type = SchedulerType::SCHEDULER_EULAR_A;
    int sd_inference_steps = 1000;
    int sd_input_width = 512;
    int sd_input_height = 512;
    uint64_t sd_inference_seed = 1000;
} StableDiffusionConfig;

typedef struct UNetConfig {
    SchedulerConfig sd_scheduler_config = {};
    SchedulerType sd_scheduler_type = SchedulerType::SCHEDULER_EULAR_A;
    int sd_inference_steps = 3;
    int sd_input_width = 512;
    int sd_input_height = 512;
    int sd_input_channel = 4;
    uint64_t sd_inference_seed = 1000;
} UNetConfig;

typedef struct StableDiffusionRemain {
    Ort::Session tokenizer_session;
    Ort::Session txtencode_session;
    Ort::Session unet_base_session;
    Ort::Session vaes_base_session;
    Ort::Session safe_tool_session;

    Tensor* init_noise_latents = nullptr;
    Tensor* prompts_embeddings = nullptr;
    Tensor* image_hint = nullptr;
} StableDiffusionRemain;

class UNet : public ModelBase {
private:
    UNetConfig sd_unet_config;
    SchedulerEntity_ptr sd_scheduler_p;

public:
    explicit UNet(std::string model_path_, UNetConfig unet_config_ = {});
    ~UNet() override;

    Tensor execute_unet(Tensor txt_embeddings_, Tensor sample_);

    void encode_first_stage(Ort::Session session_);

    void generate_latent(Ort::Session session_);

    Tensor perform_guidance(Tensor& noisePred, Tensor& noisePredText, double guidanceScale);

    IMAGE_BYTE inference(Ort::Session session_, std::string prompt, StableDiffusionConfig config);
};

UNet::UNet(std::string model_path_, UNetConfig unet_config_) : ModelBase(model_path_){
    sd_unet_config = unet_config_;
    sd_scheduler_p = SchedulerRegister::request_scheduler(
        unet_config_.sd_scheduler_type,
        unet_config_.sd_scheduler_config
    );
}

UNet::~UNet(){
    sd_scheduler_p = SchedulerRegister::recycle_scheduler(sd_scheduler_p);
}

Tensor UNet::execute_unet(Tensor txt_embeddings_, Tensor sample_) {

    // TODO init with init_sigma & sample_
    Tensor latents_ = Tensor::CreateTensor<float>(
        _OrtMemoryInfo, temp, input_tensor_length,
        _inputTensorShape.data(), _inputTensorShape.size()
    );

    float *blob = nullptr;
    std::vector<int64_t> input_shape_ {
        sd_unet_config.sd_input_width,
        sd_unet_config.sd_input_height,
        sd_unet_config.sd_input_channel,
        1
    };
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, inputTensorValues.data(), inputTensorSize,
        input_shape_.data(), input_shape_.size()
    ));

    for (int i = 0; i < sd_unet_config.sd_inference_steps; ++i) {
        Tensor timestep_ = sd_scheduler_p->time(i);

        std::vector<Tensor> input_tensors;
        Ort::AllocatorWithDefaultOptions ort_alloc;
        input_tensors.push_back(txt_embeddings_.GetValue(0, ort_alloc));
        input_tensors.push_back(latents_.GetValue(0, ort_alloc));
        input_tensors.push_back(timestep_.GetValue(0, ort_alloc));
        std::vector<Tensor> output_tensors;
        execute(input_tensors, output_tensors);

        // Split output_tensors from [2, 4, 64, 64] to [1, 4, 64, 64]
        Tensor& noise_pred_img_ = output_tensors[0];
        Tensor& noise_pred_txt_ = output_tensors[1];
        Tensor guided_pred_ = perform_guidance(noise_pred_img_, noise_pred_txt_, 1.0);
        latents_ = sd_scheduler_p->step(latents_, guided_pred_, i);
    }
    
    return latents_;
}


} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_UNET_H

