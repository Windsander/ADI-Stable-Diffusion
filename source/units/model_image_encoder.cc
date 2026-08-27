/*
 * Copyright (c) 2018-2050 SD_ImageEncoder - Arikan.Li
 * Created by Arikan.Li on 2026/08/26.
 *
 * CLIP vision tower (CLIP-ViT-H image_encoder) for SVD img2vid conditioning.
 * NOT a Clip subclass: no tokenizer, pixel input instead of text.
 *
 * Preprocessing matches diffusers StableVideoDiffusionPipeline._encode_image:
 *   image [0,1] -> *2-1 -> antialiased resize DIRECTLY to 224x224 (aspect not
 *   preserved, center crop disabled) -> /2+0.5 -> CLIP normalize
 *   (x - mean) / std, mean=[0.48145466, 0.4578275, 0.40821073],
 *   std=[0.26862954, 0.26130258, 0.27577711]
 * Resize uses stb_image_resize2 (vendored OSS, catmullrom cubic filter as the
 * bicubic-antialias approximation).
 */
#ifndef MODEL_IMAGE_ENCODER_H
#define MODEL_IMAGE_ENCODER_H

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../../clitools/stb/stb_image_resize2.h"

#include "model_base.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace Ort;
using namespace detail;

#define DEFAULT_IMAGE_ENCODER_CONFIG                                     \
    {                                                                    \
        /*sd_crop_size*/        224,                                     \
        /*sd_embed_dim*/        1024,                                    \
    }

typedef struct ModelImageEncoderConfig {
    uint64_t sd_crop_size;      // CLIP input edge (SVD: 224)
    uint64_t sd_embed_dim;      // image_embeds dim (CLIP-ViT-H projection: 1024)
} ModelImageEncoderConfig;

class ImageEncoder : public ModelBase {
private:
    ModelImageEncoderConfig sd_image_encoder_config = DEFAULT_IMAGE_ENCODER_CONFIG;

    // CLIP-ViT normalization constants (feature_extractor/preprocessor_config.json)
    static constexpr float CLIP_MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static constexpr float CLIP_STD[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

protected:
    void generate_output(std::vector<Tensor> &output_tensors_) override {
        std::vector<float> output_embeds_(sd_image_encoder_config.sd_embed_dim, 0.0f);
        TensorShape embeds_shape_ = {1, int64_t(sd_image_encoder_config.sd_embed_dim)};
        output_tensors_.emplace_back(TensorHelper::create(embeds_shape_, output_embeds_));
    }

public:
    explicit ImageEncoder(const std::string &model_path_,
                          const ModelImageEncoderConfig &encoder_config_ = DEFAULT_IMAGE_ENCODER_CONFIG)
        : ModelBase(model_path_) {
        sd_image_encoder_config = encoder_config_;
    }
    ~ImageEncoder() override = default;

    // image_data_: raw interleaved RGB bytes at (width_ x height_)
    // returns image_embeds [1, 1, sd_embed_dim] (unsqueeze(1) as in diffusers)
    Tensor embedding(const IMAGE_DATA &image_data_, uint64_t width_, uint64_t height_) {
        if (!image_data_.data_) return TensorHelper::empty<float>();
        if (sd_image_encoder_config.sd_crop_size == 0 || sd_image_encoder_config.sd_embed_dim == 0) {
            amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: image encoder config zero-initialized"));
        }

        const int crop_ = int(sd_image_encoder_config.sd_crop_size);

        // bytes -> float [0, 1], direct (aspect-distorting) resize to 224x224,
        // matching diffusers _resize_with_antialiasing(image, (224, 224))
        std::vector<float> pixels_(size_t(width_) * height_ * 3);
        for (size_t i = 0; i < pixels_.size(); ++i) {
            pixels_[i] = float(image_data_.data_[i]) / 255.0f;
        }
        std::vector<float> resized_(size_t(crop_) * crop_ * 3);
        stbir_resize_float_linear(
            pixels_.data(), int(width_), int(height_), 0,
            resized_.data(), crop_, crop_, 0,
            STBIR_RGB
        );

        // CLIP normalize, NCHW planar
        std::vector<float> tensor_value_(size_t(crop_) * crop_ * 3);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < crop_; ++h) {
                for (int w = 0; w < crop_; ++w) {
                    float v_ = resized_[(h * crop_ + w) * 3 + c];
                    tensor_value_[(c * crop_ + h) * crop_ + w] = (v_ - CLIP_MEAN[c]) / CLIP_STD[c];
                }
            }
        }

        std::vector<Tensor> input_tensors;
        input_tensors.emplace_back(TensorHelper::create(TensorShape{1, 3, crop_, crop_}, tensor_value_));
        std::vector<Tensor> output_tensors;
        generate_output(output_tensors);
        execute(input_tensors, output_tensors);

        // [1, embed_dim] -> [1, 1, embed_dim] (pipeline unsqueeze(1))
        Tensor embeds_ = std::move(output_tensors.front());
        std::vector<float> embeds_value_(
            embeds_.GetTensorData<float>(),
            embeds_.GetTensorData<float>() + sd_image_encoder_config.sd_embed_dim
        );
        return TensorHelper::create(
            TensorShape{1, 1, int64_t(sd_image_encoder_config.sd_embed_dim)}, embeds_value_
        );
    }
};

} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_IMAGE_ENCODER_H
