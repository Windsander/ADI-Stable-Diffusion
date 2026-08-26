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
    std::string onnx_clip_2_path;   // SDXL/SD3 text_encoder_2 (empty when unused)
    std::string onnx_clip_3_path;   // SD3/FLUX text_encoder_3 = T5-XXL (empty when unused)
    std::string onnx_image_encoder_path; // SVD CLIP vision tower (non-empty selects img2vid mode)
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
    float sd_random_intensity          ; //= 1.0f;
    float sd_decode_scale_strength     ; //= 0.18215f;
    float sd_decode_shift_strength     ; //= 0.0f;
    uint64_t sd_video_frames           ; //= 14;   (SVD img2vid)
    uint64_t sd_video_fps              ; //= 7;    (SVD micro-conditioning; fps-1 is fed)
    uint64_t sd_video_motion_bucket    ; //= 127;
    float sd_video_noise_aug           ; //= 0.02f;
} OrtSD_Config;

class OrtSD_Context {
private:
    typedef struct OrtSD_Remain {
        Tensor embeded_positive = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        Tensor embeded_negative = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        Tensor pooled_positive  = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        Tensor pooled_negative  = TensorHelper::create(TensorShape{0}, std::vector<float>{});
    } OrtSD_Remain;

private:
    std::mutex ort_thread_lock;

    ONNXRuntimeExecutor* ort_executor = nullptr;
    OrtSD_Config ort_config;
    OrtSD_Remain ort_remain;

    Clip *ort_sd_clip = nullptr;
    Clip *ort_sd_clip_2 = nullptr;              // SDXL/SD3 text_encoder_2 (nullptr when unused)
    Clip *ort_sd_clip_3 = nullptr;              // SD3/FLUX text_encoder_3 = T5-XXL (nullptr when unused)
    UNet *ort_sd_unet = nullptr;
    VAE *ort_sd_vae_encoder = nullptr;
    VAE *ort_sd_vae_decoder = nullptr;
    ImageEncoder *ort_sd_image_encoder = nullptr;  // SVD img2vid (nullptr in txt2img/img2img)
    UNetVideo *ort_sd_unet_video = nullptr;        // SVD img2vid (nullptr in txt2img/img2img)

    // captured at init() time: triple-encoder (SD3/FLUX) mode implies a
    // 16-channel MMDiT latent. Must not be derived from ort_sd_clip_3 later,
    // since prepare() may release encoder sessions before inference().
    bool ort_is_mmdit = false;

private:
    Tensor convert_images(const IMAGE_DATA &image_data_) const;
    IMAGE_DATA convert_result(const Tensor &infer_output_) const;
    IMAGE_DATA inference_video(const IMAGE_DATA &image_data_);   // SVD img2vid path

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
    this->ort_remain.pooled_negative.release();
    this->ort_remain.pooled_positive.release();
}

Tensor OrtSD_Context::convert_images(const IMAGE_DATA &image_data_) const {
    if (!image_data_.data_) return TensorHelper::empty<float>();
    IMAGE_BYTE* input_data_ = image_data_.data_;
    vector<float> convert_value_(image_data_.size_);

    for (int w = 0; w < ort_config.sd_input_width; ++w) {
        for (int h = 0; h < ort_config.sd_input_height; ++h) {
            for (int c = 0; c < ort_config.sd_input_channel; ++c) {
                if (c >= 3) { continue; }
                int cur_pixel_ = int(h * ort_config.sd_input_width + w) * int(ort_config.sd_input_channel) + c;
                int tensor_at_ = int(c * ort_config.sd_input_height + h) * int(ort_config.sd_input_width) + w;
                convert_value_[tensor_at_] = (float(input_data_[cur_pixel_]) / 255.0f);
            }
        }
    }

    int w_ = int(ort_config.sd_input_width);
    int h_ = int(ort_config.sd_input_height);
    TensorShape convert_shape_{1, 3, h_, w_};
    return TensorHelper::create(convert_shape_, convert_value_);
}

IMAGE_DATA OrtSD_Context::convert_result(const onnx::sd::base::Tensor &tensor_) const {
    auto tensor_info = tensor_.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    if (shape.size() != 4) {
        throw std::runtime_error("Expected 4D tensor (N, C, H, W)");
    }

    int batch_size = int(shape[0]);
    int channels = int(shape[1]);
    int height = int(shape[2]);
    int width = int(shape[3]);

    if (batch_size != 1) {
        throw std::runtime_error("Batch size > 1 is not supported");
    }

    uint64_t image_size_ = uint64_t(height * width * channels);
    auto tensor_data_ = tensor_.GetTensorData<float>();
    auto image_data_ = new IMAGE_BYTE[image_size_];

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int tensor_at_ = (c * height + h) * width + w;
                int cur_pixel_ = (h * width + w) * channels + c;
                image_data_[cur_pixel_] = static_cast<IMAGE_BYTE>(std::round(
                    min(max(tensor_data_[tensor_at_], 0.0f), 1.0f) * 255
                ));
            }
        }
    }

    return IMAGE_DATA{image_data_, image_size_};
}

void OrtSD_Context::init() {
    const bool with_clip_2_ = !ort_config.sd_modelpath_config.onnx_clip_2_path.empty();
    const bool svd_mode_ = !ort_config.sd_modelpath_config.onnx_image_encoder_path.empty();

    // ---- SVD img2vid: image_encoder + spatio-temporal UNet + temporal VAE ----
    // (no text encoders; VAE encoder scale locked to 1.0 — SVD does NOT scale
    // image latents; decoder uses the configured scaling factor 0.18215)
    if (svd_mode_) {
        ort_sd_image_encoder = new ImageEncoder(
            ort_config.sd_modelpath_config.onnx_image_encoder_path,
            DEFAULT_IMAGE_ENCODER_CONFIG   // NB: `{}` would zero-init the config struct
        );
        ort_sd_unet_video = new UNetVideo(
            ort_config.sd_modelpath_config.onnx_unet_path,
            {
                ort_config.sd_scheduler_config,
                ort_config.sd_inference_steps,
                ort_config.sd_video_frames,
                ort_config.sd_input_width / 8,
                ort_config.sd_input_height / 8,
                ort_config.sd_video_fps,
                ort_config.sd_video_motion_bucket,
                ort_config.sd_video_noise_aug,
                1.0f,                            // guidance ramp min (diffusers default)
                ort_config.sd_scale_guidance     // guidance ramp max (--guidance)
            }
        );
        ort_sd_vae_encoder = new VAE(
            ort_config.sd_modelpath_config.onnx_vae_encoder_path,
            {
                1.0f, 0.0f,
                ort_config.sd_input_width / 8,
                ort_config.sd_input_height / 8,
                4,
            }
        );
        ort_sd_vae_decoder = new VAE(
            ort_config.sd_modelpath_config.onnx_vae_decoder_path,
            {
                ort_config.sd_decode_scale_strength,
                ort_config.sd_decode_shift_strength,
                ort_config.sd_input_width,
                ort_config.sd_input_height,
                ort_config.sd_input_channel,
            }
        );
        ort_sd_image_encoder->init(*ort_executor);
        ort_sd_unet_video->init(*ort_executor);
        ort_sd_vae_encoder->init(*ort_executor);
        ort_sd_vae_decoder->init(*ort_executor);
        return;
    }

    // ---- txt2img / img2img ----

    // SDXL: both encoders condition on the penultimate hidden state
    ort_sd_clip = new Clip(
        ort_config.sd_modelpath_config.onnx_clip_path,
        {
            ort_config.sd_tokenizer_config,
            with_clip_2_
        }
    );
    if (with_clip_2_) {
        ort_sd_clip_2 = new Clip(
            ort_config.sd_modelpath_config.onnx_clip_2_path,
            {
                ort_config.sd_tokenizer_config,
                true
            }
        );
    }

    // SD3/FLUX: 3rd encoder is T5-XXL — SentencePiece tokenizer, 32100 vocab,
    // 256-token sequence, 4096-dim hidden, last_hidden_state (no penultimate)
    if (!ort_config.sd_modelpath_config.onnx_clip_3_path.empty()) {
        ort_is_mmdit = true;
        TokenizerConfig t5_cfg_ = ort_config.sd_tokenizer_config;
        t5_cfg_.tokenizer_type = TOKENIZER_SP;
        t5_cfg_.tokenizer_dictionary_at = ort_config.sd_tokenizer_config.tokenizer_sp_model_at;
        t5_cfg_.avail_token_count = 32100;
        t5_cfg_.avail_token_size = 256;
        t5_cfg_.major_hidden_dim = 4096;
        ort_sd_clip_3 = new Clip(
            ort_config.sd_modelpath_config.onnx_clip_3_path,
            {
                t5_cfg_,
                false
            }
        );
    }

    ort_sd_unet = new UNet(
        ort_config.sd_modelpath_config.onnx_unet_path,
        {
            ort_config.sd_scheduler_config,
            ort_config.sd_inference_steps,
            ort_config.sd_input_width / 8,
            ort_config.sd_input_height / 8,
            4,
            ort_config.sd_scale_guidance,
            ort_config.sd_random_intensity
        }
    );

    ort_sd_vae_encoder = new VAE(
        ort_config.sd_modelpath_config.onnx_vae_encoder_path,
        {
            ort_config.sd_decode_scale_strength,
            ort_config.sd_decode_shift_strength,
            ort_config.sd_input_width / 8,
            ort_config.sd_input_height / 8,
            4,
        }
    );

    ort_sd_vae_decoder = new VAE(
        ort_config.sd_modelpath_config.onnx_vae_decoder_path,
        {
            ort_config.sd_decode_scale_strength,
            ort_config.sd_decode_shift_strength,
            ort_config.sd_input_width,
            ort_config.sd_input_height,
            ort_config.sd_input_channel,
        }
    );

    ort_sd_clip->init(*ort_executor);
    // NOTE: clip_2 / clip_3 は prepare() 時に遅延初期化（メモリ節約のため）
    ort_sd_unet->init(*ort_executor);
    ort_sd_vae_encoder->init(*ort_executor);
    ort_sd_vae_decoder->init(*ort_executor);
}

void OrtSD_Context::prepare(const std::string &positive_prompts_, const std::string &negative_prompts_){
    // make sure thread security, prevent prepare & inference conflict
    std::lock_guard<std::mutex> lock(ort_thread_lock);

    // SVD img2vid is text-free: conditioning happens per-inference from the image
    if (ort_sd_unet_video) return;

    // NOTE: clip_2 / clip_3 は init() 時に読み込まず prepare() 時に遅延初期化
    //（SD3.5 の text_encoder_2=2.6GB, text_encoder_3=18GB を一度にロードすると OOM）
    if (ort_sd_clip_2 && !ort_sd_clip_2->is_initialized()) {
        ort_sd_clip_2->init(*ort_executor);
    }
    if (ort_sd_clip_3 && !ort_sd_clip_3->is_initialized()) {
        ort_sd_clip_3->init(*ort_executor);
    }

    // embeded_positive_ [1, 77 * pos_N, 768], txt_encoder_1
    ClipEmbedResult embed_pos_ = ort_sd_clip->embedding(positive_prompts_);
    ClipEmbedResult embed_neg_ = ort_sd_clip->embedding(negative_prompts_);

    if (ort_sd_clip_3) {
        // SD3 / FLUX: triple-encoder orchestration (see PLAN-v2.0-mmdit.md §2)
        ClipEmbedResult embed_pos_3_ = ort_sd_clip_3->embedding(positive_prompts_);
        ClipEmbedResult embed_neg_3_ = ort_sd_clip_3->embedding(negative_prompts_);
        if (ort_sd_clip_2) {
            // SD3: (L.seq|G.seq)=2048 -> zero-pad 4096 -> concat T5 seq -> 333x4096;
            // pooled = L.pooler|G.pooler = 2048
            ClipEmbedResult embed_pos_2_ = ort_sd_clip_2->embedding(positive_prompts_);
            ClipEmbedResult embed_neg_2_ = ort_sd_clip_2->embedding(negative_prompts_);
            Tensor lg_pos_ = TensorHelper::concat_last_dim<float>(embed_pos_.hidden, embed_pos_2_.hidden);
            Tensor lg_neg_ = TensorHelper::concat_last_dim<float>(embed_neg_.hidden, embed_neg_2_.hidden);
            lg_pos_ = TensorHelper::pad_last_dim<float>(lg_pos_, 4096);
            lg_neg_ = TensorHelper::pad_last_dim<float>(lg_neg_, 4096);
            // sequence-dim concat [1,77,4096] ++ [1,256,4096] -> [1,333,4096]
            // (merge() interleaves equal chunks — wrong semantics here)
            ort_remain.embeded_positive = TensorHelper::concat_sequence<float>(lg_pos_, embed_pos_3_.hidden);
            ort_remain.embeded_negative = TensorHelper::concat_sequence<float>(lg_neg_, embed_neg_3_.hidden);
            ort_remain.pooled_positive  = TensorHelper::concat_last_dim<float>(embed_pos_.pooled, embed_pos_2_.pooled);
            ort_remain.pooled_negative  = TensorHelper::concat_last_dim<float>(embed_neg_.pooled, embed_neg_2_.pooled);
        } else {
            // FLUX: T5 sequence only; pooled = CLIP-L pooler
            ort_remain.embeded_positive = std::move(embed_pos_3_.hidden);
            ort_remain.embeded_negative = std::move(embed_neg_3_.hidden);
            ort_remain.pooled_positive  = std::move(embed_pos_.pooled);
            ort_remain.pooled_negative  = std::move(embed_neg_.pooled);
        }
    } else if (ort_sd_clip_2) {
        // SDXL: concat dual-encoder hiddens on the feature dim ([1,77,768]+[1,77,1280] -> [1,77,2048]),
        // pooled conditioning comes from the 2nd encoder's pooled output
        ClipEmbedResult embed_pos_2_ = ort_sd_clip_2->embedding(positive_prompts_);
        ClipEmbedResult embed_neg_2_ = ort_sd_clip_2->embedding(negative_prompts_);
        ort_remain.embeded_positive = TensorHelper::concat_last_dim<float>(embed_pos_.hidden, embed_pos_2_.hidden);
        ort_remain.embeded_negative = TensorHelper::concat_last_dim<float>(embed_neg_.hidden, embed_neg_2_.hidden);
        ort_remain.pooled_positive  = std::move(embed_pos_2_.pooled);
        ort_remain.pooled_negative  = std::move(embed_neg_2_.pooled);
    } else {
        ort_remain.embeded_positive = std::move(embed_pos_.hidden);
        ort_remain.embeded_negative = std::move(embed_neg_.hidden);
    }

    // prepare() 完了後、CLIP エンコーダの embed 結果は ort_remain に格納済み。
    // OOM 回避のためセッションは明示的に解放する（特に SD3.5 の text_encoder_2 + text_encoder_3 は巨大）。
    // ただしオブジェクト自体は残す：再 prepare() 時に is_initialized() 経由で遅延再 init される。
    //（ここで delete すると再 prepare() が埋め込みを更新できず、stale embedding を使う壊れた状態になる）
    if (ort_sd_clip_2 && ort_sd_clip_2->is_initialized()) {
        ort_sd_clip_2->release(*ort_executor);
    }
    if (ort_sd_clip_3 && ort_sd_clip_3->is_initialized()) {
        ort_sd_clip_3->release(*ort_executor);
    }
    // ort_sd_clip（CLIP-L）は比較的小さい（472M）だが、解放してもよい。
    // ただし sd35 のみならず sdxl など他モードでも使われる可能性があるため、一旦残す。
    // 将来メモリが厳しくなった場合、同様に解放を検討。
}

IMAGE_DATA OrtSD_Context::inference(IMAGE_DATA image_data_) {
    // make sure thread security, prevent prepare & inference conflict
    std::lock_guard<std::mutex> lock(ort_thread_lock);

    if (ort_sd_unet_video) return inference_video(image_data_);

    // input_image [1, 3, 512, 512]
    Tensor sample_image_ = convert_images(image_data_);

    // encoded_image [1, 4, 64, 64] for SD 1.5/xl;
    // for SD3.5 / FLUX the VAE encoder emits 32 channels (mean+logvar) which we
    // reparameterize to a 16-channel latent for the MMDiT backbone.
    Tensor encoded_sample_ = ort_sd_vae_encoder->encode(sample_image_);
    Tensor latent_for_unet_ = std::move(encoded_sample_);
    if (ort_is_mmdit) {
        latent_for_unet_ = ort_sd_vae_encoder->sample(latent_for_unet_);
    }

    // infered_latent_ [1, 16, 64, 64] for MMDiT / [1, 4, 64, 64] otherwise
    Tensor infered_latent_ = ort_sd_unet->inference(
        ort_remain.embeded_positive, ort_remain.embeded_negative,
        ort_remain.pooled_positive, ort_remain.pooled_negative,
        latent_for_unet_
    );

    // infered_latent_ [1, 3, 512, 512]
    Tensor decoded_tensor_ = ort_sd_vae_decoder->decode(infered_latent_);

    return convert_result(decoded_tensor_);
}

// SVD img2vid: returns ALL frames stacked in one buffer
// (size = frames * width * height * channel), frame-major RGB bytes
IMAGE_DATA OrtSD_Context::inference_video(const IMAGE_DATA &image_data_) {
    const uint64_t frames_ = ort_config.sd_video_frames;
    const uint64_t w_ = ort_config.sd_input_width;
    const uint64_t h_ = ort_config.sd_input_height;
    const uint64_t c_ = ort_config.sd_input_channel;
    const int64_t lh_ = int64_t(h_ / 8), lw_ = int64_t(w_ / 8);

    // 1. CLIP image embedding [1, 1, 1024]
    Tensor image_embeds_ = ort_sd_image_encoder->embedding(image_data_, w_, h_);

    // 2. VAE conditioning latent [1, 4, h/8, w/8] (mean, unscaled, noise-aug)
    Tensor image01_ = convert_images(image_data_);
    Tensor image_latents_ = ort_sd_vae_encoder->encode_noisy(
        image01_, ort_config.sd_video_noise_aug,
        ort_config.sd_scheduler_config.scheduler_seed
    );

    // 3. spatio-temporal denoise [1, F, 4, h/8, w/8]
    Tensor video_latents_ = ort_sd_unet_video->inference(image_embeds_, image_latents_);

    // 4. per-frame decode (diffusers decode_latents: flatten -> /scaling -> decode)
    const int64_t frame_latent_size_ = 4 * lh_ * lw_;
    const float* latents_data_ = video_latents_.GetTensorData<float>();
    uint64_t frame_bytes_ = w_ * h_ * c_;
    auto *video_data_ = new IMAGE_BYTE[frames_ * frame_bytes_];
    for (uint64_t f = 0; f < frames_; ++f) {
        std::vector<float> frame_latent_(
            latents_data_ + f * frame_latent_size_,
            latents_data_ + (f + 1) * frame_latent_size_
        );
        Tensor frame_tensor_ = TensorHelper::create(TensorShape{1, 4, lh_, lw_}, frame_latent_);
        Tensor decoded_frame_ = ort_sd_vae_decoder->decode(frame_tensor_);
        IMAGE_DATA frame_image_ = convert_result(decoded_frame_);
        std::copy_n(frame_image_.data_, frame_bytes_, video_data_ + f * frame_bytes_);
        delete[] frame_image_.data_;
        CommonHelper::print_progress_bar(float(f + 1) / float(frames_));
    }
    return IMAGE_DATA{video_data_, frames_ * frame_bytes_};
}

void OrtSD_Context::release(){
    if (ort_sd_vae_decoder) ort_sd_vae_decoder->release(*ort_executor);
    if (ort_sd_vae_encoder) ort_sd_vae_encoder->release(*ort_executor);
    if (ort_sd_unet) ort_sd_unet->release(*ort_executor);
    if (ort_sd_clip) ort_sd_clip->release(*ort_executor);
    if (ort_sd_clip_2 && ort_sd_clip_2->is_initialized()) ort_sd_clip_2->release(*ort_executor);
    if (ort_sd_clip_3 && ort_sd_clip_3->is_initialized()) ort_sd_clip_3->release(*ort_executor);
    if (ort_sd_image_encoder) ort_sd_image_encoder->release(*ort_executor);
    if (ort_sd_unet_video) ort_sd_unet_video->release(*ort_executor);

    delete ort_sd_vae_decoder;
    delete ort_sd_vae_encoder;
    delete ort_sd_unet;
    delete ort_sd_clip;
    delete ort_sd_clip_2;
    delete ort_sd_clip_3;
    delete ort_sd_image_encoder;
    delete ort_sd_unet_video;
}

} // namespace context
} // namespace sd
} // namespace onnx

#endif  // ORT_SD_CONTEXT_ONCE