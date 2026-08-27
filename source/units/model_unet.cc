/*
 * Copyright (c) 2018-2050 SD_UNet - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_UNET_H
#define MODEL_UNET_H

#include "model_base.cc"
#include "scheduler_register.cc"

namespace onnx {
namespace sd {
namespace units {

using namespace base;
using namespace amon;
using namespace scheduler;
using namespace Ort;
using namespace detail;

#define DEFAULT_UNET_CONFIG                                          \
    {                                                                \
        /*sd_scheduler_config*/ DEFAULT_SCHEDULER_CONFIG,            \
        /*sd_inference_steps*/  3,                                   \
        /*sd_input_width*/      512,                                 \
        /*sd_input_height*/     512,                                 \
        /*sd_input_channel*/    4,                                   \
        /*sd_scale_guidance*/   7.5f,                                \
        /*sd_random_intensity*/ 1.0f                                 \
    }                                                                \

typedef struct ModelUNetConfig {
    SchedulerConfig sd_scheduler_config;
    uint64_t sd_inference_steps;
    uint64_t sd_input_width;
    uint64_t sd_input_height;
    uint64_t sd_input_channel;
    float sd_scale_guidance;
    float sd_random_intensity;
} ModelUNetConfig;

class UNet : public ModelBase {
private:
    ModelUNetConfig sd_unet_config = DEFAULT_UNET_CONFIG;
    SchedulerEntity_ptr sd_scheduler_p;

protected:
    void generate_output(std::vector<Tensor>& output_tensors_) override;

public:
    explicit UNet(const std::string &model_path_, const ModelUNetConfig &unet_config_ = DEFAULT_UNET_CONFIG);
    ~UNet() override;

    Tensor inference(
        const Tensor &embs_positive_, const Tensor &embs_negative_,
        const Tensor &pooled_positive_, const Tensor &pooled_negative_,
        const Tensor &encoded_img_
    );
};

UNet::UNet(const std::string &model_path_, const ModelUNetConfig& unet_config_) : ModelBase(model_path_){
    sd_unet_config = unet_config_;
    sd_scheduler_p = SchedulerRegister::request_scheduler(unet_config_.sd_scheduler_config);
}

UNet::~UNet(){
    sd_scheduler_p = SchedulerRegister::recycle_scheduler(sd_scheduler_p);
    sd_unet_config.~ModelUNetConfig();
}

void UNet::generate_output(std::vector<Tensor> &output_tensors_) {
    std::vector<float> output_hidden_(
        sd_unet_config.sd_input_width *
        sd_unet_config.sd_input_height *
        sd_unet_config.sd_input_channel, 0.0f
    );
    TensorShape hidden_shape_ = {
        1,
        int64_t(sd_unet_config.sd_input_channel),
        int64_t(sd_unet_config.sd_input_height),
        int64_t(sd_unet_config.sd_input_width)
    };
    output_tensors_.emplace_back(TensorHelper::create(hidden_shape_, output_hidden_));
}

Tensor UNet::inference(
    const Tensor &embs_positive_,
    const Tensor &embs_negative_,
    const Tensor &pooled_positive_,
    const Tensor &pooled_negative_,
    const Tensor &encoded_img_
) {
    int w_ = int(sd_unet_config.sd_input_width);
    int h_ = int(sd_unet_config.sd_input_height);
    int c_ = int(sd_unet_config.sd_input_channel);
    // MMDiT (SD3 / FLUX) backbone declares 16-channel latent, unlike SD 1.5/xl (4).
    // SD3 export has exactly 4 inputs; FLUX adds rotary ids + guidance (7 inputs)
    // and is detected by its signature "img_ids" input.
    bool flux_conditioned_ = false;
    for (size_t ii = 0; ii < model_input_count(); ++ii) {
        if (model_input_name(ii) == "img_ids") { flux_conditioned_ = true; break; }
    }
    if (model_input_count() == 4 || flux_conditioned_) {
        c_ = 16;
    }
    sd_unet_config.sd_input_channel = c_;   // keep generate_output() in sync
    const bool need_guidance_ = (sd_unet_config.sd_scale_guidance > 1);
    const uint64_t working_steps_ = sd_scheduler_p->init(sd_unet_config.sd_inference_steps);

    // adapt timestep tensor to the UNet's declared input signature:
    // legacy exports take int64 {1}; newer exports (e.g. SD v2.x via optimum)
    // declare float scalar. Feeding a mismatched tensor makes ORT throw inside
    // ModelBase::execute, which is caught and logged — the UNet output then
    // silently stays zero and the whole trajectory decodes to pure noise.
    // Locate the timestep input BY NAME: export input order varies
    // (legacy: sample, timestep, ...; optimum SD3 MMDiT: timestep last).
    ONNXTensorElementDataType timestep_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    size_t timestep_rank_ = 1;
    for (size_t ii = 0; ii < model_input_count(); ++ii) {
        const std::string n_ = model_input_name(ii);
        if (n_ == "timestep" || n_ == "t") {
            ONNXTensorElementDataType declared_type_ = model_input_element_type(ii, &timestep_rank_);
            if (declared_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
                timestep_type_ = declared_type_;
            }
            break;
        }
    }

    // SDXL UNets declare 5 inputs (sample, timestep, encoder_hidden_states,
    // text_embeds, time_ids): micro-conditioning built from the pooled
    // embedding + [orig_h, orig_w, crop_top, crop_left, target_h, target_w].
    // MMDiT (SD3/FLUX) declares 4: pooled_projections as the 4th input.
    // FLUX counts 7 but follows the MMDiT path (name-bound below).
    const bool sdxl_conditioned_ = (model_input_count() >= 5) && !flux_conditioned_;
    const bool mmdit_conditioned_ = (model_input_count() == 4) || flux_conditioned_;
    Tensor time_ids_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
    if (sdxl_conditioned_) {
        std::vector<float> time_ids_value_ = {
            float(sd_unet_config.sd_input_height * 8), float(sd_unet_config.sd_input_width * 8),
            0.0f, 0.0f,
            float(sd_unet_config.sd_input_height * 8), float(sd_unet_config.sd_input_width * 8)
        };
        time_ids_ = TensorHelper::create(TensorShape{1, 6}, time_ids_value_);
    }

    // FLUX rotary position ids & guidance embedding.
    // img_ids: [seq, 3] = (0, row, col) over the packed 2x2 grid (diffusers
    // FluxPipeline._prepare_latent_image_ids); txt_ids: zeros [txt_seq, 3];
    // guidance: schnell is guidance-distilled -> constant 0.
    // ids dtype follows the export declaration (optimum FLUX exports INT64).
    Tensor img_ids_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
    Tensor txt_ids_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
    Tensor guidance_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
    bool flux_ids_int64_ = false;
    long flux_seq_ = 0;
    if (flux_conditioned_) {
        for (size_t ii = 0; ii < model_input_count(); ++ii) {
            if (model_input_name(ii) == "img_ids") {
                flux_ids_int64_ = (model_input_element_type(ii) == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
                break;
            }
        }
        long ph_ = h_ / 2, pw_ = w_ / 2;
        flux_seq_ = ph_ * pw_;
        std::vector<float> img_ids_value_(flux_seq_ * 3, 0.0f);
        for (long p = 0; p < ph_; ++p) {
            for (long q = 0; q < pw_; ++q) {
                img_ids_value_[(p * pw_ + q) * 3 + 1] = float(p);
                img_ids_value_[(p * pw_ + q) * 3 + 2] = float(q);
            }
        }
        img_ids_ = TensorHelper::create(TensorShape{flux_seq_, 3}, img_ids_value_);
        long tseq_ = long(embs_positive_.GetTensorTypeAndShapeInfo().GetShape()[1]);
        txt_ids_ = TensorHelper::create(TensorShape{tseq_, 3}, std::vector<float>(tseq_ * 3, 0.0f));
        guidance_ = TensorHelper::create(TensorShape{1}, std::vector<float>{0.0f});
    }

    TensorShape latent_shape_{1, c_, h_, w_};
    std::vector<float> latent_empty_(c_ * h_ * w_, 0.0f);
    Tensor latents_ = (TensorHelper::have_data(encoded_img_)) ?
                      TensorHelper::clone<float>(encoded_img_, latent_shape_) :
                      TensorHelper::create(latent_shape_, latent_empty_);
    Tensor init_mask_ = sd_scheduler_p->mask(latent_shape_);
    latents_ = TensorHelper::add<float>(latents_, init_mask_, latent_shape_);

    for (int i = 0; i < working_steps_; ++i) {
        Tensor model_latent_ = sd_scheduler_p->scale(latents_, i);
        Tensor timestep_ = sd_scheduler_p->time(i);
        if (timestep_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            float timestep_value_ = float(timestep_.GetTensorData<int64_t>()[0]);
            // diffusers FluxPipeline feeds timestep/1000 to the transformer;
            // SD3 feeds sigma*1000 directly. Our schedulers emit sigma*1000.
            if (flux_conditioned_) timestep_value_ /= 1000.0f;
            TensorShape timestep_shape_ = (timestep_rank_ == 0) ? TensorShape{} : TensorShape{1};
            timestep_ = TensorHelper::create<float>(timestep_shape_, std::vector<float>{timestep_value_});
        }
        auto clone_timestep_ = [&](const Tensor& t_) -> Tensor {
            return (timestep_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ?
                   TensorHelper::clone<float_t>(t_) : TensorHelper::clone<int64_t>(t_);
        };

        // MMDiT exports declare inputs in exporter-specific order (optimum SD3:
        // hidden_states, encoder_hidden_states, pooled_projections, timestep) —
        // bind BY DECLARED INPUT NAME; positional order cannot be assumed.
        auto bind_mmdit_inputs_ = [&](const Tensor& lat_, const Tensor& ts_,
                                      const Tensor& embs_, const Tensor& pooled_) {
            std::vector<Tensor> inputs_;
            for (size_t ii = 0; ii < model_input_count(); ++ii) {
                const std::string n_ = model_input_name(ii);
                if (n_ == "hidden_states" || n_ == "sample" || n_ == "latent_model_input") {
                    inputs_.emplace_back(TensorHelper::clone<float_t>(lat_));
                } else if (n_ == "encoder_hidden_states") {
                    inputs_.emplace_back(TensorHelper::clone<float_t>(embs_));
                } else if (n_ == "pooled_projections" || n_ == "text_embeds") {
                    inputs_.emplace_back(TensorHelper::clone<float_t>(pooled_));
                } else if (n_ == "timestep" || n_ == "t") {
                    inputs_.emplace_back(clone_timestep_(ts_));
                } else if (n_ == "img_ids") {
                    inputs_.emplace_back((flux_ids_int64_) ?
                        TensorHelper::cast<int64_t, float_t>(img_ids_) :
                        TensorHelper::clone<float_t>(img_ids_));
                } else if (n_ == "txt_ids") {
                    inputs_.emplace_back((flux_ids_int64_) ?
                        TensorHelper::cast<int64_t, float_t>(txt_ids_) :
                        TensorHelper::clone<float_t>(txt_ids_));
                } else if (n_ == "guidance") {
                    inputs_.emplace_back(TensorHelper::clone<float_t>(guidance_));
                } else {
                    amon_exception(basic_exception(EXC_LOG_ERR, "ERROR:: unbound MMDiT model input"));
                }
            }
            return inputs_;
        };

        // FLUX consumes 2x2-patch-packed latents [1, seq, C*4]; the scheduler
        // state itself stays in [1, C, H, W] and we pack/unpack around the call.
        Tensor packed_latent_ = (flux_conditioned_) ?
                                TensorHelper::pack_2x2<float>(model_latent_) :
                                TensorHelper::create(TensorShape{0}, std::vector<float>{});
        const Tensor& model_input_ = (flux_conditioned_) ? packed_latent_ : model_latent_;
        auto make_output_ = [&]() {
            std::vector<Tensor> output_tensors;
            if (flux_conditioned_) {
                std::vector<float> out_buf_(flux_seq_ * c_ * 4, 0.0f);
                output_tensors.emplace_back(TensorHelper::create(
                    TensorShape{1, int64_t(flux_seq_), int64_t(c_) * 4}, out_buf_));
            } else {
                generate_output(output_tensors);
            }
            return output_tensors;
        };
        auto unpack_pred_ = [&](Tensor pred_) {
            if (flux_conditioned_) return TensorHelper::unpack_2x2<float>(pred_, h_, w_);
            return std::move(pred_);   // Ort::Value has no copy ctor
        };

        // do positive N_pos_embed_num times
        Tensor pred_positive_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        if (TensorHelper::have_data(embs_positive_)) {
            std::vector<Tensor> input_tensors;
            if (mmdit_conditioned_) {
                input_tensors = bind_mmdit_inputs_(model_input_, timestep_, embs_positive_, pooled_positive_);
            } else {
                input_tensors.emplace_back(TensorHelper::clone<float_t>(model_input_));
                input_tensors.emplace_back(clone_timestep_(timestep_));
                input_tensors.emplace_back(TensorHelper::clone<float_t>(embs_positive_));
                if (sdxl_conditioned_) {
                    input_tensors.emplace_back(TensorHelper::clone<float_t>(pooled_positive_));
                    input_tensors.emplace_back(TensorHelper::clone<float_t>(time_ids_));
                }
            }
            std::vector<Tensor> output_tensors = make_output_();
            execute(input_tensors, output_tensors);
            pred_positive_ = unpack_pred_(std::move(output_tensors[0]));
        }

        // do negative N_neg_embed_num times
        Tensor pred_negative_ = TensorHelper::create(TensorShape{0}, std::vector<float>{});
        if (TensorHelper::have_data(embs_negative_) && need_guidance_) {
            std::vector<Tensor> input_tensors;
            if (mmdit_conditioned_) {
                input_tensors = bind_mmdit_inputs_(model_input_, timestep_, embs_negative_, pooled_negative_);
            } else {
                input_tensors.emplace_back(TensorHelper::clone<float_t>(model_input_));
                input_tensors.emplace_back(clone_timestep_(timestep_));
                input_tensors.emplace_back(TensorHelper::clone<float_t>(embs_negative_));
                if (sdxl_conditioned_) {
                    input_tensors.emplace_back(TensorHelper::clone<float_t>(pooled_negative_));
                    input_tensors.emplace_back(TensorHelper::clone<float_t>(time_ids_));
                }
            }
            std::vector<Tensor> output_tensors = make_output_();
            execute(input_tensors, output_tensors);
            pred_negative_ = unpack_pred_(std::move(output_tensors[0]));
        }

        // Merge predictions
        float merge_factor_ = sd_unet_config.sd_scale_guidance;
        Tensor guided_pred_ = (
            (need_guidance_) ?
            TensorHelper::guide<float>(pred_negative_, pred_positive_, merge_factor_) :
            TensorHelper::clone<float>(pred_positive_, latent_shape_)
        );

        // Dnoise & Step
        latents_ = sd_scheduler_p->step(latents_, guided_pred_, i, sd_unet_config.sd_random_intensity);

        CommonHelper::print_progress_bar(float(i + 1) / float(working_steps_));
    }

    sd_scheduler_p->uninit();
    return latents_;
}


} // namespace units
} // namespace sd
} // namespace onnx

#endif //MODEL_UNET_H

