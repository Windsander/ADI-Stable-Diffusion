/*
 * Copyright (c) 2018-2050 ORT_SD Command Line Implements Tools - Arikan.Li
 *
 * Image tools provide by: [stb] https://github.com/nothings/stb
 * Created by Arikan.Li on 2024/05/23.
 */
#include <string>
#include <vector>

#include <cstdio>
#include <cstring>
#include <ctime>

#include "ort_sd.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "stb/stb_image_resize2.h"

// below order match AvailableBetaType order
const char* scheduler_beta_type_str[] = {
    "linear",
    "scaled_linear",
    "squared_cos_cap_v2",
};

// below order match AvailableAlphaType order
const char* scheduler_alpha_type_str[] = {
    "cos",
    "exp",
};

// below order match AvailablePredictionType order
const char* scheduler_prediction_str[] = {
    "epsilon",
    "v_prediction",
    "sample",
};

// below order match AvailablePredictionType order
const char* scheduler_sampler_fuc_str[] = {
    "euler",
    "euler_a",
    "lms",
};

// below order match AvailablePredictionType order
const char* tokenizer_series_str[] = {
    "bpe",
    "word_piece",
};

const char* modes_str[] = {
    "txt2img",
    "img2img",
    "img2vid",
};

enum AvailableOrtSDMode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    MODE_COUNT
};

struct CommandLineInput {
    AvailableOrtSDMode mode = TXT2IMG;

    std::string input_path = "input.png";
    std::string output_path = "output.png";
    std::string positive_prompt;
    std::string negative_prompt;

    std::string onnx_clip_path;                                             // Model: CLIP Path (also known as text_encoder)
    std::string onnx_unet_path;                                             // Model: UNet Path
    std::string onnx_vae_encoder_path;                                      // Model: VAE Encoder Path (also known as vae_encoder)
    std::string onnx_vae_decoder_path;                                      // Model: VAE Decoder Path (also known as vae_decoder)
    std::string onnx_control_net_path;                                      // Model: ControlNet Path (currently not available)
    std::string onnx_safty_path;                                            // Model: Safety Security Model Path (currently not available)

    AvailableSchedulerType sd_scheduler_type = AVAILABLE_SCHEDULER_EULER;   // Scheduler: scheduler type (Euler_A, LMS)
    uint64_t scheduler_training_steps = 1000;                               // Scheduler: scheduler steps when at model training stage (can be found in model details, set by manual)
    float scheduler_beta_start = 0.00085f;                                  // Scheduler: Beta start (recommend 0.00085f)
    float scheduler_beta_end = 0.012f;                                      // Scheduler: Beta end (recommend 0.012f)
    int64_t scheduler_seed = -1;                                            // Scheduler: seed for random (config if u need certain output)
    AvailableBetaType scheduler_beta_type = BETA_TYPE_LINEAR;               // Scheduler: Beta Style (Linear. ScaleLinear, CAP_V2)
    AvailableAlphaType scheduler_alpha_type = ALPHA_TYPE_COSINE;            // Scheduler: Alpha(Beta) Method (Cos, Exp)
    AvailablePredictionType scheduler_predict_type = PREDICT_TYPE_EPSILON;  // Scheduler: Prediction Style (Epsilon, V_Pred, Sample)

    AvailableTokenizerType sd_tokenizer_type = AVAILABLE_TOKENIZER_BPE;     // Tokenizer: tokenizer type (currently only provide BPE)
    std::string tokenizer_dictionary_at;                                    // Tokenizer: vocabulary lib <one vocab per line, row treate as index>
    std::string tokenizer_aggregates_at;                                    // Tokenizer: merges file <one merge-pair per line, currently only for BPE>
    int32_t avail_token_count = 49408;                                      // Tokenizer: all available token in vocabulary totally
    int32_t avail_token_size = 77;                                          // Tokenizer: max token length (include <start> & <end>, so 75 avail)
    int32_t major_hidden_dim = 768;                                         // Tokenizer: out token length
    float major_boundary_factor = 1.0f;                                     // Tokenizer: weights for <start> & <end> mark-token
    float txt_attn_increase_factor = 1.1f;                                  // Tokenizer: weights for (prompt) to gain attention by this factor
    float txt_attn_decrease_factor = 1 / 1.1f;                              // Tokenizer: weights for [prompt] to loss attention by this factor

    uint64_t sd_inference_steps = 2;                                        // Infer_Major: inference step
    uint64_t sd_input_width = 512;                                          // Infer_Major: IO image width (match SD-model training sets, Constant)
    uint64_t sd_input_height = 512;                                         // Infer_Major: IO image height (match SD-model training sets, Constant)
    uint64_t sd_input_channel = 3;                                          // Infer_Major: IO image channel (match SD-model training sets, Constant)
    float sd_scale_guidance = 7.5f;                                         // Infer_Major: immersion rate for [value * (Positive - Negative)] residual
    float sd_decode_scale_strength = 0.18215f;                              // Infer_Major: for VAE Decoding result merged (Recommend 0.18215f)

    bool verbose = false;  // CLI-Mark: for extra infos of this tools
};

void print_params(const CommandLineInput& params) {
    printf("Params: \n");
    printf("{\n");
    printf("  IO: \n");
    printf("    input_path:                     %s\n", params.input_path.c_str());
    printf("    input_path:                     %s\n", params.output_path.c_str());

    printf("  Models: \n");
    printf("    clip_path:                      %s\n", params.onnx_clip_path.c_str());
    printf("    unet_path:                      %s\n", params.onnx_unet_path.c_str());
    printf("    vae_encoder_path:               %s\n", params.onnx_vae_encoder_path.c_str());
    printf("    vae_decoder_path:               %s\n", params.onnx_vae_decoder_path.c_str());
    printf("    control_net_path:               %s\n", params.onnx_control_net_path.c_str());
    printf("    safty_path:                     %s\n", params.onnx_safty_path.c_str());
    printf("    dictionary_path:                %s\n", params.tokenizer_dictionary_at.c_str());
    printf("    mergesfile_path:                %s\n", params.tokenizer_aggregates_at.c_str());

    printf("  Major  (by User  [necessary]): \n");
    printf("    current OrtSD mode:             %s\n"  , modes_str[params.mode]);
    printf("    current seed:                   %llu\n", params.scheduler_seed);
    printf("    positive_prompt:                %s\n"  , params.positive_prompt.c_str());
    printf("    negative_prompt:                %s\n"  , params.negative_prompt.c_str());
    printf("    scheduler_beta_start:           %.8f\n", params.scheduler_beta_start);
    printf("    scheduler_beta_end:             %.8f\n", params.scheduler_beta_end);
    printf("    guidance_factor (UNet):         %.4f\n", params.sd_scale_guidance);
    printf("    decoding_factor (VAE):          %.4f\n", params.sd_decode_scale_strength);
    printf("    inference steps:                %llu\n", params.sd_inference_steps);

    printf("  Types  (by User  [default]): \n");
    printf("    scheduler_sample_method:        %s\n", scheduler_sampler_fuc_str[params.sd_scheduler_type - 1]);
    printf("    scheduler_beta_type:            %s\n", scheduler_beta_type_str[params.scheduler_beta_type - 1]);
    printf("    scheduler_alpha_type:           %s\n", scheduler_alpha_type_str[params.scheduler_alpha_type - 1]);
    printf("    scheduler_prediction:           %s\n", scheduler_prediction_str[params.scheduler_predict_type - 1]);
    printf("    tokenizer_series:               %s\n", tokenizer_series_str[params.sd_tokenizer_type - 1]);

    printf("  Static (by Models [const]): \n");
    printf("    training steps:                 %llu\n", params.scheduler_training_steps);
    printf("    SD Const width:                 %llu\n", params.sd_input_width);
    printf("    SD Const height:                %llu\n", params.sd_input_height);
    printf("    SD Const channel:               %llu\n", params.sd_input_channel);
    printf("    SD Const token_idx_count:       %d\n"  , params.avail_token_count);
    printf("    SD Const token_length:          %d\n"  , params.avail_token_size);
    printf("    SD Const token_border_w:        %.4f\n", params.major_boundary_factor);
    printf("    SD Const hidden_state_dim:      %d\n"  , params.major_hidden_dim);
    printf("    SD Const word_attention_gain:   %.4f\n", params.txt_attn_increase_factor);
    printf("    SD Const word_attention_loss:   %.4f\n", params.txt_attn_decrease_factor);

    printf("}\n");
}

void print_usage(int argc, const char* argv[]) {
    printf("usage: %s [arguments]\n", argv[0]);
    printf("\n");
    printf("arguments (necessary):\n");
    printf("  --help                             show this help message and exit\n");
    printf("  -m, --mode [MODE]                  run mode [txt2img / img2img]\n");
    printf("  -i, --input [IMAGE]                path to the input image (default input.png) [img2img]\n");
    printf("  -o, --output [IMAGE]               path to the input image (default output.png)\n");
    printf("  -p, --positive [PROMPT]            positive prompt, request necessary \n");
    printf("  -n, --negative [PROMPT]            negative prompt, optional \n");
    printf("  -s, --seed [SEED]                  seed for generating output (default -1) \n");
    printf("                                     (INFO: keep same if you need certain output)\n");
    printf("                                     (WARN: set to -1 will be random output)\n");
    printf("  -w, --width <uint>                 IO image width (match SD-model training sets, Constant) \n");
    printf("  -h, --height <uint>                IO image height (match SD-model training sets, Constant) \n");
    printf("  -c, --channel <uint>               IO image channel (match SD-model training sets, Constant) \n");
    printf("  --dims <uint>                      clip model dealt output hidden state tensor dimension \n");
    printf("                                     (WARN: this params is constant with stable-duffusion version. \n");
    printf("                                            actually, was determined by txt_encoder(clip) style.) \n");

    printf("  --clip [CLIP_PATH]                 path to clip\n");
    printf("  --unet [UNET_PATH]                 path to unet\n");
    printf("  --vae-encoder [VAE_ENCODER_PATH]   path to vae encoder\n");
    printf("  --vae-decoder [VAE_DECODER_PATH]   path to vae decoder\n");
    printf("  --control-net [CONTROL_PATH]       path to control net\n");
    printf("  --safety [SAFETY_PATH]             path to safe security\n");
    printf("  --dict [DICTIONARY_PATH]           path to vocab dictionary \n");
    printf("  --merges [MERGES_FILE_PATH]        path to merges file (only for BPE Tokenizer) \n");

    printf("  --beta-start <float>               Beta start (default 0.00085f) \n");
    printf("  --beta-end <float>                 Beta end (default 0.012f) \n");
    printf("  --guidance <float>                 Scale for classifier-free guidance, immersion rate for [value * (Positive - Negative)] residual (default 7.5f) \n");
    printf("  --decoding <float>                 for VAE Decoding result merged (default 0.18215f) \n");
    printf("  --steps <uint>                     inference step to generate output (default 3) \n");

    printf("arguments (optional, unrecommended):\n");
    printf("  --scheduler [TYPE]                 Scheduler Type [euler / euler_a / lms] (default euler_a) \n");
    printf("  --beta [TYPE]                      Beta Style [linear / scale_linear / squared_cos_cap_v2) (default linear) \n");
    printf("  --alpha [TYPE]                     Alpha(Beta) Method [cos / exp] (default cos) \n");
    printf("  --predictor [TYPE]                 Prediction Style [epsilon / v_prediction, sample) (default epsilon) \n");
    printf("  --tokenizer [TYPE]                 Tokenizer Type [bpe] (currently only provide BPE) \n");

    printf("  --train-steps <uint>               scheduler steps when at model training stage (default 1000) \n");
    printf("  --token-idx-num <uint>             all available token in vocabulary totally (default 49408) \n");
    printf("                                     (WARN: mostly is 49408, but if using custom vocab or lager one, \n");
    printf("                                            then you must set it to match the total number of vocab-index. \n");
    printf("                                            which always present as number of lines in vocab file. ) \n");
    printf("  --token-length <uint>              max token length (default 77)    \n");
    printf("                                     (WARN: mostly is 77, include <start> & <end>, so 75=77-2 avail.) \n");
    printf("  --token-border <float>             weights for <start> & <end> mark-token (default 1.0f)    \n");
    printf("  --gain <float>                     weights for (prompt) to gain attention by this factor  (default 1.1f)   \n");
    printf("  --loss <float>                     weights for [prompt] to loss attention by this factor  (default 1/1.1f) \n");

    printf("arguments (extra):\n");
    printf("  -v, --verbose                      print extra info\n");
}

#define GET_TYPE_FROM_STR(type_, max_)               \
    [&]()->int{                                      \
        if (++i >= argc) {                           \
            invalid_arg = true;                      \
             return -1;                              \
        }                                            \
        const char* selected = argv[i];              \
        int found            = -1;                   \
        for (int d = 0; d < max_; d++) {             \
            if (!strcmp(selected, type_[d])) {       \
                found = d;                           \
            }                                        \
        }                                            \
        if (found == -1) {                           \
            invalid_arg = true;                      \
            return -1;                               \
        }                                            \
        return (found + 1);                          \
    }()

void parse_args(int argc, const char** argv, CommandLineInput& params) {
    bool invalid_arg = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-m" || arg == "--mode") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            const char* mode_selected = argv[i];
            int mode_found            = -1;
            for (int d = 0; d < MODE_COUNT; d++) {
                if (!strcmp(mode_selected, modes_str[d])) {
                    mode_found = d;
                }
            }
            if (mode_found == -1) {
                fprintf(stderr,
                        "error: invalid mode %s, must be one of [txt2img, img2img, img2vid, convert]\n",
                        mode_selected);
                exit(1);
            }
            params.mode = (AvailableOrtSDMode)mode_found;
        } else if (arg == "-i" || arg == "--input") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.input_path = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.output_path = argv[i];
        } else if (arg == "-p" || arg == "--positive") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.positive_prompt = argv[i];
        } else if (arg == "-n" || arg == "--negative") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.negative_prompt = argv[i];
        } else if (arg == "-s" || arg == "--seed") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.scheduler_seed = std::stoll(argv[i]);
        } else if (arg == "-w" || arg == "--width") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sd_input_width = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--height") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sd_input_height = std::stoi(argv[i]);
        } else if (arg == "-c" || arg == "--channel") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sd_input_channel = std::stoi(argv[i]);
        } else if (arg == "--dims") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.major_hidden_dim = std::stoi(argv[i]);
        } else if (arg == "--clip") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.onnx_clip_path = argv[i];
        } else if (arg == "--unet") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.onnx_unet_path = argv[i];
        } else if (arg == "--vae-encoder") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.onnx_vae_encoder_path = argv[i];
        } else if (arg == "--vae-decoder") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.onnx_vae_decoder_path = argv[i];
        } else if (arg == "--control-net") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.onnx_control_net_path = argv[i];
        } else if (arg == "--safety") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.onnx_safty_path = argv[i];
        } else if (arg == "--dict") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.tokenizer_dictionary_at = argv[i];
        } else if (arg == "--merges") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.tokenizer_aggregates_at = argv[i];
        } else if (arg == "--beta-start") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.scheduler_beta_start = std::stof(argv[i]);
        } else if (arg == "--beta-end") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.scheduler_beta_end = std::stof(argv[i]);
        } else if (arg == "--guidance") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sd_scale_guidance = std::stof(argv[i]);
        } else if (arg == "--decoding") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sd_decode_scale_strength = std::stof(argv[i]);
        } else if (arg == "--steps") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.sd_inference_steps = std::stoi(argv[i]);
        } else if (arg == "--scheduler") {
            int schedule_found = GET_TYPE_FROM_STR(scheduler_sampler_fuc_str, AVAILABLE_SCHEDULER_COUNT);
            if (schedule_found == -1) {
                invalid_arg = true;
                break;
            }
            params.sd_scheduler_type = (AvailableSchedulerType)schedule_found;
        } else if (arg == "--beta") {
            int betae_found = GET_TYPE_FROM_STR(scheduler_beta_type_str, BETA_COUNT);
            if (betae_found == -1) {
                invalid_arg = true;
                break;
            }
            params.scheduler_beta_type = (AvailableBetaType)betae_found;
        } else if (arg == "--alpha") {
            int alpha_found = GET_TYPE_FROM_STR(scheduler_alpha_type_str, ALPHA_COUNT);
            if (alpha_found == -1) {
                invalid_arg = true;
                break;
            }
            params.scheduler_alpha_type = (AvailableAlphaType) alpha_found;
        } else if (arg == "--predictor") {
            int predictor_found = GET_TYPE_FROM_STR(scheduler_prediction_str, AVAILABLE_PREDICTOR_COUNT);
            if (predictor_found == -1) {
                invalid_arg = true;
                break;
            }
            params.scheduler_predict_type = (AvailablePredictionType) predictor_found;
        } else if (arg == "--tokenizer") {
            int tokenizer_found = GET_TYPE_FROM_STR(tokenizer_series_str, AVAILABLE_TOKENIZER_COUNT);
            if (tokenizer_found == -1) {
                invalid_arg = true;
                break;
            }
            params.sd_tokenizer_type = (AvailableTokenizerType) tokenizer_found;
        } else if (arg == "--train-steps") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.scheduler_training_steps = std::stoi(argv[i]);
        } else if (arg == "--token-idx-num") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.avail_token_count = std::stoi(argv[i]);
        } else if (arg == "--token-length") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.avail_token_size = std::stoi(argv[i]);
        } else if (arg == "--token-border") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.major_boundary_factor = std::stof(argv[i]);
        } else if (arg == "--gain") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.txt_attn_increase_factor = std::stof(argv[i]);
        } else if (arg == "--loss") {
            if (++i >= argc) {
                invalid_arg = true;
                break;
            }
            params.txt_attn_decrease_factor = std::stof(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            exit(1);
        }
    }

    if (invalid_arg) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }

    if ((params.mode == IMG2IMG || params.mode == IMG2VID) && params.input_path.length() == 0) {
        fprintf(stderr, "error: when using the img2img mode, the following arguments are required: init-img\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.output_path.length() == 0) {
        fprintf(stderr, "error: the following arguments are required: output_path\n");
        print_usage(argc, argv);
        exit(1);
    }

    if (params.sd_input_width <= 0 || params.sd_input_width % 64 != 0) {
        fprintf(stderr, "error: the width must be a multiple of 64\n");
        exit(1);
    }

    if (params.sd_input_height <= 0 || params.sd_input_height % 64 != 0) {
        fprintf(stderr, "error: the height must be a multiple of 64\n");
        exit(1);
    }

    if (params.sd_inference_steps <= 0) {
        fprintf(stderr, "error: the sample_steps must be greater than 0\n");
        exit(1);
    }

    if (params.sd_decode_scale_strength < 0.f || params.sd_decode_scale_strength > 1.f) {
        fprintf(stderr, "error: can only work with VAE Decoding scale in [0.0, 1.0]\n");
        exit(1);
    }

    // seed random check
    if (params.scheduler_seed < 0) {
        srandom((int)time(nullptr));
        params.scheduler_seed = random();
    }
}

#undef GET_TYPE_FROM_STR

static std::string get_image_params(const CommandLineInput &params) {
    std::string parameter_string = " \n";
    if (!params.positive_prompt.empty()) {
        parameter_string += "Positive prompt: " + params.positive_prompt + "\n";
    }
    if (!params.negative_prompt.empty()) {
        parameter_string += "Negative prompt: " + params.negative_prompt + "\n";
    }
    parameter_string += "Guidance: " + std::to_string(params.sd_scale_guidance) + ", ";
    parameter_string += "Steps: " + std::to_string(params.sd_inference_steps) +
                        "[ Training with " + std::to_string(params.scheduler_training_steps) +
                        "], ";
    parameter_string += "Seed: " + std::to_string(params.scheduler_seed) + ", ";
    parameter_string += "Size: " +
                        std::to_string(params.sd_input_width) + "x" +
                        std::to_string(params.sd_input_height) + ", " + "\n";
    parameter_string += "Scheduler: " + std::string(scheduler_sampler_fuc_str[params.sd_scheduler_type - 1]) +
                        "[ Beta >> " + std::string(scheduler_beta_type_str[params.scheduler_beta_type - 1]) +
                        "  Alpha >> " + std::string(scheduler_alpha_type_str[params.scheduler_alpha_type - 1]) +
                        "], " + "\n";
    parameter_string += "Predictor: " + std::string(scheduler_prediction_str[params.scheduler_predict_type - 1]) + ", " + "\n";
    parameter_string += "Tokenizer: " + std::string(tokenizer_series_str[params.sd_tokenizer_type - 1]) + ", " + "\n";
    parameter_string += "Version: ONNXRuntime-Stable-Diffusion";
    return parameter_string;
}

static void read_image(const CommandLineInput &params, uint8_t** image_data){
    int channel = 0;
    int width = 0;
    int height = 0;

    (*image_data) = stbi_load(params.input_path.c_str(), &width, &height, &channel, 3);

    // Check image_data available
    if ((*image_data) == nullptr) {
        fprintf(stderr, "load image from '%s' failed\n", params.input_path.c_str());
        return;
    }
    if (channel < 3) {
        fprintf(stderr, "the number of channels for the input image must be >= 3, but got %d channels\n", channel);
        free((*image_data));
        return;
    }
    if (width <= 0) {
        fprintf(stderr, "error: the width of image must be greater than 0\n");
        free((*image_data));
        return;
    }
    if (height <= 0) {
        fprintf(stderr, "error: the height of image must be greater than 0\n");
        free((*image_data));
        return;
    }

    // Resize to match chosen SD-models request
    if (params.sd_input_height != height || params.sd_input_width != width) {
        printf("resize input image from %dx%d to %llux%llu\n", width, height, params.sd_input_width,
               params.sd_input_height);
        int resized_height = int(params.sd_input_height);
        int resized_width = int(params.sd_input_width);

        auto *resized_image_buffer = (uint8_t *) malloc(resized_height * resized_width * 3);
        if (resized_image_buffer == nullptr) {
            fprintf(stderr, "error: allocate memory for resize input image\n");
            free((*image_data));
            return;
        }
        stbir_resize(
            (*image_data), width, height, 0,
            resized_image_buffer, resized_width, resized_height, 0,
            STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM
        );

        free((*image_data));
        (*image_data) = resized_image_buffer;
    }
}

static void save_image(const CommandLineInput &params, uint8_t* image_data){
    if (!image_data) {
        printf("generate failed\n");
        return;
    }

    size_t last = params.output_path.find_last_of('.');
    std::string file_name = (last != std::string::npos) ? params.output_path.substr(0, last) : params.output_path;
    std::string final_image_path = file_name + ".png";
    stbi_write_png(
        final_image_path.c_str(),
        (int) params.sd_input_width, (int) params.sd_input_height, (int) params.sd_input_channel,
        image_data, 0
    );
    printf("\n");
    printf("save result image to '%s'\n", final_image_path.c_str());
    printf("\n");
    printf("all done with option '%s'\n", get_image_params(params).c_str());
    printf("\n");
    image_data = nullptr;
}

int main(int argc, const char *argv[]) {
    CommandLineInput params;

    parse_args(argc, argv, params);

    if (params.verbose) {
        print_params(params);
    }

    ortsd::IOrtSDContext_ptr ort_sd_context_ = nullptr;
    ortsd::generate_context(
        &ort_sd_context_,
        {
            {
                params.onnx_clip_path.c_str(),
                params.onnx_unet_path.c_str(),
                params.onnx_vae_encoder_path.c_str(),
                params.onnx_vae_decoder_path.c_str(),
                params.onnx_control_net_path.c_str(),
                params.onnx_safty_path.c_str()
            },
            {
                params.sd_scheduler_type,
                params.scheduler_training_steps,
                params.scheduler_beta_start,
                params.scheduler_beta_end,
                params.scheduler_seed,
                params.scheduler_beta_type,
                params.scheduler_alpha_type,
                params.scheduler_predict_type
            },
            {
                params.sd_tokenizer_type,
                params.tokenizer_dictionary_at.c_str(),
                params.tokenizer_aggregates_at.c_str(),
                params.avail_token_count,
                params.avail_token_size,
                params.major_hidden_dim,
                params.major_boundary_factor,
                params.txt_attn_increase_factor,
                params.txt_attn_decrease_factor
            },
            params.sd_inference_steps,
            params.sd_input_width,
            params.sd_input_height,
            params.sd_input_channel,
            params.sd_scale_guidance,
            params.sd_decode_scale_strength
        }
    );
    if (!ort_sd_context_) {
        printf("new_sd_ctx_t failed\n");
        return 1;
    }

    // Operation begin
    uint64_t input_image_size =  params.sd_input_width * params.sd_input_height * params.sd_input_channel;
    uint8_t *input_image_data = nullptr;
    read_image(params, &input_image_data);
    {
        ortsd::init(ort_sd_context_);

        ortsd::prepare(ort_sd_context_, params.positive_prompt.c_str(), params.negative_prompt.c_str());

        IO_IMAGE result_output_ = ortsd::inference(ort_sd_context_, {input_image_data, input_image_size});

        save_image(params, result_output_.data_);
    }
    free(input_image_data);
    // Operation end

    ortsd::released_context(&ort_sd_context_);

    return 0;
}
