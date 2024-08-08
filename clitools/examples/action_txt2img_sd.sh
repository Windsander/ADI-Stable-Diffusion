#!/bin/bash

# 默认路径
DEFAULT_MODEL_FILE_PATH="../../sd/sd-base-model/onnx-sd-turbo"
DEFAULT_EXECUTABLE_PATH="../../cmake-build-debug/bin/adi-cli"

# 初始化路径
MODEL_FILE_PATH=$DEFAULT_MODEL_FILE_PATH
EXECUTABLE_PATH=$DEFAULT_EXECUTABLE_PATH

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model-file)
            if [[ -n "$2" ]]; then
                MODEL_FILE_PATH="$2"
                shift 2
            else
                echo "Error: --model-file requires a non-empty option argument."
                exit 1
            fi
            ;;
        -e|--executable)
            if [[ -n "$2" ]]; then
                EXECUTABLE_PATH="$2"
                shift 2
            else
                echo "Error: --executable requires a non-empty option argument."
                exit 1
            fi
            ;;
        *)
            shift
            ;;
    esac
done

# 清理Mac临时文件
find . -name "._*" -type f -print
find . -name "._*" -type f -delete

# 执行Stable Diffusion
"$EXECUTABLE_PATH" \
 -p "\
 best quality, extremely detailed,\
 (keep main character),\
 A cat in the water at sunset\
 "\
 -n "\
 worst quality, low quality, normal quality, lowres, watermark, monochrome, grayscale, ugly, blurry,\
 Tan skin, dark skin, black skin, skin spots, skin blemishes, age spot, glans, disabled, bad anatomy,\
 amputation, bad proportions, twins, missing body, fused body, extra head, poorly drawn face, bad eyes,\
 deformed eye, unclear eyes, cross-eyed, long neck, malformed limbs, extra limbs, extra arms, missing arms,\
 bad tongue, strange fingers, mutated hands, missing hands, poorly drawn hands, extra hands, fused hands,\
 connected hand, bad hands, missing fingers, extra fingers, 4 fingers, 3 fingers, deformed hands,\
 extra legs, bad legs, many legs, more than two legs, bad feet, extra feet\
 "\
 -m txt2img\
 -t cpu\
 -o ../../sd/io-test/output-t2i-step4-ddim.png\
 -w 512 -h 512 -c 3\
 --seed 15.0\
 --dims 1024\
 --clip $MODEL_FILE_PATH/text_encoder/model.onnx\
 --unet $MODEL_FILE_PATH/unet/model.onnx\
 --vae-encoder $MODEL_FILE_PATH/vae_encoder/model.onnx\
 --vae-decoder $MODEL_FILE_PATH/vae_decoder/model.onnx\
 --merges $MODEL_FILE_PATH/tokenizer/merges.txt\
 --dict $MODEL_FILE_PATH/tokenizer/vocab.json\
 --beta-start 0.00085\
 --beta-end 0.012\
 --beta scaled_linear\
 --alpha cos\
 --scheduler ddim\
 --predictor epsilon\
 --tokenizer bpe\
 --train-steps 1000\
 --token-idx-num 49408\
 --token-length 77\
 --token-border 1.0\
 --gain 1.1\
 --decoding 0.18215\
 --guidance 1.0\
 --strength 1.0\
 --steps 4\
 -v