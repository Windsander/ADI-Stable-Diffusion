<h1 align="center">Agile Diffusers Inference (ADI) </h1>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="https://opensource.org"><img src="https://img.shields.io/badge/Open_Source-❤️-FDA599?"/></a>
  <a href="/LICENSE"><img src="https://img.shields.io/badge/License-GNU_GPLv3-F4E28D"/></a>
  <a href="https://onnxruntime.ai"><img src="https://img.shields.io/badge/Powered%20by-ONNXRuntime-blue"/></a>
  <a href="https://github.com/Windsander/ADI-Stable-Diffusion/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/Windsander/ADI-Stable-Diffusion/test-native.yml?label=All%20platforms" alt="CI Status"/>
  </a>
  <a href="https://github.com/Windsander/ADI-Stable-Diffusion/releases">
    <img src="https://img.shields.io/github/v/release/Windsander/ADI-Stable-Diffusion?display_name=tag" alt="Latest Release"/>
  </a>
</p>

<br>

**Agile Diffusers Inference (ADI)** 是一个 **C++ 库**，附带 **CLI 命令行工具**。目标是借助 [ONNXRuntime](https://onnxruntime.ai) 的加速能力与 .onnx 模型格式的高兼容性，为 Stable Diffusion 的工程化部署提供一个便捷的解决方案，兼顾合适的包体积与高性能。

> **纯 C++17 · 推理期零 Python 依赖 · 一个 CLI，九大平台目标。**
> 从 SD v1.5 到 SD3.5 / FLUX / SVD —— 选好调度器，指向 ONNX 模型，直接生成。

## 效果展示

以下全部内容均由 **`adi` 在本地生成（C++ / ONNXRuntime，CPU）** —— 推理阶段没有 Python，也没有 diffusers：

| FLUX.1-schnell · 1024px · 4 步 | SD3.5-turbo · 1024px · 4 步 | SD v2.1 · 768px · 20 步 |
|:---:|:---:|:---:|
| ![FLUX.1-schnell](assets/showcase/flux-schnell-1024.jpg) | ![SD3.5-turbo](assets/showcase/sd35-turbo-1024.jpg) | ![SD v2.1](assets/showcase/sd21-768.jpg) |

| SDXL-turbo · 512px · 4 步 | sd-turbo · 512px · 4 步 | **SVD img2vid** · 14 帧 · 25 步 |
|:---:|:---:|:---:|
| ![SDXL-turbo](assets/showcase/sdxl-turbo-512.jpg) | ![sd-turbo](assets/showcase/sd-turbo-512.jpg) | ![SVD img2vid](assets/showcase/svd-img2vid-14f.gif) |

> 静态图的 Prompt：*"A cat in the water at sunset"* · SVD 则是将一张输入照片动画化为 14 帧短片。

## v2.0.0 新特性

**🆕 新模型家族**
- 🧠 **MMDiT 时代**：**SD3.5-turbo**（三文本编码器，含经由 SentencePiece 的 T5-XXL）与 **FLUX.1-schnell**（packed latents、rotary ids），均支持 1024px
- 🎬 **SVD img2vid**：图生视频模式，配备专用 `euler_svd` 调度器、时空 UNet 与 CLIP 视觉编码器

**⚙️ 运行时与引擎**
- 🎚️ **精度策略** —— `--precision auto|fp32|fp16`：探测可用内存，为内存受限的机器按需派生 fp16 模型副本
- 🚀 **ONNX Runtime 1.28.0**，输出与上一代引擎基线逐位一致（本地回归 25/25 通过）
- 🎛️ **完整采样器军火库** —— 14 个离散调度器 + Karras sigmas + rectified-flow 家族，全部经过与 diffusers 的 numpy 数值校验

**📦 分发**
- 🤖 **自动化发布链**：覆盖 Android ×4、Linux ×2、macOS arm64、Windows ×2 的各平台产物 —— 见 [Releases](https://github.com/Windsander/ADI-Stable-Diffusion/releases)

完整历史：[CHANGELOG.md](CHANGELOG.md) · 路线图与进度：[ROADMAP.md](ROADMAP.md)

## 性能一览

在 **Apple M4 Max（128 GB）** 上本地实测，ONNXRuntime 1.28.0、默认 provider、单次冷跑 —— 没有 GPU，没有 Python，只有 `adi` CLI 与 ONNX 模型文件：

| 模型 | 分辨率 | 步数 | 耗时 |
|---|---|---|---|
| sd-turbo | 512×512 | 4 | **≈ 9.7 秒** |
| SD3.5-turbo（MMDiT，三编码器） | 1024×1024 | 4 | **≈ 164 秒** |

## 为什么选择 ONNXRuntime 作为推理引擎？

- **开源：** ONNXRuntime 是开源项目，用户可自由使用与修改，以适配不同的应用场景。

- **可扩展：** 支持自定义算子与优化，可针对特定需求进行扩展与优化。

- **高性能：** ONNXRuntime 经过高度优化，推理速度快，适用于实时应用。

- **强兼容：** 支持从多种深度学习框架（如 PyTorch、TensorFlow）转换模型，集成与部署都很方便。

- **跨平台：** 支持 CPU、GPU、TPU 等多种硬件平台，可在各种设备上高效执行。

- **社区与企业支持：** 由 Microsoft 开发维护，社区活跃，企业支持完善，持续更新与维护有保障。

## 如何安装（CLI）？

### 方式一：通过包管理器安装命令行工具

> **注意：** **v2.0.0** 的安装包发布在
> **[Releases](https://github.com/Windsander/ADI-Stable-Diffusion/releases)** 页面
> （由 `release/release-v*` 分支的自动化发布链产出）。
> 下方的包管理器渠道目前仍是 **v1.0.1**，将另行刷新。

```bash
## macOS (Homebrew):
brew tap windsander/adi-stable-diffusion
brew install adi

## Windows (git-Bash + Chocolatey):
curl -L -o adi.1.0.1.nupkg "https://raw.githubusercontent.com/Windsander/ADI-Stable-Diffusion/deploy/adi.1.0.1.nupkg"
choco install adi.1.0.1.nupkg -y
```

### 方式二：从发布版本下载

你可以在 **[Release Assets](https://github.com/Windsander/ADI-Stable-Diffusion/releases)** 中找到最新可用版本。包的文件树如下：
```
--bin
    --adi
--lib
    --[对应平台的 ADI 库，如 libadi.a]
    --[对应平台的 ORT 库，如 libonnxruntime.dylib]
--include
    --adi.h
--CHANGELOG.md
--README.md
--LICENSE
```

解压后，你可以将 `bin` 和 `lib` 目录安装到系统中，或者直接进入解压出的 `bin` 目录，开始使用 `adi`。

### 方式三：本地构建 [adi-lib 与 adi-cli]

- **项目提供了自动化脚本，让在你的设备上编译 ADI 更加容易。**

直接执行脚本 [auto_build.sh](auto_build.sh)：

```bash
# 如果不传 BUILD_TYPE 参数，脚本会使用默认的 Debug 构建类型。
# 如果没有通过 [options] 启用某个 ORTProvider，脚本会按平台选择默认的 ORTProvider
bash ./auto_build.sh

# 示例-MacOS:
bash ./auto_build.sh --platform macos --build-type debug
           
# 示例-Windows:
bash ./auto_build.sh --platform windows --build-type debug
                    
# 示例-Linux(Ubuntu):
bash ./auto_build.sh --platform linux --build-type debug
           
# 示例-Android:
bash ./auto_build.sh --platform android \
           --build-type debug \
           --android-ndk /Volumes/AL-Data-W04/WorkingEnv/Android/sdk/ndk/26.1.10909125 \
           --android-ver 27
           
# 示例（附加选项）如下：以 CUDA=ON、TensorRT=ON 构建 release，并自定义编译器配置
bash ./auto_build.sh [params] \
           --cmake /opt/homebrew/Cellar/cmake/3.29.5/bin/cmake \
           --ninja /usr/local/bin/ninja \
           --arch-abi x86_64 \
           --jobs 8 \
           --options "-DORT_ENABLE_CUDA=ON -DORT_ENABLE_TENSOR_RT=ON"
```

目前，本项目提供以下 [选项]：
```cmake
# 1. Option list
option(ORT_COMPILED_ONLINE           "adi: using online onnxruntime(ort), otherwise local build" ${SD_ORT_ONLINE_AVAIL})
option(ORT_COMPILED_HEAVY            "adi: using HEAVY compile, ${Red}only for debug, default OFF${ColourReset}" OFF)
option(ORT_BUILD_COMMAND_LINE        "adi: build command line tools" ${CMAKE_STANDALONE})
option(ORT_BUILD_COMBINE_BASE        "adi: build combine code together to build a single output lib" OFF)
option(ORT_BUILD_SHARED_ADI          "adi: build ADI project shared libs" OFF)
option(ORT_BUILD_SHARED_ORT          "adi: build ORT in shared libs" OFF)
option(ORT_ENABLE_TENSOR_RT          "adi: using TensorRT provider to accelerate inference" ${DEFAULT_TRT_STATE})
option(ORT_ENABLE_CUDA               "adi: using CUDA provider to accelerate inference" ${DEFAULT_CUDA_STATE})
option(ORT_ENABLE_COREML             "adi: using CoreML provider to accelerate inference" ${DEFAULT_COREML_STATE})
option(ORT_ENABLE_NNAPI              "adi: using NNAPI provider to accelerate inference" ${DEFAULT_NNAPI_STATE})
option(ADI_AUTO_INSTALL              "adi: auto-install ADI-CLI to current system when build finish, request admin permission" OFF)
```
仅在确有需要时启用 **（只在你真正需要时，不推荐默认开启）**。

## 如何使用？

### 示例：单步 Euler_A img2img —— 潜空间可视化

- **下面展示了在 [示例：单步 img2img 推理] 中，潜空间内实际发生了什么（跳过所有模型）：**
![sd-euler_a-1step-latent-example.png](sd%2Fio-examples%2Fsd-euler_a-1step-latent-example.png)

- **你可以使用由 CMake 生成的命令行工具来执行本项目的相关功能**

比如执行单步 img2img 推理：
```bash
# 可选（本地构建且未安装时）：cd 到 ./[你的_adi_路径]/bin/，例如：
cd ./cmake-build-debug/bin/

# 下面是使用该工具的一个示例：
# sd-turbo, img2img, 正向提示词, inference_steps=1, guide=1.0, euler_a（用于单步场景）
adi \
 -p "A cat in the water at sunset" \
 -m img2img \
 -i ../../sd/io-test/input-test.png \
 -o ../../sd/io-test/output.png \
 -w 512 -h 512 -c 3 \
 --seed 15.0 \
 --dims 1024 \
 --clip ../../sd/sd-base-model/onnx-sd-turbo/text_encoder/model.onnx \
 --unet ../../sd/sd-base-model/onnx-sd-turbo/unet/model.onnx \
 --vae-encoder ../../sd/sd-base-model/onnx-sd-turbo/vae_encoder/model.onnx \
 --vae-decoder ../../sd/sd-base-model/onnx-sd-turbo/vae_decoder/model.onnx \
 --dict ../../sd/sd-base-model/onnx-sd-turbo/tokenizer/vocab.json \
 --merges ../../sd/sd-base-model/onnx-sd-turbo/tokenizer/merges.txt \
 --beta-start 0.00085 \
 --beta-end 0.012 \
 --beta scaled_linear \
 --alpha cos \
 --scheduler euler_a \
 --predictor epsilon \
 --tokenizer bpe \
 --train-steps 1000 \
 --token-idx-num 49408 \
 --token-length 77 \
 --token-border 1.0 \
 --gain 1.1 \
 --decoding 0.18215 \
 --guidance 1.0 \
 --steps 1 \
 -v
```

现在，你可以亲自试一试了~ (0w0 )

### 更多已验证的示例

**v2.0.0（MMDiT 时代与视频）：**

```bash
# SD3.5-turbo @ 1024px：三编码器（CLIP-L + 经 --clip2 的 OpenCLIP-G，经 --clip3 + --sp-model 的 T5-XXL）
adi -p "A cat in the water at sunset" -m txt2img -o output.png \
 -w 1024 -h 1024 -c 3 --seed 15.0 --dims 768 \
 --clip  <onnx-sd35-turbo>/text_encoder/model.onnx \
 --clip2 <onnx-sd35-turbo>/text_encoder_2/model.onnx \
 --clip3 <onnx-sd35-turbo>/text_encoder_3/model.onnx \
 --unet  <onnx-sd35-turbo>/transformer/model.onnx \
 --vae-encoder <onnx-sd35-turbo>/vae_encoder/model.onnx \
 --vae-decoder <onnx-sd35-turbo>/vae_decoder/model.onnx \
 --dict  <onnx-sd35-turbo>/tokenizer/vocab.json \
 --merges <onnx-sd35-turbo>/tokenizer/merges.txt \
 --sp-model <onnx-sd35-turbo>/tokenizer_3/spiece.model \
 --beta scaled_linear --scheduler flow_euler --shift 3.0 --predictor epsilon --tokenizer bpe \
 --decoding 1.5305 --decode-shift 0.0609 --guidance 1.0 --steps 4

# FLUX.1-schnell @ 1024px（shift=1.0，guidance 蒸馏版）：
adi ... --clip <onnx-flux-schnell>/text_encoder/model.onnx \
        --clip3 <onnx-flux-schnell>/text_encoder_2/model.onnx \
        --unet <onnx-flux-schnell>/transformer/model.onnx ... \
 --scheduler flow_euler --shift 1.0 --decoding 0.3611 --decode-shift 0.1159 \
 --guidance 1.0 --steps 4

# SVD img2vid：一张输入照片 -> 14 帧短片（output_0000.png ... output_0013.png）
adi -m img2vid -i input.png -o output.png \
 --image-encoder <onnx-svd-xt>/image_encoder/model.onnx \
 --unet  <onnx-svd-xt>/unet/model.onnx \
 --vae-encoder <onnx-svd-xt>/vae_encoder/model.onnx \
 --vae-decoder <onnx-svd-xt>/vae_decoder/model.onnx \
 -w 1024 -h 576 -c 3 --seed 15.0 \
 --scheduler euler_svd --predictor v_prediction \
 --frames 14 --fps 7 --motion-bucket 127 --noise-aug 0.02 \
 --decoding 0.18215 --guidance 3.0 --steps 4

# 内存吃紧的机器？让 ADI 按需派生 fp16 模型副本：
adi ... --precision auto    # 探测内存，一次性转换到 <model-set>-fp16/ 并缓存
```

**v1.2.0（SD v2.x / SDXL）：**

```bash
# SD v2.1 @ 768px，v_prediction，20 步：
adi -p "A cat in the water at sunset" -m txt2img -o output.png \
 -w 768 -h 768 -c 3 --seed 15.0 --dims 1024 \
 --clip  <onnx-sd-v21-768>/text_encoder/model.onnx \
 --unet  <onnx-sd-v21-768>/unet/model.onnx \
 --vae-encoder <onnx-sd-v21-768>/vae_encoder/model.onnx \
 --vae-decoder <onnx-sd-v21-768>/vae_decoder/model.onnx \
 --dict  <onnx-sd-v21-768>/tokenizer/vocab.json \
 --merges <onnx-sd-v21-768>/tokenizer/merges.txt \
 --beta scaled_linear --scheduler euler_a --predictor v_prediction \
 --guidance 7.5 --steps 20

# SDXL-turbo：通过 --clip2 接入双文本编码器（VAE 缩放系数 0.13025）：
adi -p "A cat in the water at sunset" -m txt2img -o output.png \
 -w 512 -h 512 -c 3 --seed 15.0 --dims 768 \
 --clip  <onnx-sdxl-turbo>/text_encoder/model.onnx \
 --clip2 <onnx-sdxl-turbo>/text_encoder_2/model.onnx \
 --unet  <onnx-sdxl-turbo>/unet/model.onnx \
 --vae-encoder <onnx-sdxl-turbo>/vae_encoder/model.onnx \
 --vae-decoder <onnx-sdxl-turbo>/vae_decoder/model.onnx \
 --dict  <onnx-sdxl-turbo>/tokenizer/vocab.json \
 --merges <onnx-sdxl-turbo>/tokenizer/merges.txt \
 --beta scaled_linear --scheduler euler_a --predictor epsilon \
 --decoding 0.13025 --guidance 1.0 --steps 4

# Karras sigma 调度（可与任意调度器组合）：
adi ... --scheduler dpm_m --sigma karras ...

# 全部 14 个调度器：
# euler / euler_a / lms / lcm / heun / ddpm / ddim / unipc
# dpm_m / dpm_sde / dpm_s / pndm / ipndm / deis_m
```

**各模型的参数速查：**
| 模型 | `--dims` | `--predictor` | `--decoding` | 典型用法 |
|---|---|---|---|---|
| sd v1.x / turbo | 768 (v1.x) / 1024 (turbo) | epsilon | 0.18215 | turbo：guidance 1.0，1~4 步 |
| sd v2.x | 1024 | v_prediction (v2.1-768) | 0.18215 | v2.1-768 用 768px |
| SDXL / SDXL-turbo | 768 | epsilon | **0.13025** | 需要 `--clip2` |
| SD3.5 / SD3.5-turbo | 768 | epsilon | **1.5305**（+ `--decode-shift 0.0609`） | 需要 `--clip2` + `--clip3` + `--sp-model`；`flow_euler`，`--shift 3.0` |
| FLUX.1-schnell | 768 | epsilon | **0.3611**（+ `--decode-shift 0.1159`） | `flow_euler`，`--shift 1.0`，guidance 1.0 |
| SVD (img2vid) | 768 | v_prediction | 0.18215 | `euler_svd`；`--frames/--fps/--motion-bucket/--noise-aug` |


## 文档

- **项目结构与设计说明** —— [ARCHITECTURE.md](ARCHITECTURE.md)
- **进度清单与下一阶段计划** —— [ROADMAP.md](ROADMAP.md)
- **发布历史** —— [CHANGELOG.md](CHANGELOG.md)
- **手动准备推理引擎** —— [engine/README.md](engine%2FREADME.md)
- **手动准备 ONNX 转换器与 SD 模型** —— [sd/README.md](sd%2FREADME.md)
- **参与贡献** —— [CONTRIBUTING.md](CONTRIBUTING.md)
- **安全策略** —— [SECURITY.md](SECURITY.md)

## 开发进度与路线图

完整的逐模型 / 逐调度器清单（含验证日期）与下一阶段计划，现已迁移至 **[ROADMAP.md](ROADMAP.md)**。

截至 **v2.0.0**（2026-08）的快速状态：

- **模型家族：** SD v1.x & turbo ✅ · SD v2.1 ✅ · SDXL-turbo ✅ · SD3.5-turbo ✅ · FLUX.1-schnell ✅ · SVD img2vid ✅
- **调度器：** 全部 14 个离散采样器 + karras sigmas ✅ · MMDiT 时代的 `flow_euler` ✅
- **分词器：** bpe ✅ · word_piece ✅ · sentencepiece（T5-XXL）✅
- **工程化：** smoke 矩阵运行器 ✅ · 加固的发布链 ✅ · ONNXRuntime 1.28 ✅
- **接下来：** golden-image CI、fp16 质量验证、CUDA / TensorRT 设备回归、FLUX.1-dev、ControlNet —— 详见 [ROADMAP.md](ROADMAP.md)。
