<h1 align="center">Agile Diffusers Inference (ADI) </h1>

<p align="center">
  <b>English</b> · <a href="README_CN.md">简体中文</a>
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

**Agile Diffusers Inference (ADI)** is a **C++ library** with **CLI tool**. Purpose to leverage the acceleration capabilities of [ONNXRuntime](https://onnxruntime.ai) and the high compatibility of the .onnx model format to provide a convenient solution for the engineering deployment of Stable Diffusion, with suitable package size & high performance. 

> **Pure C++17 · zero Python at inference time · one CLI, nine platform targets.**
> From SD v1.5 to SD3.5 / FLUX / SVD — pick a scheduler, point at the ONNX models, and generate.

## Showcase

Everything below was generated **locally by `adi` (C++ / ONNXRuntime, CPU)** — no Python, no diffusers at inference time:

| FLUX.1-schnell · 1024px · 4 steps | SD3.5-turbo · 1024px · 4 steps | SD v2.1 · 768px · 20 steps |
|:---:|:---:|:---:|
| ![FLUX.1-schnell](assets/showcase/flux-schnell-1024.jpg) | ![SD3.5-turbo](assets/showcase/sd35-turbo-1024.jpg) | ![SD v2.1](assets/showcase/sd21-768.jpg) |

| SDXL-turbo · 512px · 4 steps | sd-turbo · 512px · 4 steps | **SVD img2vid** · 14 frames · 25 steps |
|:---:|:---:|:---:|
| ![SDXL-turbo](assets/showcase/sdxl-turbo-512.jpg) | ![sd-turbo](assets/showcase/sd-turbo-512.jpg) | ![SVD img2vid](assets/showcase/svd-img2vid-14f.gif) |

> Prompt for the stills: *"A cat in the water at sunset"* · SVD animates a single input photo into a 14-frame clip.

## What's new in v2.0.0

**🆕 New model families**
- 🧠 **MMDiT era**: **SD3.5-turbo** (triple text encoders incl. T5-XXL via SentencePiece) and **FLUX.1-schnell** (packed latents, rotary ids), both at 1024px
- 🎬 **SVD img2vid**: image-to-video mode with a dedicated `euler_svd` scheduler, spatio-temporal UNet and CLIP vision encoder

**⚙️ Runtime & engine**
- 🎚️ **Precision policy** — `--precision auto|fp32|fp16`: probes available RAM and derives fp16 model copies on demand for memory-constrained machines
- 🚀 **ONNX Runtime 1.28.0** with bit-identical output vs the previous engine baseline (25/25 local regression cases)
- 🎛️ **Full sampler arsenal** — 14 discrete schedulers + Karras sigmas + rectified-flow family, all numpy-verified against diffusers

**📦 Distribution**
- 🤖 **Automated release chain**: per-platform artifacts for Android ×4, Linux ×2, macOS arm64, Windows ×2 — see [Releases](https://github.com/Windsander/ADI-Stable-Diffusion/releases)

Full history: [CHANGELOG.md](CHANGELOG.md) · Roadmap & progress: [ROADMAP.md](ROADMAP.md)

## Performance at a glance

Measured locally on an **Apple M4 Max (128 GB)**, ONNXRuntime 1.28.0, default provider, single cold run — no GPU, no Python, just the `adi` CLI and the ONNX model files:

| Model | Resolution | Steps | Wall time |
|---|---|---|---|
| sd-turbo | 512×512 | 4 | **≈ 9.7 s** |
| SD3.5-turbo (MMDiT, triple encoders) | 1024×1024 | 4 | **≈ 164 s** |

## Why choose ONNXRuntime as our Inference Engine?

- **Open Source:** ONNXRuntime is an open-source project, allowing users to freely use and modify it to suit different application scenarios.

- **Scalability:** It supports custom operators and optimizations, allowing for extensions and optimizations based on specific needs.

- **High Performance:** ONNXRuntime is highly optimized to provide fast inference speeds, suitable for real-time applications.

- **Strong Compatibility:** It supports model conversion from multiple deep learning frameworks (such as PyTorch, TensorFlow), making integration and deployment convenient.

- **Cross-Platform Support:** ONNXRuntime supports multiple hardware platforms, including CPU, GPU, TPU, etc., enabling efficient execution on various devices.

- **Community and Enterprise Support:** Developed and maintained by Microsoft, it has an active community and enterprise support, providing continuous updates and maintenance.

## How to install (CLI)?

### Method 1: Install the Command Line Tool Using a Package Manager

> **Note:** packages are published on the
> **[Releases](https://github.com/Windsander/ADI-Stable-Diffusion/releases)** page,
> and the package channels below are refreshed automatically by the deploy chain
> on every release. Since **v2.0.0**, the Homebrew tap serves **Apple Silicon
> (arm64)** only — upstream ONNXRuntime discontinued the osx-x86_64 prebuilt,
> so Intel Macs please build from source (Method 3).

```bash
## macOS (Homebrew, Apple Silicon):
brew tap windsander/adi-stable-diffusion
brew install adi

## Windows (git-Bash + Chocolatey):
curl -L -o adi.2.0.0.nupkg "https://raw.githubusercontent.com/Windsander/ADI-Stable-Diffusion/deploy/adi.2.0.0.nupkg"
choco install adi.2.0.0.nupkg -y
```

### Method 2: Download from the Released Version

You can find the latest available version from the **[Release Assets](https://github.com/Windsander/ADI-Stable-Diffusion/releases)**. The file tree of the package will look like this:
```
--bin
    --adi
--lib
    --[Corresponding platform's ADI library, e.g., libadi.a]
    --[Corresponding platform's ORT library, e.g., libonnxruntime.dylib]
--include
    --adi.h
--CHANGELOG.md
--README.md
--LICENSE
```

After unzipping, you can simply install the `bin` and `lib` directories to your system, or just go into the unzipped `bin` directory, and start using `adi`.

### Method 3: Build [adi-lib & adi-cli] Locally

- **An automated script is provided to compile ADI on your device more easily.**

Simply execute the script [auto_build.sh](auto_build.sh):

```bash
# if you do not pass the BUILD_TYPE parameter, the script will use the default Debug build type.
# and, if you not enable certain ORTProvider by [options]], script will choose default ORTProvider by platform
bash ./auto_build.sh

# Example-MacOS:
bash ./auto_build.sh --platform macos --build-type debug
           
# Example-Windows:
bash ./auto_build.sh --platform windows --build-type debug
                    
# Example-Linux(Ubuntu):
bash ./auto_build.sh --platform linux --build-type debug
           
# Example-Android:
bash ./auto_build.sh --platform android \
           --build-type debug \
           --android-ndk /Volumes/AL-Data-W04/WorkingEnv/Android/sdk/ndk/26.1.10909125 \
           --android-ver 27
           
# Example(with Extra Options) as below, build release with CUDA=ON TensorRT=ON, and custom compiler configs
bash ./auto_build.sh [params] \
           --cmake /opt/homebrew/Cellar/cmake/3.29.5/bin/cmake \
           --ninja /usr/local/bin/ninja \
           --arch-abi x86_64 \
           --jobs 8 \
           --options "-DORT_ENABLE_CUDA=ON -DORT_ENABLE_TENSOR_RT=ON"
```

currently, this project provide below [Options]:
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
enable if you have to **(ONLY FOR YOU TRULY NEEDS, UNRECOMMENDED)**.

## How to use?

### Example: 1-step Euler_A img2img latent space visualized

- **Below show What actually happened in [Example: 1-step img2img inference] in Latent Space (Skip All Models):**
![sd-euler_a-1step-latent-example.png](sd%2Fio-examples%2Fsd-euler_a-1step-latent-example.png)

- **You can use the command-line tools generated by CMake to execute the relevant functionalities of this project**

doing 1-step img2img inference, like:
```bash
# Optional(if using local build & not install): cd to ./[your_adi_path]/bin/ ,like: 
cd ./cmake-build-debug/bin/

# and here is an example of using this tool:
# sd-turbo, img2img, positive, inference_steps=1, guide=1.0, euler_a(for 1-step purpose)
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

And now, you can have a try~ (0w0 )

### More verified examples

**v2.0.0 (MMDiT era & video):**

```bash
# SD3.5-turbo @ 1024px: triple encoders (CLIP-L + OpenCLIP-G via --clip2, T5-XXL via --clip3 + --sp-model)
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

# FLUX.1-schnell @ 1024px (shift=1.0, guidance-distilled):
adi ... --clip <onnx-flux-schnell>/text_encoder/model.onnx \
        --clip3 <onnx-flux-schnell>/text_encoder_2/model.onnx \
        --unet <onnx-flux-schnell>/transformer/model.onnx ... \
 --scheduler flow_euler --shift 1.0 --decoding 0.3611 --decode-shift 0.1159 \
 --guidance 1.0 --steps 4

# SVD img2vid: one input photo -> 14-frame clip (output_0000.png ... output_0013.png)
adi -m img2vid -i input.png -o output.png \
 --image-encoder <onnx-svd-xt>/image_encoder/model.onnx \
 --unet  <onnx-svd-xt>/unet/model.onnx \
 --vae-encoder <onnx-svd-xt>/vae_encoder/model.onnx \
 --vae-decoder <onnx-svd-xt>/vae_decoder/model.onnx \
 -w 1024 -h 576 -c 3 --seed 15.0 \
 --scheduler euler_svd --predictor v_prediction \
 --frames 14 --fps 7 --motion-bucket 127 --noise-aug 0.02 \
 --decoding 0.18215 --guidance 3.0 --steps 4

# Low-memory machine? let ADI derive fp16 model copies on demand:
adi ... --precision auto    # probes RAM, converts to <model-set>-fp16/ once, caches
```

**v1.2.0 (SD v2.x / SDXL):**

```bash
# SD v2.1 @ 768px, v_prediction, 20 steps:
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

# SDXL-turbo: dual text encoders via --clip2 (VAE scaling 0.13025):
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

# Karras sigma schedule (composable with any scheduler):
adi ... --scheduler dpm_m --sigma karras ...

# All 14 schedulers:
# euler / euler_a / lms / lcm / heun / ddpm / ddim / unipc
# dpm_m / dpm_sde / dpm_s / pndm / ipndm / deis_m
```

**Model-specific parameter notes:**
| Model | `--dims` | `--predictor` | `--decoding` | typical |
|---|---|---|---|---|
| sd v1.x / turbo | 768 (v1.x) / 1024 (turbo) | epsilon | 0.18215 | turbo: guidance 1.0, 1~4 steps |
| sd v2.x | 1024 | v_prediction (v2.1-768) | 0.18215 | 768px for v2.1-768 |
| SDXL / SDXL-turbo | 768 | epsilon | **0.13025** | requires `--clip2` |
| SD3.5 / SD3.5-turbo | 768 | epsilon | **1.5305** (+ `--decode-shift 0.0609`) | requires `--clip2` + `--clip3` + `--sp-model`; `flow_euler`, `--shift 3.0` |
| FLUX.1-schnell | 768 | epsilon | **0.3611** (+ `--decode-shift 0.1159`) | `flow_euler`, `--shift 1.0`, guidance 1.0 |
| SVD (img2vid) | 768 | v_prediction | 0.18215 | `euler_svd`; `--frames/--fps/--motion-bucket/--noise-aug` |


## Documentation

- **Project structure & design notes** — [ARCHITECTURE.md](ARCHITECTURE.md)
- **Progress checklist & next-phase plan** — [ROADMAP.md](ROADMAP.md)
- **Release history** — [CHANGELOG.md](CHANGELOG.md)
- **Manually prepare the inference engine** — [engine/README.md](engine%2FREADME.md)
- **Manually prepare the ONNX converter & SD models** — [sd/README.md](sd%2FREADME.md)
- **Contributing** — [CONTRIBUTING.md](CONTRIBUTING.md)
- **Security policy** — [SECURITY.md](SECURITY.md)

## Development Progress & Roadmap

The full per-model / per-scheduler checklist (with verification dates) and the next-phase plan now live in **[ROADMAP.md](ROADMAP.md)**.

Quick status as of **v2.0.0** (2026-08):

- **Model families:** SD v1.x & turbo ✅ · SD v2.1 ✅ · SDXL-turbo ✅ · SD3.5-turbo ✅ · FLUX.1-schnell ✅ · SVD img2vid ✅
- **Schedulers:** all 14 discrete samplers + karras sigmas ✅ · `flow_euler` for the MMDiT era ✅
- **Tokenizers:** bpe ✅ · word_piece ✅ · sentencepiece (T5-XXL) ✅
- **Engineering:** smoke-matrix runner ✅ · hardened release chain ✅ · ONNXRuntime 1.28 ✅
- **Next up:** golden-image CI, fp16 quality validation, CUDA / TensorRT device regression, FLUX.1-dev, ControlNet — details in [ROADMAP.md](ROADMAP.md).