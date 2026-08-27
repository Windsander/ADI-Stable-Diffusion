<h1 align="center">Agile Diffusers Inference (ADI) </h1>

<p align="center">
  <a href="https://opensource.org"><img src="https://img.shields.io/badge/Open_Source-❤️-FDA599?"/></a>
  <a href="/LICENSE"><img src="https://img.shields.io/badge/License-GNU_GPLv3-F4E28D"/></a>
  <a href="https://onnxruntime.ai"><img src="https://img.shields.io/badge/Powered%20by-ONNXRuntime-blue"/></a>
  <a href="https://github.com/Windsander/ADI-Stable-Diffusion/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/Windsander/ADI-Stable-Diffusion/test-native.yml?label=All%20platforms" alt="CI Status"/>
  </a>
</p>

<br>

**Agile Diffusers Inference (ADI)** is a **C++ library** with **CLI tool**. Purpose to leverage the acceleration capabilities of [ONNXRuntime](https://onnxruntime.ai) and the high compatibility of the .onnx model format to provide a convenient solution for the engineering deployment of Stable Diffusion, with suitable package size & high performance. 

## Why choose ONNXRuntime as our Inference Engine?

- **Open Source:** ONNXRuntime is an open-source project, allowing users to freely use and modify it to suit different application scenarios.

- **Scalability:** It supports custom operators and optimizations, allowing for extensions and optimizations based on specific needs.

- **High Performance:** ONNXRuntime is highly optimized to provide fast inference speeds, suitable for real-time applications.

- **Strong Compatibility:** It supports model conversion from multiple deep learning frameworks (such as PyTorch, TensorFlow), making integration and deployment convenient.

- **Cross-Platform Support:** ONNXRuntime supports multiple hardware platforms, including CPU, GPU, TPU, etc., enabling efficient execution on various devices.

- **Community and Enterprise Support:** Developed and maintained by Microsoft, it has an active community and enterprise support, providing continuous updates and maintenance.

## How to install (CLI)?

### Method 1: Install the Command Line Tool Using a Package Manager

> **Note:** published packages are currently **v1.0.1**; v1.1.0/v1.2.0 packages will be
> produced by the automated release chain (see `release/release-v*` branches).

```bash
## macOS (Homebrew):
brew tap windsander/adi-stable-diffusion
brew install adi

## Windows (git-Bash + Chocolatey):
curl -L -o adi.1.0.1.nupkg "https://raw.githubusercontent.com/Windsander/ADI-Stable-Diffusion/deploy/adi.1.0.1.nupkg"
choco install adi.1.0.1.nupkg -y
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

### More verified examples (v1.2.0)

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


## Extra intelligence：

- **Project structure & design notes, see at: [ARCHITECTURE.md](ARCHITECTURE.md)**

- **Manually Prepare Inference Engine, see at: [Engine's README.md](engine%2FREADME.md)**

- **Manually Prepare ONNX-Format Converter & SD-Models, see at: [SD_ORT's README.md](sd%2FREADME.md)**

## Development Progress Checklist (latest):

> Audited against the **v2.0.0** codebase (2026-08); roadmap targets aligned with [PLAN-supplement.md](PLAN-supplement.md).

**Basic Pipeline Functionalities (Major)**
- [x] **[SD_v1] Stable-Diffusion (v1.0 ~ v1.5, turbo)** <span style="color:green;">_(after 2024/06/04 tested)_</span>
    - **v1.0** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion): Initial version ✅
    - **v1.1** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-1): Improved image quality and generation speed ✅
    - **v1.2** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-2): Further optimized generation effects ✅
    - **v1.3** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-3): Added more training data ✅
    - **v1.4** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-4): Enhanced image generation diversity ✅
    - **v1.5** [(HuggingFace)](https://huggingface.co/runwayml/stable-diffusion-v1-5): Final optimized version ✅
    - **turbo** [(HuggingFace)](https://huggingface.co/stabilityai/sd-turbo): Community-driven optimized version, faster and efficiency ✅

- [x] **[SD_v2] Stable-Diffusion (v2.0, v2.1)** <span style="color:green;">_(v2.1 768px v-prediction after 2026/07/31 ✅tested)_</span>
    - **v2.0** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-2): Significant improvements in image quality and generation efficiency <span style="color:gray;">_(untested; identical architecture/pipeline to the verified v2.1)_</span>
    - **v2.1** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-2-1): Further optimized model stability and generation effects ✅ <span style="color:gray;">_(official repo gated since 2025 — acquired via `sd2-community` mirror + optimum ONNX export chain, see `sd/auto_prepare_sd_models.sh`)_</span>

- [x] **[SD_v3.x] Stable-Diffusion 3 / 3.5 (MMDiT era)** <span style="color:green;">_(v2.0.0 — SD3.5-turbo 1024px ✅tested 2026/08; MMDiT slot + triple-encoder orchestration + T5-XXL/sp + flow_euler all landed)_</span>
    - **v3.0** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-3-medium): First MMDiT release (2024/06) <span style="color:gray;">_(superseded by v3.5; no longer the primary target)_</span>
    - **v3.5 (Large / Large-Turbo / Medium)** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-3.5-large): Stability flagship since 2024/10 and the open-weights mainstream of 2026. Requires: MMDiT transformer unit (structural `source/units/` extension, not a config-level one), triple text encoders (CLIP-L + OpenCLIP-G + **T5-XXL → SentencePiece becomes mandatory**), rectified-flow scheduling (new scheduler family below)

- [x] **[SDXL] Stable-Diffusion-XL** <span style="color:green;">_(SDXL-turbo after 2026/07/31 ✅tested)_</span>
    - **SDXL** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): Experimental version for larger-scale models and higher-resolution image <span style="color:gray;">_(untested; same pipeline as SDXL-turbo — dual-encoder `--clip2`, pooled embedding, time-ids micro-conditioning all landed in v1.2.0)_</span>
    - **SDXL-turbo** [(HuggingFace)](https://huggingface.co/stabilityai/sdxl-turbo): Community-driven optimized version, faster and efficiency ✅

- [x] **[FLUX-class] Flow-Matching Models** <span style="color:green;">_(v2.0.0 — FLUX.1-schnell 1024px 4-step ✅tested 2026/08; 2x2 pack/unpack + img_ids/txt_ids/guidance name-binding; dev variant untested)_</span>
    - **FLUX.1 [dev] / [schnell]** [(HuggingFace)](https://huggingface.co/black-forest-labs/FLUX.1-dev): The community-favorite open-weights family of 2026. Shares the MMDiT + rectified-flow + T5-XXL stack with SD3.5, so most prerequisites land together with the SD3.5 work; extras: single-file checkpoint conversion chain, guidance-distilled variants, non-commercial license review

- [x] **[SVD] Stable-Video-Diffusion** <span style="color:green;">_(v2.0.0 — img2vid mode live 2026/08: euler_svd scheduler + UNetVideo + CLIP ViT-H ImageEncoder, 14-frame fp32 smoke ✅tested)_</span>
    - **SVD** [(HuggingFace)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid): Version specifically for video generation and editing <span style="color:gray;">_(F=14 fixed at export; F=25 XT tier needs a separate export; 25-step quality tier not yet run)_</span>

**Scheduler Abilities**
- [x] **Strategy**
    - [x] Discrete/Method Default (discrete) _(after 2024/05/22)_
    - [x] Karras (karras, ρ=7) <span style="color:green;">_(after 2026/07/30 ✅tested — sigma sequence value-identical to diffusers `use_karras_sigmas`)_</span>

- [x] **Sampling Methods (14/14 complete since v1.1.0)**
    - [x] Euler (euler) <span style="color:green;">_(after 2024/06/04 ✅tested)_</span> 
    - [x] Euler Ancestral (euler_a) <span style="color:green;">_(after 2024/05/24 ✅tested)_</span>
    - [x] Laplacian Pyramid Sampling (lms) <span style="color:green;">_(after 2024/07/09 ✅tested)_</span>
    - [x] Latent Consistency Models (lcm) <span style="color:green;">_(after 2024/07/04 ✅tested)_</span>
    - [x] Heun's Predictor-Corrector (heun) <span style="color:green;">_(after 2024/07/08 ✅tested)_</span>
    - [x] Unified Predictor-Corrector (unipc) <span style="color:green;">_(completed after 2026/07/29 ✅tested — was an empty-stub segfault before v1.1.0)_</span>
    - [x] Pseudo Numerical Diffusion Model Scheduler (pndm) <span style="color:green;">_(after 2026/07/30 ✅tested)_</span>
    - [x] Improved Pseudo Numerical Diffusion Model Scheduler (ipndm) <span style="color:green;">_(after 2026/07/30 ✅tested — paper-form AB4 + DDIM update; diffusers' ADM-grid variant verified unsuitable for SD)_</span>
    - [x] Diffusion Exponential Integrator Sampler Multistep (deis_m) <span style="color:green;">_(after 2026/07/30 ✅tested)_</span>
    - [x] Denoising Diffusion Implicit Models (ddim) <span style="color:green;">_(after 2024/07/12 ✅tested)_</span>
    - [x] Denoising Diffusion Probabilistic Models (ddpm) <span style="color:green;">_(after 2024/07/09 ✅tested)_</span>
    - [x] Diffusion Probabilistic Models Solver in Stochastic Differential Equations (dpm_sde) <span style="color:green;">_(after 2026/07/30 ✅tested)_</span>
    - [x] Diffusion Probabilistic Models Solver in Multistep (dpm_m) <span style="color:green;">_(after 2026/07/30 ✅tested)_</span>
    - [x] Diffusion Probabilistic Models Solver in Singlestep (dpm_s) <span style="color:green;">_(after 2026/07/30 ✅tested)_</span>

- [x] **Flow-Matching / Rectified-Flow Family** <span style="color:green;">_(v2.0.0 — new `FlowSchedulerBase` family, separate from `scheduler_discrete_*`)_</span>
    - [x] Flow Euler discrete <span style="color:green;">_(v2.0.0 ✅tested — default sampler of SD3.5 / FLUX-class; numpy-verified against diffusers)_</span>
    - [ ] Flow Heun / higher-order flow solvers <span style="color:blue;">_[if necessary]_</span>

**Tokenizer Type**
- [x] Byte-Pair Encoding (bpe) <span style="color:green;">_(after 2024/07/03 ✅tested — covers CLIP-L / OpenCLIP-G/H, i.e. every model supported so far)_</span> 
- [x] Word Piece Encoding (word_piece) <span style="color:green;">_(after 2024/05/27 ✅tested)_</span> <span style="color:orange;">_(note: available via internal registry & CLI, but not yet exposed in the public `AvailableTokenizerType` C enum)_</span>
- [x] Sentence Piece Encoding (sp) <span style="color:green;">_(v2.0.0 ✅tested — vendored libsentencepiece static build, official library; T5-XXL for SD3.5 / FLUX)_</span>

**Engineering & Distribution** _(audited 2026-08)_
- [x] Smoke-matrix runner scripted (`sd/io-test/run_smoke_matrix.sh`, 19 quick / 24 full cases; hard gates: ORT-exception count, output size, flat-pixel check) <span style="color:green;">_(after 2026/07/31 — local quick 19/19 green)_</span>
- [x] Release chain hardened (CHANGELOG-driven auto-publish; retired node12 actions replaced; artifact actions v2→v4; `deploy_linux` formally removed per v2.0.0 decision G3) <span style="color:green;">_(after 2026/07/31)_</span>
- [ ] Golden-image regression in CI (`test-native` is compile-only today; the smoke matrix is not yet wired into workflows)
- [x] ONNXRuntime engine upgrade <span style="color:green;">_(1.18.0 → 1.28.0 prebuilt packages; gpu-cuda12 → gpu_cuda12 renames handled; osx-x86_64 / win-x86 prebuilts discontinued upstream → local-build fallback; local regression turbo 13/13 + sd35 1024px bit-match with the 1.18 baseline, 2026/08/26)_</span>
- [x] Linux .deb/.rpm packaging — **decision: removed, not repaired** <span style="color:green;">_(v2.0.0 decision G3, 2026/08/26: deb rules predate the ORT 1.19+ package rename/soname change and have no maintainer; Linux build+run support stays covered by test-native ubuntu legs and the manual smoke job)_</span>
- [ ] ControlNet / safety-checker integration <span style="color:gray;">_(fields reserved in `IOrtSDConfig` as `onnx_control_net_path` / `onnx_safty_path`, currently not available)_</span>