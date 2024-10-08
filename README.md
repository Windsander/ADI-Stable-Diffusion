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
 --dict ../../sd/sd-dictionary/vocab.txt \
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

## Extra intelligence：

- **Manually Prepare Inference Engine, see at: [Engine's README.md](engine%2FREADME.md)**

- **Manually Prepare ONNX-Format Converter & SD-Models, see at: [SD_ORT's README.md](sd%2FREADME.md)**

## Development Progress Checklist (latest):

**Basic Pipeline Functionalities (Major)**
- [x] **[SD_v1] Stable-Diffusion (v1.0 ~ v1.5, turbo)** <span style="color:green;">_(after 2024/06/04 tested)_</span>
    - **v1.0** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion): Initial version ✅
    - **v1.1** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-1): Improved image quality and generation speed ✅
    - **v1.2** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-2): Further optimized generation effects ✅
    - **v1.3** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-3): Added more training data ✅
    - **v1.4** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-4): Enhanced image generation diversity ✅
    - **v1.5** [(HuggingFace)](https://huggingface.co/runwayml/stable-diffusion-v1-5): Final optimized version ✅
    - **turbo** [(HuggingFace)](https://huggingface.co/stabilityai/sd-turbo): Community-driven optimized version, faster and efficiency ✅

- [ ] **[SD_v2] Stable-Diffusion (v2.0, v2.1)**
    - **v2.0** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-2): Significant improvements in image quality and generation efficiency
    - **v2.1** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-2-1): Further optimized model stability and generation effects

- [ ] **[SD_v3] Stable-Diffusion (v3.0)**
    - **v3.0** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-3): Anticipated next-generation version with more improvements and new features

- [ ] **[SDXL] Stable-Diffusion-XL**
    - **SDXL** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-xl): Experimental version for larger-scale models and higher-resolution image
    - **SDXL-turbo** [(HuggingFace)](https://huggingface.co/stabilityai/sdxl-turbo): Community-driven optimized version, faster and efficiency

- [ ] **[SVD] Stable-Video-Diffusion**
    - **SVD** [(HuggingFace)](https://huggingface.co/stabilityai/stable-video-diffusion): Version specifically for video generation and editing

**Scheduler Abilities**
- [ ] **Strategy**
    - [x] Discrete/Method Default (discrete) _(after 2024/05/22)_
    - [ ] Karras (karras)

- [ ] **Sampling Methods**
    - [x] Euler (euler) <span style="color:green;">_(after 2024/06/04 ✅tested)_</span> 
    - [x] Euler Ancestral (euler_a) <span style="color:green;">_(after 2024/05/24 ✅tested)_</span>
    - [x] Laplacian Pyramid Sampling (lms) <span style="color:green;">_(after 2024/07/09 ✅tested)_</span>
    - [x] Latent Consistency Models (lcm) <span style="color:green;">_(after 2024/07/04 ✅tested)_</span>
    - [x] Heun's Predictor-Corrector (heun) <span style="color:green;">_(after 2024/07/08 ✅tested)_</span>
    - [ ] Unified Predictor-Corrector (uni_pc)
    - [ ] Pseudo Numerical Diffusion Model Scheduler (pndm)
    - [ ] Improved Pseudo Numerical Diffusion Model Scheduler (ipndm)
    - [ ] Diffusion Exponential Integrator Sampler Multistep (deis_m)
    - [x] Denoising Diffusion Implicit Models (ddim) <span style="color:green;">_(after 2024/07/12 ✅tested)_</span>
    - [x] Denoising Diffusion Probabilistic Models (ddpm) <span style="color:green;">_(after 2024/07/09 ✅tested)_</span>
    - [ ] Diffusion Probabilistic Models Solver in Stochastic Differential Equations (dpm_sde)
    - [ ] Diffusion Probabilistic Models Solver in Multistep (dpm_m)
    - [ ] Diffusion Probabilistic Models Solver in Singlestep (dpm_s)

**Tokenizer Type**
- [x] Byte-Pair Encoding (bpe) <span style="color:green;">_(after 2024/07/03 ✅tested)_</span> 
- [x] Word Piece Encoding (wp) <span style="color:green;">_(after 2024/05/27 ✅tested)_</span>
- [ ] Sentence Piece Encoding (sp)  <span style="color:blue;">_[if necessary]_</span>