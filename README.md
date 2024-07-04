# CppFast Diffusers Inference (CFDI)

CppFast Diffusers Inference (CFDI) is a C++ project. Its purpose is to leverage the acceleration capabilities of ONNXRuntime and the high compatibility of the .onnx model format to provide a convenient solution for the engineering deployment of Stable Diffusion.

## Why choose ONNXRuntime as our Inference Engine?

- **Open Source:** ONNXRuntime is an open-source project, allowing users to freely use and modify it to suit different application scenarios.

- **Scalability:** It supports custom operators and optimizations, allowing for extensions and optimizations based on specific needs.

- **High Performance:** ONNXRuntime is highly optimized to provide fast inference speeds, suitable for real-time applications.

- **Strong Compatibility:** It supports model conversion from multiple deep learning frameworks (such as PyTorch, TensorFlow), making integration and deployment convenient.

- **Cross-Platform Support:** ONNXRuntime supports multiple hardware platforms, including CPU, GPU, TPU, etc., enabling efficient execution on various devices.

- **Community and Enterprise Support:** Developed and maintained by Microsoft, it has an active community and enterprise support, providing continuous updates and maintenance.

## How to use?

### Example: 1-step Euler_A img2img latent space visualized

- **1. prepare ORT(ONNXRuntime) environment (Mark: unnecessary, Step-3 will do auto initialize)**
 
by simply executing auto_script [auto_prepare_engine_env.sh](engine%2Fauto_prepare_engine_env.sh):
```bash
cd ./engine
# if setting [VERSION] [PLATFORM], script will downloading target [PLATFORM] ORT, like onnxruntime-linux-x64-1.18.0.tgz at official
bash auto_prepare_engine_env.sh [VERSION] [PLATFORM]

# if not setting [VERSION] [PLATFORM], script will default using onnxruntime-osx-arm64-1.18.0, so be careful!!!
bash auto_prepare_engine_env.sh
```

or download ONNXRuntime 1.18.0 manually from [ORT official](https://github.com/microsoft/onnxruntime/releases/v1.18.0/), there's an example for [onnxruntime-osx-arm64-1.18.0](https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-arm64-1.18.0.tgz) in MacOS:
```bash
cd ./engine
# if using curl
curl -L -o onnxruntime-osx-arm64-1.18.0.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-arm64-1.18.0.tgz

# if using wget
wget -O onnxruntime-osx-arm64-1.18.0.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-arm64-1.18.0.tgz
```

Don't forget to replace `[your_ort_file]` with your ORT version folder, for example: `onnxruntime-osx-universal2-1.18.0`.

- **2. prepare ONNX models of target Stable-Diffusion**

by simply executing auto_script [auto_prepare_converter_env.sh](sd%2Fauto_prepare_converter_env.sh):
```bash
cd ./sd
bash auto_prepare_converter_env.sh
```

or download SD manually from [HuggingFace](https://huggingface.co), there's an example for [sd-turbo official](https://huggingface.co/stabilityai/sdxl-turbo/tree/main):

```bash
cd ./sd/sd-base-model/
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 -b onnx onnx-official-sd-v15/
```

- **3. build [ort-sd-clitools] for local using**

by simply executing script [auto_build.sh](auto_build.sh):
```bash
cd ./sd
# if you do not pass the BUILD_TYPE parameter, the script will use the default Debug build type.
# and, if you not enable certain ORTProvider by [options]], script will choose default ORTProvider by platform
bash auto_build.sh

# Example-MacOS:
bash ./auto_build.sh --platform macos \
           --cmake /opt/homebrew/Cellar/cmake/3.29.5/bin/cmake \
           --ninja /usr/local/bin/ninja \
           --build-type debug \
           --jobs 4
           
# Example-Android:
bash ./auto_build.sh --platform android \
           --cmake /opt/homebrew/Cellar/cmake/3.29.5/bin/cmake \
           --ninja /usr/local/bin/ninja \
           --build-type Debug \
           --jobs 4 \
           --android-sdk /Volumes/AL-Data-W04/WorkingEnv/Android/sdk \
           --android-ndk /Volumes/AL-Data-W04/WorkingEnv/Android/sdk/ndk/26.1.10909125 \
           --android-ver 27
           
# Example(with Extra Options) as below, build release with CUDA=ON TensorRT=ON
bash auto_build.sh [params] -DORT_ENABLE_CUDA=ON -DORT_ENABLE_TENSOR_RT=ON
```

currently, this project provide below [Options]:
```cmake
option(ORT_BUILD_COMMAND_LINE        "ort-sd: build command line tools" ${SD_STANDALONE})
option(ORT_BUILD_SHARED_CFDI         "ort-sd: build CFDI project shared libs" OFF)
option(ORT_BUILD_SHARED_LIBS         "ort-sd: build ORT in shared libs" OFF)
option(ORT_COMPILED_ONLINE           "ort-sd: using online onnxruntime(ort), otherwise local build" ${SD_ORT_ONLINE_AVAIL})
option(ORT_ENABLE_TENSOR_RT          "ort-sd: using TensorRT provider to accelerate inference" ${DEFAULT_TRT_STATE})
option(ORT_ENABLE_CUDA               "ort-sd: using CUDA provider to accelerate inference" ${DEFAULT_CUDA_STATE})
option(ORT_ENABLE_COREML             "ort-sd: using CoreML provider to accelerate inference" ${DEFAULT_COREML_STATE})
option(ORT_ENABLE_NNAPI              "ort-sd: using NNAPI provider to accelerate inference" ${DEFAULT_NNAPI_STATE})
```
enable if you need.

- **4. Now, you can use the command-line tools generated by CMake to execute the relevant functionalities of this project**

doing 1-step img2img inference, like:
```bash
# cd to ./[cmake_output]/bin/ ,like: 
cd ./cmake-build-debug/bin/

# and here is an example of using this tool:
# sd-turbo, img2img, positive, inference_steps=1, guide=1.0, euler_a(for 1-step purpose)
ort-sd-clitools -p "A cat in the water at sunset" -m img2img -i ../../sd/io-test/input-test.png -o ../../sd/io-test/output.png -w 512 -h 512 -c 3 --seed 15.0 --dims 1024 --clip ../../sd/sd-base-model/onnx-sd-turbo/text_encoder/model.onnx --unet ../../sd/sd-base-model/onnx-sd-turbo/unet/model.onnx --vae-encoder ../../sd/sd-base-model/onnx-sd-turbo/vae_encoder/model.onnx --vae-decoder ../../sd/sd-base-model/onnx-sd-turbo/vae_decoder/model.onnx --dict ../../sd/sd-dictionary/vocab.txt --beta-start 0.00085 --beta-end 0.012 --beta scaled_linear --alpha cos --scheduler euler_a --predictor epsilon --tokenizer bpe --train-steps 1000 --token-idx-num 49408 --token-length 77 --token-border 1.0 --gain 1.1 --decoding 0.18215 --guidance 1.0 --steps 1 -v
```

- **Below show What actually happened in [Example: 1-step img2img inference] in Latent Space (Skip All Models):**
![sd-euler_a-1step-latent-example.png](sd%2Fio-examples%2Fsd-euler_a-1step-latent-example.png)

And now, you can have a try~ (0w0 )

## Development Progress Checklist (latest):

**Basic Pipeline Functionalities (Major):**
- [x] [SD_v1] Stable-Diffusion (v1.0, v1.5, turbo) _(after 2024/06/04)_
- [ ] [SD_v2] Stable-Diffusion (v2.0, v2.1)
- [ ] [SD_v3] Stable-Diffusion (v3.0)
- [ ] [SDXL] Stable-Diffusion-XL
- [ ] [SVD] Stable-Video-Diffusion

**Scheduler Strategy**
- [x] Discrete/Method Default (discrete) _(after 2024/05/22)_
- [ ] Karras (karras)

**Scheduler Method**
- [x] Euler (euler) _(after 2024/06/04)_
- [x] Euler Ancestral (euler_a) _(after 2024/05/24)_
- [x] Laplacian Pyramid Sampling (lms) _(after 2024/05/24)_
- [x] Latent Consistency Models (lcm) _(after 2024/07/04)_
- [ ] Heun's Predictor-Corrector (heun)
- [ ] Unified Predictor-Corrector (uni_pc)
- [ ] Pseudo Numerical Diffusion Model Scheduler (pndm)
- [ ] Improved Pseudo Numerical Diffusion Model Scheduler (ipndm)
- [ ] Diffusion Exponential Integrator Sampler Multistep (deis_m)
- [ ] Denoising Diffusion Implicit Models Inverse (ddim_i)
- [ ] Denoising Diffusion Implicit Models (ddim)
- [ ] Denoising Diffusion Probabilistic Models (ddpm)
- [ ] Diffusion Probabilistic Models Solver in Stochastic Differential Equations (dpm_sde)
- [ ] Diffusion Probabilistic Models Solver in Multistep Inverse (dpm_mi)
- [ ] Diffusion Probabilistic Models Solver in Multistep (dpm_m)
- [ ] Diffusion Probabilistic Models Solver in Singlestep (dpm_s)

**Tokenizer Type**
- [x] Byte-Pair Encoding (bpe) _(after 2024/07/03)_
- [x] Word Piece Encoding (wp) _(after 2024/05/27)_
- [ ] Sentence Piece Encoding (sp)  _[if necessary]_