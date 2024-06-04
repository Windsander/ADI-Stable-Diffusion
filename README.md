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

- **First, prepare ORT(ONNXRuntime) environment**
 
    by simply executing auto_script:
```bash

```

    or download ONNXRuntime 1.73.3 manually:


Below is What actually happened in 1-step img2img inference in Latent Space (Skip All Models):
![sd-euler_a-1step-latent-example.png](sd%2Fio-examples%2Fsd-euler_a-1step-latent-example.png)

And now, you can have a try~ (0w0 )

## Development Progress (latest):

**Basic Pipeline Functionalities (Major):**
- [x] [SD] Stable-Diffusion 
- [ ] [SDXL] Stable-Diffusion-XL
- [ ] [SVD] Stable-Video-Diffusion

**Scheduler Strategy**
- [x] Discrete/Method Default (discrete)
- [ ] Karras (karras)

**Scheduler Method**
- [x] Euler (euler)
- [x] Euler Ancestral (euler_a)
- [x] Laplacian Pyramid Sampling (lms)
- [ ] Latent Consistency Models (lcm)
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
- [ ] Byte-Pair Encoding (bpe)
- [x] Word Piece Encoding (wp)
- [ ] Sentence Piece Encoding (sp)  _[if necessary]_