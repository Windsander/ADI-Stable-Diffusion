# Development Progress & Roadmap

> Split out from README.md after v2.0.0 (the list outgrew the front page).
> Audited against the **v2.0.0** codebase (2026-08); historical plan context in [PLAN-supplement.md](PLAN-supplement.md).

---

## ✅ Delivered Milestones

### Basic Pipeline Functionalities (Major)

- [x] **[SD_v1] Stable-Diffusion (v1.0 ~ v1.5, turbo)** _(after 2024/06/04 tested)_
    - **v1.0** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion): Initial version ✅
    - **v1.1** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-1): Improved image quality and generation speed ✅
    - **v1.2** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-2): Further optimized generation effects ✅
    - **v1.3** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-3): Added more training data ✅
    - **v1.4** [(HuggingFace)](https://huggingface.co/CompVis/stable-diffusion-v-1-4): Enhanced image generation diversity ✅
    - **v1.5** [(HuggingFace)](https://huggingface.co/runwayml/stable-diffusion-v1-5): Final optimized version ✅
    - **turbo** [(HuggingFace)](https://huggingface.co/stabilityai/sd-turbo): Community-driven optimized version, faster and efficiency ✅

- [x] **[SD_v2] Stable-Diffusion (v2.0, v2.1)** _(v2.1 768px v-prediction after 2026/07/31 ✅tested)_
    - **v2.0** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-2): Significant improvements in image quality and generation efficiency _(untested; identical architecture/pipeline to the verified v2.1)_
    - **v2.1** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-2-1): Further optimized model stability and generation effects ✅ _(official repo gated since 2025 — acquired via `sd2-community` mirror + optimum ONNX export chain, see `sd/auto_prepare_sd_models.sh`)_

- [x] **[SD_v3.x] Stable-Diffusion 3 / 3.5 (MMDiT era)** _(v2.0.0 — SD3.5-turbo 1024px ✅tested 2026/08; MMDiT slot + triple-encoder orchestration + T5-XXL/sp + flow_euler all landed)_
    - **v3.0** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-3-medium): First MMDiT release (2024/06) _(superseded by v3.5; no longer the primary target)_
    - **v3.5 (Large / Large-Turbo / Medium)** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-3.5-large): Stability flagship since 2024/10 and the open-weights mainstream of 2026. Landed: MMDiT transformer unit (structural `source/units/` extension), triple text encoders (CLIP-L + OpenCLIP-G + **T5-XXL via SentencePiece**), rectified-flow scheduling

- [x] **[SDXL] Stable-Diffusion-XL** _(SDXL-turbo after 2026/07/31 ✅tested)_
    - **SDXL** [(HuggingFace)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): Experimental version for larger-scale models and higher-resolution image _(untested; same pipeline as SDXL-turbo — dual-encoder `--clip2`, pooled embedding, time-ids micro-conditioning all landed in v1.2.0)_
    - **SDXL-turbo** [(HuggingFace)](https://huggingface.co/stabilityai/sdxl-turbo): Community-driven optimized version, faster and efficiency ✅

- [x] **[FLUX-class] Flow-Matching Models** _(v2.0.0 — FLUX.1-schnell 1024px 4-step ✅tested 2026/08; 2x2 pack/unpack + img_ids/txt_ids/guidance name-binding; dev variant untested)_
    - **FLUX.1 [dev] / [schnell]** [(HuggingFace)](https://huggingface.co/black-forest-labs/FLUX.1-dev): The community-favorite open-weights family of 2026. Shares the MMDiT + rectified-flow + T5-XXL stack with SD3.5; extras: single-file checkpoint conversion chain, guidance-distilled variants, non-commercial license review

- [x] **[SVD] Stable-Video-Diffusion** _(v2.0.0 — img2vid mode live 2026/08: euler_svd scheduler + UNetVideo + CLIP ViT-H ImageEncoder, 14-frame fp32 smoke ✅tested)_
    - **SVD** [(HuggingFace)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid): Version specifically for video generation and editing _(F=14 fixed at export; F=25 XT tier needs a separate export; 25-step quality tier not yet run)_

### Scheduler Abilities

- [x] **Strategy**
    - [x] Discrete/Method Default (discrete) _(after 2024/05/22)_
    - [x] Karras (karras, ρ=7) _(after 2026/07/30 ✅tested — sigma sequence value-identical to diffusers `use_karras_sigmas`)_

- [x] **Sampling Methods (14/14 complete since v1.1.0)**
    - [x] Euler (euler) _(after 2024/06/04 ✅tested)_
    - [x] Euler Ancestral (euler_a) _(after 2024/05/24 ✅tested)_
    - [x] Laplacian Pyramid Sampling (lms) _(after 2024/07/09 ✅tested)_
    - [x] Latent Consistency Models (lcm) _(after 2024/07/04 ✅tested)_
    - [x] Heun's Predictor-Corrector (heun) _(after 2024/07/08 ✅tested)_
    - [x] Unified Predictor-Corrector (unipc) _(completed after 2026/07/29 ✅tested — was an empty-stub segfault before v1.1.0)_
    - [x] Pseudo Numerical Diffusion Model Scheduler (pndm) _(after 2026/07/30 ✅tested)_
    - [x] Improved Pseudo Numerical Diffusion Model Scheduler (ipndm) _(after 2026/07/30 ✅tested — paper-form AB4 + DDIM update; diffusers' ADM-grid variant verified unsuitable for SD)_
    - [x] Diffusion Exponential Integrator Sampler Multistep (deis_m) _(after 2026/07/30 ✅tested)_
    - [x] Denoising Diffusion Implicit Models (ddim) _(after 2024/07/12 ✅tested)_
    - [x] Denoising Diffusion Probabilistic Models (ddpm) _(after 2024/07/09 ✅tested)_
    - [x] Diffusion Probabilistic Models Solver in Stochastic Differential Equations (dpm_sde) _(after 2026/07/30 ✅tested)_
    - [x] Diffusion Probabilistic Models Solver in Multistep (dpm_m) _(after 2026/07/30 ✅tested)_
    - [x] Diffusion Probabilistic Models Solver in Singlestep (dpm_s) _(after 2026/07/30 ✅tested)_

- [x] **Flow-Matching / Rectified-Flow Family** _(v2.0.0 — new `FlowSchedulerBase` family, separate from `scheduler_discrete_*`)_
    - [x] Flow Euler discrete _(v2.0.0 ✅tested — default sampler of SD3.5 / FLUX-class; numpy-verified against diffusers)_

### Tokenizer Type

- [x] Byte-Pair Encoding (bpe) _(after 2024/07/03 ✅tested — covers CLIP-L / OpenCLIP-G/H)_
- [x] Word Piece Encoding (word_piece) _(after 2024/05/27 ✅tested; available via internal registry & CLI)_
- [x] Sentence Piece Encoding (sp) _(v2.0.0 ✅tested — vendored libsentencepiece static build, official library; T5-XXL for SD3.5 / FLUX)_

### Engineering & Distribution

- [x] Smoke-matrix runner scripted (`sd/io-test/run_smoke_matrix.sh`; hard gates: ORT-exception count, output size, flat-pixel check) _(local 25/25 green @ v2.0.0, 2026/08/26)_
- [x] Release chain hardened (CHANGELOG-driven auto-publish → auto-deploy; 9 platform artifacts on release-v2.0.0)
- [x] ONNXRuntime engine upgrade _(1.18.0 → 1.28.0 prebuilt; gpu-cuda12 → gpu_cuda12 renames; osx-x86_64 / win-x86 prebuilts discontinued upstream → local-build fallback; regression bit-match with the 1.18 baseline, 2026/08/26)_
- [x] Linux .deb/.rpm packaging — **decision: removed, not repaired** _(v2.0.0 decision G3; Linux build+run stays CI-covered, plain tarballs still shipped)_
- [x] CI governance: aggregate gate jobs (`native-gate` / `cross-gate`) as stable branch-protection checks; manual-only inference smoke workflow (`smoke.yml`)
- [x] Runtime precision policy `--precision auto|fp32|fp16` _(v2.0.0 — RAM-probe driven; fp16 derived on demand via `sd/tools/onnx_fp16_convert.py`)_

---

## 🧭 Next Phase Candidates (v2.1.0+)

> Priorities are proposals, not commitments. Items marked 🔬 need hardware/data we don't currently have in CI.

### Quality & Verification
- [ ] **Golden-image regression in CI** — today `test-native`/`test-cross` are compile-only; wire the smoke matrix (or a cheap slice) plus golden-image comparison into workflows so numerics can't silently drift
- [ ] **fp16 quality validation** — `--precision fp16` passes smoke gates, but no systematic visual-diff vs fp32 goldens (esp. SD3.5/FLUX at full steps)
- [ ] **CUDA / TensorRT device regression** 🔬 — provider paths compile but have never been validated on real GPU hardware; needs a GPU runner or manual rig

### Model Coverage & Fidelity
- [ ] **SD3.5 Large / Medium full-step validation** — only SD3.5-turbo at 4/8 steps is verified; run the quality tiers (20~28 steps, cfg>1) on the non-distilled variants
- [ ] **FLUX.1-dev** — non-commercial license review + guidance-distilled chain (shares the schnell pipeline)
- [ ] **SVD upgrades** — F=25 (XT tier) export; 25-step quality tier; motion-bucket/fps sweeps for output quality tuning
- [ ] **Flow Heun / higher-order flow solvers** — flow_euler only today; add when a model family demands it

### Engine & API
- [ ] **ORT provider V2 API migration** — deferred from v2.0.0 (legacy `SessionOptionsAppendExecutionProvider*` still fine in 1.28); migrate before upstream removes legacy APIs
- [ ] **Public enum parity** — expose `word_piece` (and any new v2.0.0 additions) in the public `AvailableTokenizerType` C enum at the next ABI window
- [ ] **ControlNet / safety-checker integration** — fields reserved in `IOrtSDConfig` (`onnx_control_net_path` / `onnx_safty_path`), currently not available

### Developer Experience
- [ ] **One-command model acquisition** — `sd/auto_prepare_sd_models.sh` covers several sets; extend to SD3.5/FLUX/SVD export chains (currently manual + `debug/export_svd_onnx.py`)
