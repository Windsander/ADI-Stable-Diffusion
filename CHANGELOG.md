# Changelog

## [v2.0.0] - 2026-08-26

### Added
- **SD3.5-turbo (MMDiT) support**: triple-encoder orchestration (dual CLIP + T5-XXL via vendored libsentencepiece), MMDiT backbone slot with name-ordered input binding, 16-channel latent auto-detection, VAE `shift_factor` (0.0609) end-to-end, and the rectified-flow scheduler family (`flow_euler`). Verified at 1024px (s4/s8/cfg acceptance 6/6).
- **FLUX.1-schnell support**: 2x2 latent pack/unpack, `img_ids`/`txt_ids`/`guidance` name-binding, FLUX auto-detection via `img_ids`, INT64 rotary ids. Verified at 1024px 4-step (std=60.94).
- **SVD img2vid path**: new `euler_svd` scheduler (karras explicit boundaries, continuous timesteps, v-prediction; numpy-verified against diffusers), `UNetVideo` (5D tensors, per-frame CFG + guidance ramp), `ImageEncoder` (CLIP ViT-H vision tower), and CLI flags `--image-encoder --frames --fps --motion-bucket --noise-aug`; outputs a frame sequence. Smoke PASSED (first-frame std=79.14).
- **Runtime precision policy** `--precision auto|fp32|fp16`: probes available RAM at load time, estimates peak usage, and swaps component paths to a `<set>-fp16` sibling directory on memory-constrained machines, converting on demand via `sd/tools/onnx_fp16_convert.py` (ORT transformers float16 converter; path-based shape inference for >2GB models, LayerNorm kept fp32). fp16 is a memory-driven fallback, not an upper limit — high-RAM machines stay fp32.
- CI manual smoke gate (`workflow_dispatch`): builds linux x86_64 and runs the turbo smoke matrix with hard gates.

### Changed
- **ONNX Runtime 1.18.0 → 1.28.0** prebuilt packages. Handles the ORT 1.19+ package renames (`win-x64-gpu-cuda12` → `win-x64-gpu_cuda12`, same for linux) and the new dylib/soname scheme (`@rpath/libonnxruntime.1.dylib`, `libonnxruntime.so.1`). Local regression on macOS arm64 is bit-identical to the 1.18 baseline (turbo std=54.61, sd35 1024px std=47.53).
- osx-x86_64 / win-x86 prebuilt packages were discontinued upstream; those targets now fall back to the local source-build path with a configure-time warning, and their CI matrix legs are removed (macOS keeps arm64, Windows keeps x86_64 + arm64).

### Fixed
- `clitools` post-build dylib/soname normalization now rewrites both the ORT ≥1.19 names (`.1.dylib` / `.so.1`) and the ≤1.18 fully-versioned names to the unversioned library we ship (also fixes the Linux rewrite target missing the `lib` prefix) — previously the binary failed at startup with `Library not loaded: @rpath/libonnxruntime.1.dylib`.
- Smoke-matrix runner: comments inside a backslash-continuation silently dropped the sd35 resolution/steps arguments (cases ran at default size); fixed and re-verified at 1024px.
- SD3.5/FLUX plumbing: MMDiT name-ordered input binding (optimum SD3 declares timestep last), timestep dtype detection by name, `concat_sequence` for CLIP⊕T5, VAE decode scale validation for factors > 1, FLUX timestep /1000 (diffusers feeds sigma, not sigma×1000).

### Removed
- Linux .deb/.rpm packaging from the deploy chain (decision G3: supersedes v1.2 decision 0.4). The deb rules predate the ORT 1.19+ package rename/soname change and have no maintainer; Linux **build+run** support stays covered by the test-native ubuntu legs and the manual smoke job.

### Notes
- ORT execution-provider setup still uses the legacy `SessionOptionsAppendExecutionProvider*` APIs, which remain available in 1.28; migrating to the V2 provider-registration API is deferred until upstream announces legacy removal (untested-on-device migration judged riskier than the status quo).
- Not re-tested under ORT 1.28 in this window: sd35 8-step, FLUX, SVD (turbo + sd35 bit-match makes an engine-side behavior change unlikely; rerun if a full-coverage record is needed).

## [v1.2.0] - 2026-07-31

### Added
- **SD v2.x support (tested on SD v2.1-768)**: model acquisition via community mirror + optimum ONNX export chain (`sd/sd-base-model/onnx-sd-v21-768`); verified 768px v-prediction inference end-to-end.
- **SDXL support (tested on SDXL-turbo)**: dual text-encoder conditioning pipeline — `hidden_states[-2]` selection per encoder, feature-dim concat (768+1280→2048), pooled embedding from text_encoder_2, `time_ids` micro-conditioning; new CLI flag `--clip2`; `IOrtSDConfig.sd_modelpath_config` extended with `onnx_clip_2_path` (single ABI window for this release).
- Input-signature adaptation: UNet timestep and CLIP `input_ids` tensors now adapt to each model's declared dtype/rank (legacy int64 vs newer float/int64 exports), fixing silent pure-noise outputs with SD v2.x-era exports.

### Fixed
- `ModelBase::execute` previously swallowed ORT exceptions after logging, leaving preallocated zero outputs and decoding to pure noise; model input mismatches are now prevented by signature adaptation (`ModelBase::model_input_element_type`, `TensorHelper::cast`).
- ORT cxx `TypeInfo` dangling-view misuse that could hang model inspection.

### Notes
- SD v2.0 and SDXL-base were not tested (models not downloaded); architecture is identical to the tested variants.
- Linux .deb/.rpm packaging remains disabled in this window (decision 0.4); repair is scheduled with the ORT upgrade window.

## [v1.1.0] - 2026-07-30

### Added
- **UniPC completion**: full UniPC predictor-corrector math (λ-space exponential integrator + Lagrange interpolation); fixed the empty-stub segfault when selecting `unipc`.
- **Karras sigma schedule** (rho=7): new `--sigma [default/karras]` option and `AvailableSigmaType` (append-only enum); sequences verified value-identical to diffusers `use_karras_sigmas`.
- **DPM++ family**: `dpm_m` (DPM-Solver++ 2M), `dpm_s` (2S), `dpm_sde` (2nd-order midpoint SDE) ported from diffusers 0.39 and translated to the framework's EDM sample space (numerics verified against diffusers).
- **Legacy samplers**: `pndm` (prk+plms, leading spacing), `ipndm` (paper-form AB4 eps extrapolation + DDIM update; diffusers' ADM-grid variant shown unsuitable for SD), `deis_m` (log-rho Lagrange exponential integrator, up to 3rd order). All 14 schedulers are now usable.
- CLI help text synchronized (scheduler list, tokenizer list, `--sigma`).

### Fixed
- macOS arm64 architecture mapping in `auto_build.sh` (was mis-mapped to aarch64, breaking the toolchain).
- PNDM coordinate-space and initial-noise-alignment defects found during validation.

## [v1.0.1] - 2024-08-16

- Baseline: 8 schedulers (euler/euler_a/lms/lcm/heun/ddpm/ddim/unipc-stub), bpe/word_piece tokenizers, txt2img/img2img, CPU + provider scaffolding.
