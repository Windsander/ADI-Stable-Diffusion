# Changelog

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
