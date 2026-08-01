# ADI Architecture

> Audited against the **v1.2.0** codebase (2026-08). This document describes the
> current structure and the design decisions that hold it together — derived
> from the code, not from aspirational docs. For the forward-looking plan see
> [PLAN-supplement.md](PLAN-supplement.md); for per-release changes see
> [CHANGELOG.md](CHANGELOG.md).

---

## 1. Overview

**Agile Diffusers Inference (ADI)** is a C++ library with a CLI front-end for
Stable-Diffusion-family inference on top of ONNXRuntime (ORT). The design goals:

- **Small, portable packages** — a handful of ORT sessions (CLIP / UNet / VAE)
  orchestrated by plain C++17, no Python or framework runtime at inference time.
- **Model-format compatibility** — everything goes through `.onnx`; acquisition
  and conversion live outside the runtime (optimum export chain, see
  `sd/auto_prepare_sd_models.sh`).
- **Provider-level acceleration** — CPU baseline with CoreML / NNAPI / TensorRT /
  CUDA execution providers selectable per deployment.
- **Numerical fidelity to diffusers** — scheduler math is ported from and
  validated against HuggingFace `diffusers` (trajectory-level cross-checks,
  max deviation in the 1e-5 ~ 1e-7 range, see §12).

## 2. Repository Layout

```
ADI-Stable-Diffusion/
├── include/adi.h            # The ONLY public header: pure C ABI surface
├── outlet/adi.cc            # ABI bridge: C struct -> internal C++ config, context lifecycle
├── source/                  # Internal C++ implementation (not ABI-stable)
│   ├── adi_context.cc       # Pipeline orchestration (OrtSD_Context)
│   ├── base/                # Tensor helpers, ORT executor, session/provider plumbing
│   ├── units/               # Model units: Clip / UNet / VAE (+ ModelBase, wrapper)
│   ├── scheduler/           # SchedulerBase + 14 discrete schedulers + registry
│   ├── tokenizer/           # BPE & WordPiece tokenizers + registry (nlohmann json)
│   ├── amon/                # Logging / exceptions / stats infrastructure
│   └── apex/                # Internal config structs (SchedulerConfig, TokenizerConfig, ...)
├── clitools/                # CLI front-end (main.cc), stb image backends, example scripts
├── engine/                  # ORT: 2024-era git submodule + per-platform prebuilt packages
│   │                        #   (1.17.3 / 1.18.0), auto_prepare_engine_env.sh
├── apex/                    # CMake helper modules (colors, ort-env, static, utils)
├── apex-toolchain/          # Cross-compile toolchains: android / darwin / linux / windows
├── sd/                      # Model zoo + io-test smoke matrix + io-examples latent visuals
│   ├── sd-base-model/       #   onnx-sd-turbo / onnx-sd-v15 / onnx-sd-v21-768 / onnx-sdxl-turbo
│   ├── io-test/             #   run_smoke_matrix.sh + smoke outputs + comparisons/
│   └── auto_prepare_sd_models.sh  # Model acquisition & optimum export chains
├── debug/                   # diffusers cross-validation scripts & golden references
│                            #   (NOT under version control — see §14)
├── .github/workflows/       # test-native / test-cross / auto-deploy / auto-publish
├── CMakeLists.txt           # Root build (options in §10)
├── auto_build.sh            # One-shot local build driver
├── auto_deploy.sh           # Local deploy/packaging driver
└── ort_sd_py_imp.py         # Python reference implementation (conversion/validation aid)
```

## 3. Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Consumers:  CLI (clitools/)  |  any C/C++ host via adi.h    │
├─────────────────────────────────────────────────────────────┤
│  Public C ABI          include/adi.h                         │
│  - 6 entry points      generate/init/prepare/inference/…     │
│  - IOrtSDConfig        flat, pass-by-value (ABI windows, §4) │
├─────────────────────────────────────────────────────────────┤
│  ABI bridge            outlet/adi.cc                         │
│  C struct → OrtSD_Config, context new/delete                 │
├─────────────────────────────────────────────────────────────┤
│  Orchestration         source/adi_context.cc                 │
│  OrtSD_Context: owns units, caches prompt embeddings,        │
│  mutex-guarded prepare()/inference()                         │
├──────────────┬──────────────────────┬───────────────────────┤
│  units/      │  scheduler/          │  tokenizer/           │
│  Clip ×2     │  SchedulerBase       │  registry → BPE / WP  │
│  UNet        │  + 14 discrete impls │                       │
│  VAE ×2      │  + sigma strategies  │                       │
├──────────────┴──────────────────────┴───────────────────────┤
│  base/                 TensorHelper / ONNXRuntimeExecutor    │
├─────────────────────────────────────────────────────────────┤
│  ONNXRuntime           engine/ (prebuilt 1.17.3/1.18.0       │
│                        or submodule build)                   │
│  Providers: CPU · CoreML · NNAPI · TensorRT · CUDA           │
└─────────────────────────────────────────────────────────────┘
```

The internal code lives in nested namespaces `onnx::sd::{base, context, units,
scheduler, tokenizer, amon}`. Note: the outer `onnx` namespace collides with
ONNX's own — flagged as debt (§14).

## 4. Public API & ABI Discipline

`include/adi.h` exposes exactly six functions:

```c
void     generate_context(IOrtSDContext_ptr*, IOrtSDConfig);  // create + configure
void     released_context(IOrtSDContext_ptr*);                // destroy
void     init(ctx);        // open all ORT sessions (5 or 6 models)
void     prepare(ctx, positive, negative);  // tokenize + embed prompts (cached)
IO_IMAGE inference(ctx, IO_IMAGE);          // one full diffusion pass
void     release(ctx);     // close sessions, free unit resources
```

**ABI constraints (load-bearing):**

- `IOrtSDConfig` is a **flat struct passed by value**. Adding/removing a field
  changes the ABI — all struct/enum extensions are batched into a single
  *version window* per release (v1.1.0 added the sigma-strategy enum; v1.2.0
  added `onnx_clip_2_path`). Two windows per major line, never drip-fed.
- Public enums are **append-only**; existing numeric values never move.
- `CURRENT_ADI_VERSION` ("v1.2.0") is the single version source.
- Reserved-but-unwired fields exist deliberately: `onnx_control_net_path`,
  `onnx_safty_path` (ControlNet / safety checker slots).

**Known gap:** the internal tokenizer registry supports WordPiece, and the CLI
exposes `--tokenizer word_piece`, but the public enum `AvailableTokenizerType`
has WP commented out — C-ABI consumers cannot select it today (§14).

## 5. Inference Pipeline

```
prepare():                                   inference(image?):
  positive ──► Clip ──► hidden [1,77,768]      image ──► convert_images ──► [1,3,H,W]
                 │(SDXL: hidden[-2])             │
  positive ──► Clip2 ─► hidden [1,77,1280]     ▼
                 │(SDXL: hidden[-2], pooled)   VAE-encoder ──► latent [1,4,H/8,W/8]
  concat_last_dim(768+1280→2048)               │
  pooled from text_encoder_2                   ▼
        │                                      UNet loop (steps):
        ▼                                        CFG: ε = neg + g·(pos − neg)
  ort_remain {embed±, pooled±}  ────────────►    scheduler.step() per t
        (cached across inferences)             │
                                               ▼
                                               VAE-decoder (×1/0.18215 or /0.13025)
                                               │
                                               ▼
                                               convert_result ──► IO_IMAGE
```

- `prepare()` and `inference()` are serialized by one mutex — a context may be
  reused across images without re-embedding prompts.
- txt2img is img2img with zero input (`convert_images` returns an empty tensor
  for null data; UNet seeds from pure noise instead).
- Batch size is fixed at 1 (`convert_result` rejects N>1).

## 6. Scheduler Subsystem

### 6.1 The unifying decision: EDM sample space

All schedulers operate in **EDM convention** `x = x0 + σ·ε`, not the
VP-normalized space most papers/diffusers use. Every ported algorithm is
algebraically translated into this space at port time (VP formulas degenerate
to σ-space updates; e.g. PNDM eq.(9) becomes `x + (σ_prev−σ_ref)·ε` with
RK/AB-extrapolated ε). This gives one coordinate system for:

- `SchedulerBase`: sigma generation from `alphas_cumprod` interpolation
  (single injection point for sigma strategies — Karras ρ=7 lands here, with
  σ→t binary inversion),
- the predictor contract (`epsilon` / `v_prediction` / `sample`),
- the base API: `mask()` (initial noise), `scale()` (add noise to step σ),
  `time()` (per-model timestep conditioning), `step()` (one update).

### 6.2 Extension contract

Schedulers register by name + enum (**append-only**) and implement
`execute_method()` plus optionally `correction_steps()` (for multi-evaluation
structures: heun-style doubling, DPM++ 2S midpoint pairs, SDE midpoint slots).
Adding a sampler touches exactly three places: new
`scheduler_discrete_<name>.cc`, registry entry, CLI help text.

### 6.3 Inventory (14/14 since v1.1.0)

| Family | Schedulers | Notes |
|---|---|---|
| Basic | euler, euler_a, lms, heun | original 2024 set |
| Consistency | lcm | few-step distilled models |
| Classic | ddpm, ddim | leading/trailling spacing handled in base |
| Predictor-corrector | unipc | full λ-space integrator + Lagrange interpolation (was an empty-stub segfault pre-v1.1.0; final-step λ-jump needs order capping) |
| DPM++ | dpm_m (2M), dpm_s (2S), dpm_sde | ported from diffusers 0.39, translated to EDM space; final σ=0 forces order-1 |
| Legacy | pndm, ipndm, deis_m | ipndm is **paper-form** (AB4 + DDIM update): diffusers' ADM-grid variant is verified unsuitable for SD checkpoints |

Sigma strategies: `default` (training-grid interpolation), `karras` (ρ=7,
value-identical to diffusers `use_karras_sigmas`). Karras composes with every
sampler that consumes the base sigma grid.

**Not yet present:** the flow-matching / rectified-flow family (Flow Euler et
al.) required by SD3.5 / FLUX-class models — a new scheduler paradigm scheduled
for the v2.0.0 window, not another `scheduler_discrete_*` entry.

## 7. Tokenizer Subsystem

Registry-based like the schedulers. **BPE** (CLIP-style vocab.json + merges.txt,
attention weighting via `(prompt)` / `[prompt]` factors) covers every supported
model so far — CLIP-L and OpenCLIP-G/H are both BPE variants. **WordPiece**
exists internally (`TOKENIZER_WORD_PIECE`, `WPTokenizer`) and via CLI, but is
absent from the public C enum (§4 gap). **SentencePiece** is the next-required
addition: the T5-XXL text encoder of SD3.5 / FLUX-class models mandates it, so
its status moved from *if necessary* to *required* for v2.0.0.

## 8. Model Units

`ModelBase` owns the ORT session lifecycle (`init` / `release`) and two
load-bearing mechanisms added in v1.2.0:

- **Input-signature adaptation** — `model_input_element_type` + `TensorHelper::cast`
  adapt each input tensor to the model's declared dtype/rank (legacy int64
  timesteps vs newer float scalars; int32 vs int64 token ids). This is what lets
  one binary drive both 2023-era and 2025-era optimum exports.
- **Fail-loud execution** — `execute()` no longer swallows ORT exceptions into
  preallocated zero outputs (the historical "silent pure-noise" amplifier);
  the smoke matrix gates on exception count (§12).
- ORT C++ `TypeInfo` objects are kept alive for the duration of shape queries
  (a dangling-view misuse previously corrupted/hung model inspection).

| Unit | Responsibility | SDXL extensions (v1.2.0) |
|---|---|---|
| Clip | tokenize → embed prompts | `use_penultimate` (hidden_states[-2], no final_layer_norm), pooled-output capture; dual instances with feature-dim concat (768+1280→2048) |
| UNet | denoising loop + CFG | 5-input signature detection: binds `text_embeds` (pooled) + `time_ids` {1,6} = [H,W,0,0,H,W] micro-conditioning |
| VAE | encode/decode pixels↔latents (÷8 spatial, 4ch) | decode scaling via config (0.18215 SD1/2 vs 0.13025 SDXL) |

## 9. Execution Providers & Engine

`ONNXRuntimeExecutor` maps `AvailableExecutionType` to ORT providers;
CPU is always appended as final fallback. `EXECUTOR_GPU_AUTO` stacks
TensorRT → CUDA → CoreML → NNAPI in that order. Sessions run with
`ORT_PARALLEL` execution mode and `ORT_ENABLE_ALL` graph optimization.

The engine itself is **prepared at build time**, not committed: per-platform
prebuilt packages (1.17.3 / 1.18.0) under `engine/`, or the (2024-era)
`onnxruntime` submodule for from-source builds (`ORT_COMPILED_ONLINE` /
`ORT_COMPILED_HEAVY`). An ORT upgrade with a 4-provider regression pass is
scheduled for the v2.0.0 window (§14).

## 10. Build System

- **CMake** root (`CMakeLists.txt`, C++17) with option switches:
  `ORT_COMPILED_ONLINE/HEAVY`, `ORT_BUILD_COMMAND_LINE/COMBINE_BASE/SHARED_ADI/
  SHARED_ORT`, `ORT_ENABLE_{TENSOR_RT,CUDA,COREML,NNAPI}`, `ADI_AUTO_INSTALL`.
  Provider defaults are chosen per platform.
- **apex/**: reusable CMake modules; **apex-toolchain/**: cross toolchains for
  android / darwin / linux / windows (macOS arm64 mapping fixed in v1.1.0).
- **auto_build.sh**: one-shot driver (`--platform --build-type --arch-abi
  --options ...`); CI forces the Ninja generator and pins MSVC/NDK quirks.

## 11. CLI Front-end

`clitools/main.cc` (~850 lines): full argument surface mirroring
`IOrtSDConfig` (models, scheduler, sigma, tokenizer, guidance, seed, sizes),
image IO via stb. Modes: `txt2img`, `img2img` — with `img2vid` / `convert`
names **reserved** for the SVD and conversion roadmap items. Example scripts
under `clitools/examples/`; README carries verified per-model invocations
(sd-turbo, sd-v2.1-768, sdxl-turbo, Karras combinations).

## 12. Testing & Validation

- **Smoke matrix** (`sd/io-test/run_smoke_matrix.sh`, 19 quick / 24 full cases):
  hard gates on (1) ORT exception count in logs, (2) output existence/size,
  (3) flat-pixel detection (std < 5 ⇒ zero-latent symptom). Aesthetics are left
  to `comparisons/` golden images. Local quick pass: 19/19 (2026-07-31).
- **diffusers cross-validation** (`debug/`, torch 2.13 + diffusers 0.39):
  synthetic-field trajectory checks (max dev 2.7e-06 for PNDM), sigma-sequence
  value equality for Karras, real-pipeline reference images per sampler/model
  combination. **These assets are currently untracked** (§14).
- **CI**: `test-native` / `test-cross` compile matrices (incl. Windows MSVC and
  Android-on-Windows hardening, v1.2.0). Golden-image inference regression is
  not yet wired into workflows.

## 13. Release & Distribution

- `auto-publish.yml`: CHANGELOG-driven GitHub release (gh CLI +
  `softprops/action-gh-release@v2`; retired node12 actions replaced in v1.2.0).
- `auto-deploy.yml`: platform packages → brew formula / choco nupkg
  (artifact actions v2→v4; `update_homebrew_formula` artifact-download bug
  fixed). `deploy_linux` is **formally disabled** (decision 0.4) pending the
  .deb/.rpm repair bundled with the ORT upgrade.
- Published packages currently v1.0.1; the v1.1.0/v1.2.0 packages ride the
  `release/release-v*` branch chain.

## 14. Known Limitations & Technical Debt

| # | Item | Disposition |
|---|---|---|
| 1 | `IOrtSDConfig` pass-by-value ⇒ any field change breaks ABI | Version-window discipline; next window v2.0.0 |
| 2 | WordPiece missing from public `AvailableTokenizerType` enum | Expose at next ABI window |
| 3 | Internal `namespace onnx` collides with ONNX's own | Rename in v2.0.0 refactor |
| 4 | ORT engine stale (prebuilt 1.17.3/1.18.0, 2024-era submodule); 4-provider paths unregressed | Upgrade + regression in v2.0.0 window |
| 5 | Smoke/golden regression not in CI (`test-native` is compile-only) | Wire matrix into workflows |
| 6 | Linux .deb/.rpm packaging disabled since 2024-08 | Repair with ORT upgrade (decision 0.4) |
| 7 | `debug/` validation assets untracked — porting rationale lives only locally | Curate and commit |
| 8 | Batch fixed at 1; `convert_images` hard-codes 3-channel skip | Revisit with pipeline batching |
| 9 | ControlNet / safety-checker fields reserved but unwired | Post-v2.0.0 evaluation |

## 15. Roadmap Pointers

- **v2.0.0** (structural, per PLAN-supplement 2.3/2.4): SD3.5 (MMDiT unit,
  triple text encoders with T5-XXL ⇒ SentencePiece, rectified-flow scheduling),
  SVD `img2vid`, ORT upgrade, Linux packaging. FLUX-class support is a
  follow-on candidate sharing the same MMDiT+flow+T5 stack.
- The detailed phase plan, execution logs, and risk register live in
  [PLAN-supplement.md](PLAN-supplement.md).
