# ADI 项目补充开发计划

> 基线版本：v1.0.1（main @ 0dcff5a，2024-08-16）
> 制定日期：2026-07-29
> 依据：对当前代码库的实证分析（非 README 声明）

---

## 0. 现状关键事实（规划依据）

| 事实 | 位置 | 对计划的影响 |
|---|---|---|
| v-prediction 已接入调度流程 | `source/scheduler/scheduler_base.cc:106` | SD v2.x 适配成本大幅降低 |
| sigma 在基类由 alphas_cumprod 插值生成 | `scheduler_base.cc:83-91` | Karras 策略有明确单一注入点 |
| 调度器/分词器均为注册表模式 | `scheduler_register.cc`、`tokenizer_register.cc` | 新采样器是纯增量，无侵入 |
| **unipc 是空壳 stub**：corrector/predictor 函数体均返回空向量，用户选择即 segfault（exit 139，2026-07-29 实证） | `scheduler_discrete_unipc.cc:51-63` | README 未勾选是**准确**的；补全实现列为阶段 1 首要任务 |
| CLI 帮助文本滞后（scheduler 只列 3 项、tokenizer 只写 bpe） | `clitools/main.cc:272-276` | 已随阶段 0.2 修正为 8 调度器 + bpe/word_piece |
| `IOrtSDConfig` 为按值传递的扁平 C 结构体 | `include/adi.h` | 任何字段增删 = ABI 破坏，需版本窗口集中处理 |
| CLI 预留 `img2vid` / `convert` 模式名 | `clitools/main.cc:341` | 视频/转换功能有既定坑位 |
| ORT 子模块停留在 2024 年（预编译包 1.17.3/1.18.0 时代） | `engine/` | 依赖升级是前置风险项 |
| Linux .deb/.rpm 打包 2024-08-09 起被禁用 | git 历史 | 需要正式决策而非搁置 |

---

## 阶段 0：基线修复与债务清理（阻塞项，约 1 周）

> 目标：恢复绿色构建基线，让文档与代码一致，做掉悬而未决的决策。

- **0.1 重建构建基线**：旧构建目录已清理（缓存指向失效路径），在 macOS arm64 上执行 `auto_build.sh` 全量构建，跑通 `sd/io-test` 的 txt2img / img2img 双模式冒烟
- **0.2 文档真实化**：
  - ~~README 勾选 uni_pc~~ **实证后撤回**：unipc 是空壳 stub（选择即 segfault），README 未勾选本来就准确，不做勾选
  - 已修正：CLI `--scheduler` 帮助文本（3 项 → 8 项）、`--tokenizer` 帮助文本（补 word_piece）
  - 清单与 `SchedulerType` 枚举（8 项）其余条目核对一致，无需改动
- **0.3 依赖体检与升级决策**：评估 onnxruntime 子模块升级到当前稳定版的兼容性（重点回归 CoreML / NNAPI / TensorRT / CUDA 四个 provider 路径）；检查 CI runner 镜像与 brew/choco 打包脚本可用性
- **0.4 Linux 打包正式决策**：修复 .deb/.rpm 流程，或从 deploy 工作流与 README 中正式移除——二选一，不再搁置
- **产出**：绿色构建 + 真实文档 + 两份决策记录（ORT 版本、Linux 打包）

### 阶段 0 执行记录（2026-07-29）

| 项 | 结果 |
|---|---|
| 0.1 构建基线 | ✅ 修复 `auto_build.sh` 架构映射 bug（macOS arm64 被错映射为 aarch64，触发 toolchain FATAL_ERROR）；debug 全量构建通过；txt2img / img2img 双模式冒烟产出正常（euler_a，sd-turbo，单步约 9s） |
| 0.2 文档真实化 | ✅ CLI 帮助文本已修正；**实证修正计划本身**：unipc 系空壳 stub 且 segfault，README 未勾选本就准确 |
| 0.3 ORT 决策 | **v1.2 窗口维持 1.18.0**：本地预编译包构建与推理全绿，升级评估与回归（四 provider 路径）并入 v2.0.0 前置工作，避免在 v1.2 窗口引入引擎变量 |
| 0.4 Linux 打包决策 | **v1.2 窗口维持禁用**：README 安装章节本就不含 Linux 包承诺，无需文档变更；.deb/.rpm 修复移交 v2.0.0 窗口与 ORT 升级一并处理 |

### 阶段 1 执行记录（进行中）

| 项 | 结果 |
|---|---|
| 1.0 unipc 补全（2026-07-29） | ✅ 完整实现 UniPC 预测-校正数学（λ 空间指数积分器 + Lagrange 插值，EDM 约定与 `scale()`/`find_predict_params_at` 一致），修复空壳 segfault。两个关键实证：①样本空间是 EDM x=x0+σ·ε 而非 VP 归一化（初版公式坐标系错误致纯噪声）；②末步 λ 跳变（Δλ≈3.5）需降阶保护（`SD_LAMBDA_JUMP_CAP=1.25`，推广 diffusers lower_order_final）。合成场仿真验证校正器精确度达 1e-11。验证矩阵：turbo t2i 1/4/8 步 ✓、i2i ✓、CFG7.5 ✓、v15 20步 ✓ |
| 参数坑位记录 | sd-v15 CLIP 需 `--dims 768`，sd-turbo 需 `--dims 1024`；用错会导致 CLIP/UNet 维度报错且**静默输出零张量**（图像纯噪声）。阶段 2 文档/校验需覆盖 |
| 1.1 Karras sigma 策略（2026-07-30） | ✅ 全链路落地：公开枚举 `AvailableSigmaType`（adi.h，追加式）→ 内部 `SigmaType` → `SchedulerBase::init()` karras 分支（ρ=7，σ→t 二分反查）→ CLI `--sigma`。sigma 序列与 diffusers `use_karras_sigmas` **逐值一致**（14.6146/3.1686/0.4469/0.0292 @4步）；turbo 4 步 euler_a/unipc + default 对照三组推理全绿。验证环境：conda base（torch 2.13 CPU + diffusers 0.39 已装，供阶段 1.2 参考比对） |
| 1.2 dpm_m（2026-07-30） | ✅ DPM-Solver++ 2M（midpoint）移植自 diffusers 0.39 源码并翻译到 EDM 样本空间（坐标映射 x_vp=α·x_edm，与 diffusers 数值对比误差 2.4e-07）；末步零 σ 走 order-1（同 diffusers lower_order_final+zero）。turbo 4 步 ✓、v15 20 步 ✓、**DPM++ 2M Karras 组合** ✓。diffusers 关键机制备查：末步用 first_order_update 规避 h=∞（`final_sigmas_type=="zero"` 条件分支） |

## 阶段 1：采样器与 Sigma 策略补全（v1.1.0，纯增量低风险）

> 目标：14 个采样器全部可用（含 unipc 补全）+ Karras 策略，覆盖社区主流用法。

按"社区使用率 × 实现成本"排序：

0. **补全 unipc（首要，修复线上崩溃）**：当前 `get_unified_correction` / `get_unified_prediction` 均为空壳返回 `{}`，调用方解引用空向量导致 segfault。需按 arxiv 2302.04867 实现 Bh corrector + multistep predictor 数学，复用既有 `history_dnoise` / `last_samples_` 结构与 `find_predict_params_at` 的 c_skip/c_out 参数
1. **Karras sigma 策略**：`SchedulerConfig` 追加 sigma 策略字段（枚举尾部追加，保持既有枚举值不变），注入点为 `scheduler_base.cc` 的 sigma 生成处
2. **DPM++ 家族（优先）**：`dpm_m`（multistep）→ `dpm_s`（singlestep）→ `dpm_sde` —— 当前社区最常用
3. **遗留补全**：`pndm` → `ipndm` → `deis_m`

每个采样器的固定动作清单（注册表模式保证零侵入）：
- 新增 `source/scheduler/scheduler_discrete_<name>.cc`
- `scheduler_register.cc` 注册 + `SchedulerType` 枚举**尾部追加**
- CLI `--scheduler` 帮助文本同步
- `sd/io-test` 增加金图对比用例

**版本纪律**：本阶段集中完成所有 `IOrtSDConfig` / 枚举扩展，一次性发布 v1.1.0，避免多次 ABI 波动。

## 阶段 2：模型代际扩展（v1.2.0 → v2.0.0，按成本递增）

| 顺序 | 目标 | 主要差异工作 | 已知便利 | 建议版本 |
|---|---|---|---|---|
| 2.1 | SD v2.0 / v2.1 | 768 分辨率链路验证、OpenCLIP-H 分词（BPE 变体）、penultimate hidden layer 抽取 | **v-prediction 已就绪** | v1.2.0 |
| 2.2 | SDXL / SDXL-turbo | 双文本编码器（CLIP-L + OpenCLIP-G）、pooled embedding、time-id 微条件注入、模型路径配置扩展 | 分词仍走 BPE 路径，sp 非必需 | v1.2.0 |
| 2.3 | SD v3 | MMDiT 新架构、T5-XXL 分词（**SentencePiece 从 "if necessary" 提为必须**）、rectified-flow 调度（新调度器家族，非 discrete 扩展） | — | v2.0.0 |
| 2.4 | SVD 视频 | 启用 CLI 预留的 `img2vid` 模式；时序模块 + 帧批处理；显存/内存策略 | 坑位已留 | v2.0.0 |

**架构预判**：2.3/2.4 涉及模型单元体系（`source/units/`）的结构性扩展，与 2.1/2.2 的"配置级扩展"不同，因此拆分为 v2.0.0 大版本。

## 阶段 3：工程化加固（贯穿各阶段）

- **测试**：per-scheduler 金图回归 CI 化（复用 `sd/io-test/comparisons` 机制）；test-native / test-cross 工作流纳入版本门禁
- **版本纪律**：`adi.h` 公共头变更 → 次版本号起步；已实现的 union version check 接入 CI 强制校验
- **分发链**：SDXL/SD3 模型体积显著增大，`sd/auto_prepare_sd_models.sh` 需扩展模型清单与校验；`ort_sd_py_imp.py` 作为转换/验证参考实现同步演进

---

## 优先级矩阵

```
高价值 ▲
       │  0.2 文档同步      2.1 SD v2.x
       │  0.1 构建基线      1.2 DPM++ 家族
       │  1.1 Karras
       │
       │  0.3 ORT 升级      2.2 SDXL
       │  0.4 Linux 打包    2.3 SD v3 / 2.4 SVD
       └──────────────────────────► 高成本
```

**推荐执行顺序**：阶段 0（全部）→ 1.1 + 1.2（v1.1.0 核心卖点）→ 2.1（低成本验证代际扩展管线）→ 2.2 → 1.3（补尾）→ 2.3/2.4（v2.0.0 独立规划）

## 风险登记

1. **ORT 大版本升级**可能破坏既有 provider 集成路径 → 阶段 0.3 先出兼容性报告再定升级窗口
2. **ABI 破坏面**：`IOrtSDConfig` 按值传递，字段变更即破坏 → 所有结构扩展收敛到单一版本窗口（v1.1.0 / v2.0.0 各一次）
3. **模型可得性**：SDXL/SD3 官方 .onnx 格式供应不稳定 → 可能需要先补自有 PyTorch→ONNX 转换链（`ort_sd_py_imp.py` 扩展）再谈推理支持
4. **搁置成本**：仓库已停滞约 23 个月，外部生态（diffusers 调度器实现、模型格式）持续演进 → 阶段 1/2 的参考实现选型需在动手前重新核对上游现状
