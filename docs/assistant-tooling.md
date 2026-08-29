# ADI 项目助手配置说明

本目录包含 ADI-Stable-Diffusion 项目的「项目助手代理」配置，旨在利用 GitHub 原生能力（减少额外 token 消耗）实现：

1. **技术问题应答**（Discussions + Copilot）
2. **建议 / Bug 收集**（Issue 模板 + 自动分类 + 关联 Issue 创建）
3. **每周汇总周报**（Discussions，季度一份、期间追加）

## 目录结构

```
.github/
├── ISSUE_TEMPLATE/          # Issue 创建表单 (YAML format)
│   ├── config.yml           # 关闭空白 Issue，引导至 Discussions / 文档
│   ├── bug_report.yml       # 🐛 Bug 报告
│   ├── feature_request.yml  # ✨ 特性请求（新模型、工程改进等）
│   ├── model_support_request.yml   # 🤖 新模型支持请求
│   ├── webbridge_enhancement.yml   # 🌐 Webbridge 增强
│   └── tech_question.yml           # ❓ 技术问题
├── copilot-instructions.md  # Copilot 仓库级自定义提示词（GitHub 约定的标准文件名）
├── workflows/               # GitHub Actions
│   ├── issue-classifier.yml # Issue 自动分类器
│   ├── weekly-report.yml    # 周报生成（Discussions，GraphQL API）
│   ├── auto-deploy.yml      # （已有）自动部署
│   └── ...
└── CODEOWNERS               # （已有）负责人配置
```

## 使用方式

### 1. 创建 Issue

访问 `https://github.com/Windsander/ADI-Stable-Diffusion/issues/new/choose` 选择合适的模板：

- **Bug Report** — 报告错误
- **Feature Request** — 请求新功能（新模型、工程改进等）
- **Model Support Request** — 请求支持特定模型
- **Webbridge Enhancement** — 提出 Webbridge 改进建议
- **Technical Question** — 提问技术问题

### 2. 技术问答

在 **Discussions** 中提问，Copilot 可以基于 `.github/copilot-instructions.md` 中的上下文回答。你也可以直接在 Issue/Discussions 中 @GitHub Copilot。

### 3. 周报

每周六上午 9:00 北京时间（UTC 1:00），如果本周有新增内容，工作流自动在 Discussions 中创建或追加周报。季度内使用同一个 Discussion 帖，季度末自然归档。

周报位置示例：`[周报] ADI 项目 2026-Q3 周报汇总`

## 自动分类

新创建的 Issue 会自动打上类型和领域标签，例如：

- `type/bug` / `type/enhancement` / `type/question`
- `area/model-support` / `area/webbridge` / `area/build-system` 等

分类逻辑见 `.github/workflows/issue-classifier.yml`。

## Label 配置

项目助手需要以下 Labels（共 14 个）。可通过下列方式导入：

### 方法一：gh CLI（推荐）

```bash
# 确保 gh 已认证
gh auth status

# 运行导入脚本
bash scripts/import-labels.sh
```

### 方法二：手动创建

在 GitHub 仓库 → Settings → Labels 中手动创建以下 Labels：

| Label | 颜色 | 说明 |
|-------|------|------|
| `type/bug` | `#d73a4a` | Bug 报告 |
| `type/enhancement` | `#a8d692` | 新功能/增强 |
| `type/question` | `#0066cc` | 技术问题 |
| `area/model-support` | `#9b59b6` | 新模型支持相关 |
| `area/webbridge` | `#f39c12` | Webbridge 增强相关 |
| `area/build-system` | `#7f8c8d` | CMake/跨平台构建 |
| `area/cli-tools` | `#1abc9c` | CLI 工具改进 |
| `area/python-binding` | `#e91e63` | Python 绑定 |
| `area/engine` | `#f1c40f` | 推理引擎核心 |
| `status/needs-investigation` | `#f39c12` | 待调查 |
| `status/needs-design` | `#e67e22` | 待设计决策 |
| `status/ready-for-dev` | `#27ae60` | 可开发 |
| `status/in-progress` | `#3498db` | 开发中 |
| `status/blocked` | `#c0392b` | 阻塞 |

## 周报工作流详情

### 触发方式

- **定时**: 每周六 UTC 1:00（北京时间 9:00）
- **手动**: GitHub Actions 页面 → Run workflow

### 逻辑

1. 收集过去 7 天新建的 Issues（按 `created_at` 过滤，排除 PR）、合并的 PRs（按 `merged_at` 过滤）、新建的 Discussions
2. 若无内容 → 跳过，不生成空周报
3. 若有内容：
   - 通过 GraphQL 查找当前季度的周报 Discussion 帖（标题 `[周报] ADI 项目 YYYY-QX 周报汇总`）
   - 若不存在 → 创建新帖（自动选择 `general` 分类，无则退到第一个可用分类）
   - 若存在 → 在原帖追加本周内容

> 注意：GitHub Discussions **只有 GraphQL API**（`github.rest.repos.*Discussion*` 并不存在），
> 本工作流统一使用 `github.graphql` 读写 Discussions。

### 周报内容

- 本周概览（Issues、PRs、Discussions 数量）
- Bug 报告列表
- 建议 & 增强列表
- 技术问题列表
- 模型支持 / Webbridge 进展
- 合并 PR 列表
- 待跟进事项

## Copilot 提示词

`.github/copilot-instructions.md` 是 GitHub Copilot 约定的仓库级自定义指令文件名（旧名 `COPILOT_INSTRUCTIONS.md` 不会被识别）。内容涵盖：

- 项目概述（ADI 是什么）
- 核心组件结构
- 已支持模型列表
- 架构关键点
- 回答准则和反馈处理准则

## 注意事项

1. **Discussions 需要开启**: 周报依赖 Discussions 功能。如未开启，请在仓库 Settings → General → Discussions 启用；未开启时工作流会给出 warning 并跳过发布。
2. **Labels 需先导入**: 分类器在标签不存在时不会失败（仅 warning），但建议先运行 `scripts/import-labels.sh` 保证 `type/*`、`area/*`、`status/*` 全套标签可用。
3. **Discussion 分类**: 周报自动选择 `general` 分类（无则取第一个可用分类），无需再手工调整 `category_id`。
4. **模板格式**: 本配置使用 YAML format Issue forms；`config.yml` 已关闭空白 Issue，并将一般性问题引导至 Discussions。

## 维护

- 如需新增领域标签，在 `scripts/github-labels.json` 中添加，并重新运行导入脚本（幂等，可重复执行）
- 如需调整分类逻辑，编辑 `.github/workflows/issue-classifier.yml`
- 如需修改 Copilot 行为，编辑 `.github/copilot-instructions.md`
