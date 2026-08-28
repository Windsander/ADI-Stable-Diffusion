# ADI-Stable-Diffusion 项目助手

## 项目概述
ADI (All Device Inference) Stable Diffusion 是一个本地 Stable Diffusion 推理引擎，使用 ONNX Runtime 实现跨平台部署（macOS aarch64/arm64、Android arm64-v8a/x86_64、Linux、Windows）。

## 核心组件
- **engine/**: 推理核心，使用 ONNX Runtime
- **clitools/**: 命令行工具，用于图像生成
- **webbridge/**: Web 桥接接口
- **ort_sd_py_imp.py**: Python 绑定
- **CMakeLists.txt**: 跨平台 CMake 构建系统

## 已支持模型（参见 showcase/）
- flux-schnell (1024x1024)
- sd-turbo (512x512)
- sd21 (768x768)
- sd35-turbo (1024x1024)
- sdxl-turbo (512x512)
- svd-img2vid (图像到视频)

## 架构关键点
- 使用 ONNX Runtime 进行推理
- CMake 跨平台构建
- 支持多种输出格式和分辨率
- Webbridge 提供 Web 接口
- 采用 C++ 主要实现，Python 作为辅助绑定

## 回答准则
1. 优先引用项目代码或文档
2. 技术问题尽量给出具体建议
3. 不确定时明确说明
4. 涉及模型支持时，参考现有模型架构

## 反馈处理准则
- **Bug**: 确认复现步骤、环境信息，提取关键信息
- **建议**: 评估可行性，记录 rationale
- **问题**: 提供答案或指引到相关资源
