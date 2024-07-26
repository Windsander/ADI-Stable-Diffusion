# Attention: 

## Currently, no need to execute script below, because we are dynamic prepared inference engine at building stage. 

## but if you want to do it by self, there's 2 ways to do it.

- **Method-1:** by simply executing auto_script [auto_prepare_engine_env.sh](engine%2Fauto_prepare_engine_env.sh):
```bash
cd ./engine
# if setting [VERSION] [PLATFORM], script will downloading target [PLATFORM] ORT, like onnxruntime-linux-x64-1.18.0.tgz at official
bash auto_prepare_engine_env.sh [VERSION] [PLATFORM]

# if not setting [VERSION] [PLATFORM], script will default using onnxruntime-osx-arm64-1.18.0, so be careful!!!
bash auto_prepare_engine_env.sh
```

- **Method-2:** or download ONNXRuntime 1.18.0 manually from [ORT official](https://github.com/microsoft/onnxruntime/releases/v1.18.0/), there's an example for [onnxruntime-osx-arm64-1.18.0](https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-arm64-1.18.0.tgz) in MacOS:
```bash
cd ./engine
# if using curl
curl -L -o onnxruntime-osx-arm64-1.18.0.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-arm64-1.18.0.tgz

# if using wget
wget -O onnxruntime-osx-arm64-1.18.0.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-osx-arm64-1.18.0.tgz
```

## Bash Tools:
- **[auto_prepare_engine_env]** at: [auto_prepare_engine_env.sh](auto_prepare_engine_env.sh) <br>
  This script tool helps you quickly prepare ONNXRuntime inference engine from from [ORT official](https://github.com/microsoft/onnxruntime/releases/)
