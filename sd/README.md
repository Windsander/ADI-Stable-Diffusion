# Attention: 

## the dir under this root, is defined used by clitools/examples/<action>.sh scripts.

- **[io-test]** dir: contain clitools input & output, by using clitools/examples/<action>.sh scripts.
- **[sd-base-model]** dir: you can found target model from HuggingFace, and clone it to this dir, 
    clitools/examples/<action>.sh scripts will rely on it.
- **[sd-dictionary]** dir: put Tokenizer reference Vocabulary-Dictionary under here.


## so, if you run command tools on you own, just be careful about the path setting.


## but if you want to do it by self, there's 2 ways to do it.

- **Method-1:** by simply executing auto_script [auto_prepare_converter_env.sh](sd%2Fauto_prepare_converter_env.sh):
```bash
cd ./sd
bash auto_prepare_converter_env.sh
```

- **Method-2:** or download SD manually from [HuggingFace-SD-Official](https://huggingface.co/stabilityai), there's an example for [sd-turbo official](https://huggingface.co/stabilityai/sdxl-turbo/tree/main). **However, not all models have an official .onnx format provided.**:

```bash
cd ./sd/sd-base-model/
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 -b onnx onnx-official-sd-v15/
```


- **Method-3:** or download SD manually from [HuggingFace-Our-Converted](https://huggingface.co/Windsander), there's an example for [sd-turbo official](https://huggingface.co/stabilityai/sdxl-turbo/tree/main):

```bash
cd ./sd/sd-base-model/
git clone https://huggingface.co/Windsander/onnx-sd-v15 onnx-sd-v15/
```


## Bash Tools:
- **[auto_prepare_env]** at: [auto_prepare_converter_env.sh](auto_prepare_converter_env.sh) <br>
     This script tool helps you quickly prepare an environment for self-hosted HuggingFace models,  
     which needs converting from **\<model\>.sfaetensors** to **\<model\>.onnx format**. <br>
     Making them compatible with more platforms and tools, by using **optimum-cli** from readied env. <br> 
     You can find more information about **optimum-cli** from: [Optimum-Official: ONNX Runtime](https://huggingface.co/docs/diffusers/optimization/onnx)
- **[auto_ready_model]** at: [auto_prepare_sd_models.sh](sd-base-model%2Fauto_prepare_sd_v15.sh) <br>
     If you want to prepare convert environment & sd-series at same time, then just bash this script. <br>
     The models we support, will download by script completely.