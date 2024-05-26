# Attention: 

## the dir under this root, is defined used by clitools/examples/<action>.sh scripts.

- **[io-test]** dir: contain clitools input & output, by using clitools/examples/<action>.sh scripts.
- **[sd-base-model]** dir: you can found target model from HuggingFace, and clone it to this dir, 
    clitools/examples/<action>.sh scripts will rely on it.
- **[sd-dictionary]** dir: put Tokenizer reference Vocabulary-Dictionary under here.


## so, if you run command tools on you own, just be careful about the path setting.


## Bash Tools:
- **[env_auto_prepare]** at: [auto_prepare_converter_env.sh](auto_prepare_converter_env.sh) <br>
     This script tool helps you quickly prepare an environment for self-hosted HuggingFace models,  
     which needs converting from **\<model\>.sfaetensors** to **\<model\>.onnx format**. <br>
     Making them compatible with more platforms and tools, by using **optimum-cli** from readied env. <br> 
     You can find more information about **optimum-cli** from: [Optimum-Official: ONNX Runtime](https://huggingface.co/docs/diffusers/optimization/onnx)