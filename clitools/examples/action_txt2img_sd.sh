# for Mac temp files cleaning
find . -name "._*" -type f -print
find . -name "._*" -type f -delete

# executing Stable Diffusion with params below
../../cmake-build-debug/bin/ort-sd-clitools\
 -p "\
 best quality, extremely detailed,\
 (keep main character),\
 A cat in the water at sunset\
 "\
 -n "\
 worst quality, low quality, normal quality, lowres, watermark, monochrome, grayscale, ugly, blurry,\
 Tan skin, dark skin, black skin, skin spots, skin blemishes, age spot, glans, disabled, bad anatomy,\
 amputation, bad proportions, twins, missing body, fused body, extra head, poorly drawn face, bad eyes,\
 deformed eye, unclear eyes, cross-eyed, long neck, malformed limbs, extra limbs, extra arms, missing arms,\
 bad tongue, strange fingers, mutated hands, missing hands, poorly drawn hands, extra hands, fused hands,\
 connected hand, bad hands, missing fingers, extra fingers, 4 fingers, 3 fingers, deformed hands,\
 extra legs, bad legs, many legs, more than two legs, bad feet, extra feet\
 "\
 -m txt2img\
 -o ../../sd/io-test/output.png\
 -w 512 -h 512 -c 3\
 --seed 15.0\
 --dims 768\
 --clip ../../sd/sd-base-model/onnx-sd-v15/text_encoder/model.onnx\
 --unet ../../sd/sd-base-model/onnx-sd-v15/unet/model.onnx\
 --vae-encoder ../../sd/sd-base-model/onnx-sd-v15/vae_encoder/model.onnx\
 --vae-decoder ../../sd/sd-base-model/onnx-sd-v15/vae_decoder/model.onnx\
 --merges ../../sd/sd-base-model/onnx-sd-v15/tokenizer/merges.txt\
 --dict ../../sd/sd-base-model/onnx-sd-v15/tokenizer/vocab.json\
 --beta-start 0.00085\
 --beta-end 0.012\
 --beta linear\
 --alpha cos\
 --scheduler euler_a\
 --predictor epsilon\
 --tokenizer bpe\
 --train-steps 1000\
 --token-idx-num 49408\
 --token-length 77\
 --token-border 1.0\
 --gain 1.1\
 --decoding 0.18215\
 --guidance 1.5\
 --steps 10\
 -v
