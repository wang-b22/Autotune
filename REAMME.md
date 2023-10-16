# Neural Network Vocoder for AutoTune 
## Model Introduction
- the model supports key shift and speed scale frame by frame, the infer file only shows the global autotune 
- the model is based on hifigan network
- The sampling rate is 32000Hz
- due to the inexact pitch extracted by praat, pitch was interpolated before input to the network
- model weight is avaliable at: [model.pth](https://drive.google.com/drive/folders/1Q8OKcdihc-HUNo7SHtbMYMo4FXcHdxt1?usp=sharing)

## Inference
```
python infer.py --ckpt ckpt/model.pth --config ckpt/config.json --speech_scale 1.0 --key_shift 0
```

## reference
- [Hifigan](https://github.com/jik876/hifi-gan)
- [Hifi++](https://github.com/rishikksh20/HiFiplusplus-pytorch)
- [neural_source_filter](https://github.com/Takaaki-Saeki/simplified_neural_source_filter)
