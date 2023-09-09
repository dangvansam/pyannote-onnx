# Voice Activity Detection with `pyannote.audio` (ONNX version)

Suported ONNX runtime for [pyannote.audio](https://github.com/pyannote/pyannote-audio)

## Installation
Only Python 3.8+ is supported.
```bash
# for CPU
pip install onnxruntime
# for GPU, check version on: https://onnxruntime.ai/docs/build/eps.html#cuda
pip install onnxruntime-gpu
# install pyannote
pip install -e .
```

## 1. Export ONNX from PyTorch model
```bash
# 1. Download pytorch model (.bin) from https://huggingface.co/pyannote/segmentation/blob/main/pytorch_model.bin
wget https://huggingface.co/pyannote/segmentation/blob/main/pytorch_model.bin -O pytorch_model/vad_model.bin
# 2. Export
python onnx/export_onnx.py -i pytorch_model/vad_model.bin -o onnx_model/

```
## Run VAD

```bash
# use onnx model (2x faster)
python vad.py -m onnx_model/vad_model.onnx -i tests/data/test_vad.wav
# mean time cost = 5.32921104

# use pytorch model
python vad.py -m onnx_model/vad_model.bin -i tests/data/test_vad.wav
# mean time cost = 9.56711404
```

## Benchmark

Test file [tests/data/test_vad.wav](tests/data/test_vad.wav) with duration 6m15s
+ CPU Intel(R) Xeon(R) CPU E5-2683 v3 @ 2.00GHz
+ GPU Nvidia RTX 3090

Batch size 32
| Backend | CPU time (s)   | GPU time (s)   |
| :---:   | :---: | :---: |
| PyTorch | 12.0    | 1.76   |
| ONNX    | 4.33    | 1.88   |

<!-- Batch size 64
| Backend | CPU time (s)   | GPU time (s)   |
| :---:   | :---: | :---: |
| PyTorch |  inf   | 1.99   |
| ONNX    | 4.02    | NA   | -->

## Citations

If you use `pyannote.audio` please use the following citations:

```bibtex
@inproceedings{Bredin2020,
  Title = {{pyannote.audio: neural building blocks for speaker diarization}},
  Author = {{Bredin}, Herv{\'e} and {Yin}, Ruiqing and {Coria}, Juan Manuel and {Gelly}, Gregory and {Korshunov}, Pavel and {Lavechin}, Marvin and {Fustes}, Diego and {Titeux}, Hadrien and {Bouaziz}, Wassim and {Gill}, Marie-Philippe},
  Booktitle = {ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing},
  Year = {2020},
}
```

```bibtex
@inproceedings{Bredin2021,
  Title = {{End-to-end speaker segmentation for overlap-aware resegmentation}},
  Author = {{Bredin}, Herv{\'e} and {Laurent}, Antoine},
  Booktitle = {Proc. Interspeech 2021},
  Year = {2021},
}
```