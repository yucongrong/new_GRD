# DINOv3‑Retrieval
PyTorch implementation for image retrieval based on DINOv3, supporting multiple standard retrieval benchmarks (ROxford5k, RParis6k, Holidays, GLDv2‑clean‑val, CUB‑200‑2011, Stanford‑Dogs, Stanford‑Cars).
This repository implements attention‑guided mask filtering, multiple post‑processing methods including PCA‑whitening, PWs‑Entropy whitening, random projection(RP), product quantization(PQ).

> ⚠️ Note: This is research‑code for paper experiments, not production‑ready.

## 📋 Environment
```bash
# python >=3.10
pip install torch torchvision faiss‑cpu numpy pillow scipy
```
- `faiss‑cpu`: for retrieval ranking / PQ quantization
- `torch>=2.2` recommended for DINOv3 model

## 📁 Project structure
```
DINOv3‑Retrieval
├── config.py                 # global config, device, dtype, layer dict
├── project_utils.py          # ViT helper modules: LayerScale, PatchEmbed, RoPE utils
├── layers.py                 # Self‑Attention, SwiGLUFFN, RopePositionEmbedding
├── dinov3_model.py           # DINOv3 backbone model definition
├── dinov3_extract.py         # feature extraction wrapper
├── datasets.py               # multi‑dataset dataset / ground‑truth building
├── retrieval_metrics.py      # mAP, P@k evaluation for ROxford / RParis / Holidays
├── retrieval_utils.py        # cub / dog / car dataset helper functions
├── transforms_utils.py       # PCA‑whitening, PWs‑Entropy‑whitening, RP implementation
├── main.py                   # main entry for retrieval evaluation
└── feature_dump/             # auto‑generated: PCA/PQ checkpoint & extracted features
```

## 📥 Prepare weights & datasets
1. **DINOv3 pretrained weight**
Download official pretrained checkpoint, modify `WEIGHT_PATH` in `config.py` point to your local `.pth` file.

2. **Datasets**
- ROxford5k / RParis6k: download official dataset + ground‑truth pkl file.
- Holidays: upright version images.
- GLDv2‑clean‑val: download Google Landmark v2 train images and pre‑computed `GLDv2‑clean‑val‑split.pkl`.
- CUB‑200‑2011 / Stanford Dogs / Stanford Cars: follow official dataset download instructions.

Modify dataset root paths inside `init_params()` in `main.py` to match your local disk path.






