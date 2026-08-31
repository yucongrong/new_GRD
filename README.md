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

## 🚀 Quick start
### 1. Run retrieval evaluation (main entry)
```python
# modify params["test_set"] to choose dataset: "roxford5k", "rparis6k", "holidays", "googlelandmark5k"
python main.py
```

Supported post‑processing modes (only **one** can be enabled at a time):
```python
params["enable_pca"]   = True    # PCA‑whitening / PWs‑Entropy whitening
params["enable_rp"]    = False   # random projection
params["enable_pq"]    = False   # product quantization
params["pca_mode"]     = "pws_entropy" # "standard" or "pws_entropy"
```

When running first time with SfM‑120k:
- Code will auto‑extract SfM‑120k train features and save to `sfm120k_feat.pkl`.
- PCA / PQ codebook will be fitted on SfM‑120k features and saved under `./feature_dump`.

### 2. Feature extraction API example
```python
from dinov3_model import DINOv3
from dinov3_extract import extract
from torch.utils.data import DataLoader
from datasets import ImageDataset

model = DINOv3(weight_path="your/checkpoint.pth").cuda()
ds = ImageDataset(filename_list, root_dir="./images", transform=model.transform)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
feats_np = extract(model, loader, device="cuda")
# feats_np: numpy array shape [N, 1024]
```

## 📊 Supported benchmarks & metrics
| Dataset | Metrics |
|---|---|
| ROxford5k / RParis6k | mAP(Easy / Medium / Hard), P@1 |
| Holidays / GLDv2‑clean‑val | Global mAP, P@1 / P@5 / P@10 |
| CUB‑200‑2011 | Top‑1 Acc, Top‑5 Acc |

Output example for ROxford5k:
```
===== RESULT (PCA + Standard‑Whitening) | roxford5k =====
Easy   mAP: 0.7821  P@1: 0.8429
Medium mAP: 0.7133  P@1: 0.7714
Hard   mAP: 0.5247  P@1: 0.5857
```

## ⚙️ Core algorithm description
1. **Attention‑guided foreground filtering**
Inside `SelfAttention.compute_attention()` (`layers.py`):
- Compute CLS‑to‑patch attention map; separate foreground / background by dynamic threshold; zero‑out background attention weights before weighted sum with value tokens.

2. **Post‑processing options**
- **Standard PCA‑Whitening**: SVD‑based PCA + whitening + L2 normalization.
- **PWs‑Entropy whitening**: entropy‑adapted whitening transformation (MATLAB‑style ported to PyTorch).
- **Random Projection**: Gaussian random projection + L2 norm.
- **Product Quantization**: faiss‑PQ for fast approximate search.

## 📝 Notes
1. **Memory**
- If you encounter OOM for SfM‑120k fitting, decrease `sfm120k_max_sample` in params.
2. **Feature array ownership warning**
The code checks `arr.flags.owndata` to avoid torch tensor memory view bugs. If you modify `extract()` remember return `feat.copy()`.
3. **CUB / Stanford Dogs / Stanford Cars**
These datasets are implemented inside `retrieval_utils.py`, you can write separate script to invoke `extract_cub_all_features()` / `evaluate_cub_features()`.

## 📄 Citation
If you use this code for your research, please cite our paper (to‑be‑updated).
```
@inproceedings{yourpaper2026,
  title={......},
  author={......},
  booktitle={......},
  year={2026}
}
```

## 📃 License
MIT‑License, for academic research purpose only.

> This project is built upon DINOv3. Please follow DINOv3 original license when using pre‑trained weights.

## ❓ Trouble‑shooting
1. `FileNotFoundError`: double‑check dataset paths and weight path in `config.py` and `main.py`.
2. PCA checkpoint load returns `ckpt_mode == pws_entropy`: your npz saved with PWs‑Entropy mode, cannot load as standard PCA object.
3. GLDv2 missing images: code will automatically skip missing files and filter classes with insufficient samples.

如果你需要，我可以再给你生成一份**中文版本README**，或者补充`.gitignore`文件内容。
