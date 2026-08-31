# DINOv3‑Retrieval
PyTorch implementation for image retrieval based on DINOv3, supporting multiple standard retrieval benchmarks (ROxford5k, RParis6k, Holidays, GLDv2‑clean‑val, CUB‑200‑2011, Stanford‑Dogs, Stanford‑Cars).
The main highlights of this algorithm include:(1) A two-stage refinement method is proposed to adaptively purify the attention weights. This method is called the global threshold mask refinement (GTMR). It can effectively suppress background noise and enhance the perception ability of target objects. (2) The GRD directly uses the output of the last layer attention module (LLA) to optimize the output structure of the DINOv3-[ViT-L/16] model. It can retain the target-related information and avoid feature distortion. (3) A lightweight equalized feature weighting (EFW) method is proposed for dimensionality reduction. It possesses the robustness to maintain high retrieval accuracy even when the feature dimension is reduced by half. Extensive comparative experiments on the ROxford, RParis, ROxford+1M, and RParis+1M datasets demonstrate that GRD method outperforms the state-of-the-art methods in mean average precision (mAP) metric. Furthermore, our method also demonstrates its strong robustness in complex background scenarios.

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
└── main.py                   # main entry for retrieval evaluation      
```

## 📥 Prepare weights & datasets
1. **DINOv3 pretrained weight**
Download official pretrained checkpoint, modify `WEIGHT_PATH` in `config.py` point to your local `.pth` file. The pre-trained model weights of Dinov3 can be obtained from the following link: https://pan.baidu.com/s/1HqWR3CppWbP7ynTPTvG-4w (password: GTMR)

2. **Datasets**
- ROxford5k / RParis6k: download official dataset + ground‑truth pkl file.
- Holidays: upright version images.
- GLDv2‑clean‑val: download Google Landmark v2 train images and pre‑computed `GLDv2‑clean‑val‑split.pkl`.
- CUB‑200‑2011 / Stanford Dogs / Stanford Cars: follow official dataset download instructions.

Modify dataset root paths inside `init_params()` in `main.py` to match your local disk path.






