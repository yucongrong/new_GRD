import torch
from functools import partial
import torch.nn as nn
from layers import Mlp, SwiGLUFFN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_PATH = r"G:\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
VIS_SAVE_DIR = r"H:\WYW\paper2\vision"

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
}

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}
