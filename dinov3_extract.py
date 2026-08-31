import numpy as np
import torch
from tqdm import tqdm
from config import device


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats = []
    for batch in tqdm(loader):
        if len(batch) == 3:
            x, idx, im_np = batch
        else:
            x, idx = batch
            im_np = None
        feat = model(x.to(device), original_imgs=im_np).cpu().numpy()
        feat1 = feat[:, :512]
        feat2 = feat[:, -512:]
        feat = 0.9 * feat1 + 0.1 * feat2
        feats.append(feat)
    return np.concatenate(feats)
