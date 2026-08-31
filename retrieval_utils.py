import os
import gc
import pickle
import numpy as np
import faiss
import warnings
import random
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets import RawBytesImageDataset, cid2filename
from dinov3_model import DINOv3
from retrieval_metrics import compute_map, compute_ap


def extract_and_dump_sfm120k(model, sfm_root: str, save_pkl_path: str, device):
    os.makedirs(os.path.dirname(save_pkl_path), exist_ok=True)
    train_path_pkl = os.path.join(sfm_root, "train_paths.pkl")
    val_path_pkl = os.path.join(sfm_root, "val_paths.pkl")
    ims_root = os.path.join(sfm_root, "ims")
    with open(train_path_pkl, "rb") as f:
        train_cids = pickle.load(f)
    with open(val_path_pkl, "rb") as f:
        val_cids = pickle.load(f)
    train_abs_paths = [cid2filename(cid, ims_root) for cid in train_cids]
    val_abs_paths = [cid2filename(cid, ims_root) for cid in val_cids]
    for idx, p in enumerate(train_abs_paths[:2]):
        if not os.path.exists(p):
            print(f" Warning: train image not exist: {p}")
    for idx, p in enumerate(val_abs_paths[:2]):
        if not os.path.exists(p):
            print(f" Warning: val image not exist: {p}")
    print(f"\nSfM‑120k dataset: train={len(train_abs_paths)} imgs, val={len(val_abs_paths)}")
    train_ds = RawBytesImageDataset(train_abs_paths, model.transform)
    val_ds = RawBytesImageDataset(val_abs_paths, model.transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    from dinov3_extract import extract
    print("Extracting SfM‑120k TRAIN features ...")
    feat_train = extract(model, train_loader, device)
    print("Extracting SfM‑120k VAL features ...")
    feat_val = extract(model, val_loader, device)
    dump_dict = {"train": feat_train.copy(), "val": feat_val.copy()}
    with open(save_pkl_path, "wb") as f:
        pickle.dump(dump_dict, f)
    print(f" SfM‑120k特征已保存至 {save_pkl_path}\n")
    return dump_dict


def load_sfm120k_db_features(pkl_path: str, max_sample: int = None) -> np.ndarray:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"[DEBUG] sfm120k pkl keys: {list(data.keys())}")
    if "train" not in data:
        raise KeyError("pkl中必须包含'train'键，请重新运行特征提取")
    feat = data["train"]
    if not isinstance(feat, np.ndarray):
        raise TypeError(f"train特征不是numpy数组，type={type(feat)}")
    feat = feat.astype(np.float32)
    print(f"Loaded SFM‑120k train full features, shape: {feat.shape}")
    if max_sample is not None and feat.shape[0] > max_sample:
        rng = np.random.default_rng(42)
        perm = rng.permutation(feat.shape[0])[:max_sample]
        feat = feat[perm].copy()
        print(f"[INFO] Random sample {max_sample} samples for training, new shape: {feat.shape}")
    return feat


def save_pq_codebook(index_pq: faiss.IndexPQ, path: str):
    if not index_pq.is_trained:
        raise RuntimeError("PQ index尚未训练，不能保存码本")
    index_pq.reset()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index_pq, path)
    print(f"PQ码本已保存到 {path}")


def load_pq_codebook(path: str) -> faiss.IndexPQ:
    index = faiss.read_index(path)
    if not index.is_trained:
        raise RuntimeError("加载的PQ index未训练")
    print(f"PQ码本已加载 {path}")
    return index


def save_feature_dump(save_dir: str, test_set: str, run_mode: str,
                      db_feat: np.ndarray, q_uncrop: np.ndarray, q_crop: np.ndarray,
                      imlist: list, qimlist: list):
    os.makedirs(save_dir, exist_ok=True)
    safe_mode = run_mode.replace(" ", "_").replace("|", "_").replace("=", "")
    save_name = f"{test_set}_{safe_mode}.npz"
    save_path = os.path.join(save_dir, save_name)
    np.savez_compressed(
        save_path,
        db_feat=db_feat,
        q_uncrop=q_uncrop,
        q_crop=q_crop,
        imlist=np.array(imlist, dtype=object),
        qimlist=np.array(qimlist, dtype=object)
    )
    print(f"\n特征已保存 -> {save_path}")
    print(f"   db_feat shape: {db_feat.shape}")
    print(f"   q_uncrop shape: {q_uncrop.shape}")
    print(f"   q_crop shape: {q_crop.shape}")
    return save_path


CUB200_CFG = {
    "dataset_root": r"G:\datasets\CUB_200_2011\CUB_200_2011",
    "images_dir": "images",
    "train_test_split": "train_test_split.txt",
    "images_txt": "images.txt",
    "image_class_labels_txt": "image_class_labels.txt",
    "classes_txt": "classes.txt",
    "bbox_file": "bounding_boxes.txt",
    "save_features_path": "./cub200_features",
    "distance_metric": "euclidean",
    "pca_dims": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "fusion_method": "weighted",
    "raw_weight": 1.0,
    "crop_weight": 1.0,
}

bbox_dict = {}
test_data_cub = {"filepath": [], "target": [], "img_id": []}


def load_bbox_from_txt_cub():
    global bbox_dict
    if bbox_dict:
        return bbox_dict
    bbox_path = os.path.join(CUB200_CFG["dataset_root"], CUB200_CFG["bbox_file"])
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"bbox file not found: {bbox_path}")
    bbox_dict.clear()
    with open(bbox_path, "r", encoding="utf‑8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            img_id, x, y, w, h = parts
            img_id = int(img_id)
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)
            if x1 >= x2 or y1 >= y2:
                warnings.warn(f"invalid bbox img_id={img_id}")
                continue
            bbox_dict[img_id] = (x1, y1, x2, y2)
    print(f"loaded {len(bbox_dict)} cub bboxes")
    return bbox_dict


def cub_load():
    root = CUB200_CFG["dataset_root"]
    images_dict = {}
    with open(os.path.join(root, CUB200_CFG["images_txt"]), "r", encoding="utf‑8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, img_rel = line.split(maxsplit=1)
            images_dict[int(img_id)] = img_rel
    label_dict = {}
    with open(os.path.join(root, CUB200_CFG["image_class_labels_txt"]), "r", encoding="utf‑8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, cid = line.split()
            label_dict[int(img_id)] = int(cid)
    split_dict = {}
    with open(os.path.join(root, CUB200_CFG["train_test_split"]), "r", encoding="utf‑8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, is_train = line.split()
            split_dict[int(img_id)] = int(is_train) == 1
    test_img_ids = sorted([iid for iid in split_dict if not split_dict[iid]])
    global test_data_cub
    test_data_cub["filepath"] = []
    test_data_cub["target"] = []
    test_data_cub["img_id"] = []
    for iid in test_img_ids:
        if iid in images_dict and iid in label_dict:
            test_data_cub["filepath"].append(images_dict[iid])
            test_data_cub["target"].append(label_dict[iid])
            test_data_cub["img_id"].append(iid)
    test_data_cub["target"] = np.array(test_data_cub["target"])
    train_data = {"filepath": [], "target": []}
    classes_dict = {}
    with open(os.path.join(root, CUB200_CFG["classes_txt"]), "r", encoding="utf‑8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid, cname = line.split(maxsplit=1)
            classes_dict[int(cid)] = cname
    return test_data_cub, train_data, classes_dict


def compute_cubtopmap(ranklist: np.ndarray, db_labels: np.ndarray, q_labels: np.ndarray):
    if ranklist.ndim != 2:
        raise ValueError("ranklist must be 2‑D")
    nq = ranklist.shape[1]
    total = min(nq, len(q_labels))
    acc1 = 0
    acc5 = 0
    for i in range(total):
        r = ranklist[:, i]
        top1_idx = int(r[0])
        q_cls = q_labels[i]
        if db_labels[top1_idx] == q_cls:
            acc1 += 1
        top5_idx = r[:5].astype(int)
        top5_cls = db_labels[top5_idx]
        if q_cls in top5_cls:
            acc5 += 1
    top1_acc = acc1 / total
    top5_acc = acc5 / total
    return top1_acc, top5_acc


def extract_fused_feature_cub(model, img_path: str, img_id: int, img_size=448):
    bbox = bbox_dict.get(img_id, None)
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def _get_feat(pil_img):
        tens = trans(pil_img).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad():
            feat = model(tens).cpu().numpy()[0]
        return feat

    try:
        im_raw = Image.open(img_path).convert("RGB")
    except Exception as e:
        warnings.warn(f"open image failed {img_path}: {e}")
        dim = CUB200_CFG["embed_dim"] if "embed_dim" in CUB200_CFG else 1024
        return np.zeros(dim), np.zeros(dim), np.zeros(dim)
    raw_feat = _get_feat(im_raw)
    if bbox is not None:
        im_crop = im_raw.crop(bbox)
        crop_feat = _get_feat(im_crop)
    else:
        crop_feat = raw_feat.copy()
    if CUB200_CFG["fusion_method"] == "weighted":
        w_r = CUB200_CFG["raw_weight"]
        w_c = CUB200_CFG["crop_weight"]
        fused = (w_r * raw_feat + w_c * crop_feat) / (w_r + w_c)
    elif CUB200_CFG["concat"]:
        fused = np.concatenate([raw_feat, crop_feat], axis=0)
    else:
        raise ValueError("fusion_method only support weighted / concat")
    return fused, raw_feat, crop_feat


def extract_cub_all_features(model):
    load_bbox_from_txt_cub()
    test_data_cub_local, _, _ = cub_load()
    root = CUB200_CFG["dataset_root"]
    img_info = []
    for img_id in test_data_cub_local["img_id"]:
        rel_path = test_data_cub_local["filepath"][test_data_cub_local["img_id"].index(img_id)]
        abs_path = os.path.join(root, CUB200_CFG["images_dir"], rel_path)
        if os.path.exists(abs_path):
            img_info.append({"abs_path": abs_path, "img_id": img_id})
    print(f"CUB‑200‑2011 test set total valid images: {len(img_info)}")
    fused_list, raw_list, crop_list = [], [], []
    for info in img_info:
        f, r, c = extract_fused_feature_cub(model, info["abs_path"], info["img_id"])
        fused_list.append(f)
        raw_list.append(r)
        crop_list.append(c)
    fused_arr = np.array(fused_list)
    raw_arr = np.array(raw_list)
    crop_arr = np.array(crop_list)
    return fused_arr, raw_arr, crop_arr, img_info


def evaluate_cub_features(feat_mat: np.ndarray, db_labels: np.ndarray, q_labels: np.ndarray):
    dist = cdist(feat_mat, feat_mat, metric=CUB200_CFG["distance_metric"])
    ranklist = np.argsort(dist, axis=0)
    top1, top5 = compute_cubtopmap(ranklist, db_labels, q_labels)
    print(f"CUB‑200‑2011 eval result | Top‑1: {top1:.4f}, Top‑5: {top5:.4f}")
    return {"top1": top1, "top5": top5, "ranklist": ranklist}


def save_cub_features(fused, raw, crop):
    out_dir = CUB200_CFG["save_features_path"]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "cub_fused.npy"), fused)
    np.save(os.path.join(out_dir, "cub_raw.npy"), raw)
    np.save(os.path.join(out_dir, "cub_crop.npy"), crop)
    print(f"CUB‑200‑2011 features saved to {out_dir}")
