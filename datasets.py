import os
import io
import pickle
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import warnings
import numpy as np
import xml.etree.ElementTree as ET
from scipy.io import loadmat


bbox_cache = {}
test_data_cub = {
    "db_target": np.array([]),
    "query_target": np.array([]),
    "filepath": [],
    "img_id": []
}


class RawBytesImageDataset(Dataset):
    def __init__(self, path_list, transform):
        self.paths = path_list
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        p = self.paths[index]
        with open(p, "rb") as f:
            img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = img
        return img_tensor, index

def cid2filename(cid, ims_root):
    folder1 = cid[-2:]
    folder2 = cid[-4:-2]
    folder3 = cid[-6:-4]
    file_name = cid
    return os.path.join(ims_root, folder1, folder2, folder3, file_name)

def gldv2_imageid_to_path(image_id: str, train_root: str):
    sub1 = image_id[0]
    sub2 = image_id[1]
    sub3 = image_id[2]
    return os.path.join(train_root, sub1, sub2, sub3, f"{image_id}.jpg")

def build_gldv2_cleanval_gt(split_pkl_path: str, train_img_root: str, sample_num: int = None):
    with open(split_pkl_path, "rb") as f:
        data_dict = pickle.load(f)
    file_ids = data_dict["file_ids"]
    labels = data_dict["labels"]
    assert len(file_ids) == len(labels), "GLD pkl: file_ids与labels长度不一致！"
    group_dict_all = defaultdict(list)
    for img_id, lid in zip(file_ids, labels):
        group_dict_all[lid].append(img_id)
    if sample_num is not None and isinstance(sample_num, int) and sample_num > 0:
        import random
        rng = random.Random(2)
        valid_classes = [lid for lid, imid_list in group_dict_all.items() if len(imid_list) >= 2]
        rng.shuffle(valid_classes)
        selected_img_ids = []
        selected_labels = []
        total_img = 0
        for cls in valid_classes:
            imid_list = group_dict_all[cls]
            selected_img_ids.extend(imid_list)
            selected_labels.extend([cls] * len(imid_list))
            total_img += len(imid_list)
            if total_img >= sample_num:
                break
        file_ids = selected_img_ids
        labels = selected_labels
        print(f"[GLDv2 debug sample] target ~{sample_num} imgs, actual selected {len(file_ids)} imgs, classes:{len(set(labels))}")
    total = len(file_ids)
    print(f"GLDv2 clean‑val total records from pkl: {total}")
    valid_imgid_label = []
    missing_count = 0
    for img_id, lid in zip(file_ids, labels):
        fp = gldv2_imageid_to_path(img_id, train_img_root)
        if os.path.exists(fp):
            valid_imgid_label.append((img_id, lid))
        else:
            missing_count += 1
            if missing_count <= 5:
                print(f"Missing image skip: {fp}")
    if missing_count > 0:
        print(f"[WARNING] Total skip missing image count = {missing_count}")
    file_ids = [x[0] for x in valid_imgid_label]
    labels = [x[1] for x in valid_imgid_label]
    group_after_filter = defaultdict(list)
    for img_id, lid in zip(file_ids, labels):
        group_after_filter[lid].append(img_id)
    keep_img_ids = []
    keep_labels = []
    for lid, imid_list in group_after_filter.items():
        if len(imid_list) >= 2:
            keep_img_ids.extend(imid_list)
            keep_labels.extend([lid] * len(imid_list))
    file_ids = keep_img_ids
    labels = keep_labels
    group_dict = defaultdict(list)
    for img_id, lid in zip(file_ids, labels):
        group_dict[lid].append(img_id)
    imlist_fullpath = []
    imid_2_path = dict()
    for img_id in file_ids:
        fp = gldv2_imageid_to_path(img_id, train_img_root)
        imid_2_path[img_id] = fp
        imlist_fullpath.append(fp)
    for idx, test_fp in enumerate(imlist_fullpath[:5]):
        print(f"GLDv2 check path: {test_fp}")
    imlist_filename_only = [os.path.basename(p) for p in imlist_fullpath]
    im2idx = {name: i for i, name in enumerate(imlist)}
    qimlist_filename = []
    qimlist_fullpath = []
    gnd_list = []
    for lid, iid_list in group_dict.items():
        iid_list.sort()
        q_iid = iid_list[0]
        q_fp = gldv2_imageid_to_path(q_iid, train_img_root)
        q_fn = os.path.basename(q_fp)
        pos_iids = iid_list[1:]
        pos_fns = [os.path.basename(gldv2_imageid_to_path(i, train_img_root)) for i in pos_iids]
        qimlist_filename.append(q_fn)
        qimlist_fullpath.append(q_fp)
        ok_idx = [im2idx[p] for p in pos_fns]
        junk_idx = [im2idx[q_fn]]
        gnd_list.append({"ok": ok_idx, "junk": junk_idx})
    print(f"[GLDv2 clean‑val GT after filter] classes={len(group_dict)}, queries={len(qimlist_filename)}, total images={len(imlist_filename_only)}")
    return imlist_filename_only, qimlist_filename, gnd_list, imlist_fullpath, qimlist_fullpath

def build_holidays_gt(all_filenames: list):
    group_dict = defaultdict(list)
    for fn in all_filenames:
        gid = fn[:4]
        group_dict[gid].append(fn)
    for g in group_dict:
        group_dict[g].sort()
    imlist = list(all_filenames)
    im2idx = {name: i for i, name in enumerate(imlist)}
    qimlist = []
    gnd_list = []
    for gid, file_list in group_dict.items():
        q_fn = file_list[0]
        pos_fns = file_list[1:]
        qimlist.append(q_fn)
        ok_idx = [im2idx[p] for p in pos_fns]
        junk_idx = [im2idx[q_fn]]
        gnd_list.append({"ok": ok_idx, "junk": junk_idx})
    print(f"[Holidays custom GT] total groups={len(group_dict)}, queries={len(qimlist)}, total database images={len(imlist)}")
    return imlist, qimlist, gnd_list


def compute_cubtopmap(ranklist):
    global test_data_cub
    if ranklist.ndim != 2:
        raise ValueError(f"ranklist维度异常，期望2维，实际{ranklist.ndim}")
    top1 = ranklist[0, :]
    total_queries = ranklist.shape[1]
    db_labels = test_data_cub["db_target"]
    q_labels = test_data_cub["query_target"]
    if total_queries != len(q_labels):
        warnings.warn(f"查询数({total_queries})与查询标签数({len(q_labels)})不匹配，将截断")
        total_queries = min(total_queries, len(q_labels))
        top1 = top1[:total_queries]
    acc1 = 0
    for i in range(total_queries):
        db_idx = int(top1[i])
        if db_idx >= len(db_labels):
            continue
        if q_labels[i] == db_labels[db_idx]:
            acc1 += 1
    top1_acc = acc1 / total_queries if total_queries else 0.0

    top5 = ranklist[0:5, :]
    acc5 = 0
    for i in range(total_queries):
        current_top5 = top5[:, i]
        current_top5 = current_top5[current_top5 < len(db_labels)]
        if len(current_top5) == 0:
            continue
        match_cls = any(db_labels[int(pid)] == q_labels[i] for pid in current_top5)
        if match_cls:
            acc5 += 1
    top5_acc = acc5 / total_queries if total_queries else 0.0


    return top1_acc, top5_acc

def load_cub_bbox(dataset_root: str) -> dict[int, tuple[int, int, int, int]]:
    bbox_path = os.path.join(dataset_root, "bounding_boxes.txt")
    bbox_dict = {}
    with open(bbox_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            img_id = int(parts[0])
            x, y, w, h = parts[1], parts[2], parts[3], parts[4]
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)
            bbox_dict[img_id] = (x1, y1, x2, y2)
    print(f"加载CUB bounding box 共 {len(bbox_dict)} 张")
    return bbox_dict

class CUBCropImageDataset(Dataset):
    def __init__(self, filename_list, imgid_list, root, bbox_dict, transform):
        self.filenames = filename_list
        self.imgid_list = imgid_list
        self.root = root
        self.bbox_dict = bbox_dict
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        rel_path = self.filenames[idx]
        full_path = os.path.join(self.root, rel_path)
        img = Image.open(full_path).convert("RGB")
        img_id = self.imgid_list[idx]
        x1, y1, x2, y2 = self.bbox_dict[img_id]
        crop_img = img.crop((x1, y1, x2, y2))
        if self.transform is not None:
            crop_img = self.transform(crop_img)
        return crop_img, idx

def build_cub200_gt(dataset_dir: str):
    global test_data_cub
    images_txt = os.path.join(dataset_dir, "images.txt")
    label_txt = os.path.join(dataset_dir, "image_class_labels.txt")
    split_txt = os.path.join(dataset_dir, "train_test_split.txt")
    img_root = os.path.join(dataset_dir, "images")
    imgid2relpath = {}
    imgid2cls = {}
    imgid2train = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, rel_sub_path = line.split(maxsplit=1)
            imgid2relpath[int(img_id)] = rel_sub_path
    with open(label_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, cid = line.split()
            imgid2cls[int(img_id)] = int(cid)
    with open(split_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, flag = line.split()
            imgid2train[int(img_id)] = int(flag) == 1
    db_imgids = [iid for iid in imgid2train if imgid2train[iid]]
    query_imgids = [iid for iid in imgid2train if not imgid2train[iid]]
    cls2dbids = defaultdict(list)
    for iid in db_imgids:
        c = imgid2cls[iid]
        cls2dbids[c].append(iid)
    imlist = []
    im2idx = {}
    db_targets = []
    for idx, iid in enumerate(db_imgids):
        rel_sub_path = imgid2relpath[iid]
        imlist.append(rel_sub_path)
        im2idx[rel_sub_path] = idx
        db_targets.append(imgid2cls[iid])
    qimlist = []
    q_targets = []
    gnd = []
    for q_iid in query_imgids:
        q_cls = imgid2cls[q_iid]
        q_rel_sub_path = imgid2relpath[q_iid]
        qimlist.append(q_rel_sub_path)
        q_targets.append(q_cls)
        pos_ids = cls2dbids[q_iid]
        pos_rel_paths = [imgid2relpath[i] for i in pos_ids]
        ok_idx = [im2idx[rp] for rp in pos_rel_paths]
        gnd.append({"ok": ok_idx, "junk": []})
    test_data_cub["db_target"] = np.array(db_targets)
    test_data_cub["query_target"] = np.array(q_targets)
    test_data_cub["filepath"] = imlist
    test_data_cub["img_id"] = db_imgids
    print(f"[CUB200 GT] DB:{len(imlist)}, Query:{len(qimlist)}")
    return imlist, qimlist, gnd, query_imgids

def get_real_filename_list(test_set, id_list):
    if test_set in ("roxford5k", "rparis6k"):
        return [f"{x}.jpg" for x in id_list]
    else:
        return id_list.copy()

def get_image_files(params: dict) -> dict:
    from config import FEATURE_CFG
    test_set = params["test_set"]
    image_paths = params["image_paths"].get(test_set, {})
    paths = {}
    if not image_paths:
        raise ValueError(f"未配置测试集 {test_set} 的图像路径")
    dataset_dir = image_paths.get("dataset")
    if dataset_dir and os.path.exists(dataset_dir) and test_set not in ("googlelandmark5k","cub200"):
        paths["test_files"] = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.lower().endswith(FEATURE_CFG["image_extensions"])
        ]
    if test_set not in ("holidays", "googlelandmark5k","cub200","stanford_dogs"):
        query_crop_dir = image_paths.get("query_crop")
        if query_crop_dir and os.path.exists(query_crop_dir):
            paths["query_files_crop"] = [
                os.path.join(query_crop_dir, f)
                for f in os.listdir(query_crop_dir)
                if f.lower().endswith(FEATURE_CFG["image_extensions"])
            ]
        query_uncrop_dir = image_paths.get("query_uncrop")
        if query_uncrop_dir and os.path.exists(query_uncrop_dir):
            paths["query_files_uncrop"] = [
                os.path.join(query_uncrop_dir, f)
                for f in os.listdir(query_uncrop_dir)
                if f.lower().endswith(FEATURE_CFG["image_extensions"])
            ]
    return paths

def load_stanford_dog_bbox(root: str, all_img_names: list) -> dict:
    global bbox_cache
    bbox_dict = {}
    anno_root = os.path.join(root, "annotation")
    for img_rel_path in all_img_names:
        cache_key = img_rel_path
        if cache_key in bbox_cache:
            bbox_dict[img_rel_path] = bbox_cache[cache_key]
            continue
        breed_folder, img_name = os.path.split(img_rel_path)
        img_id = os.path.splitext(img_name)[0]
        anno_file = os.path.join(anno_root, breed_folder, img_id)
        if not os.path.exists(anno_file):
            warnings.warn(f"标注文件缺失: {anno_file}，使用整张图")
            bbox_cache[cache_key] = (0, 0, 9999, 9999)
            bbox_dict[img_rel_path] = (0, 0, 9999, 9999)
            continue
        try:
            tree = ET.parse(anno_file)
            root_xml = tree.getroot()
            bnd = root_xml.find("./object/bndbox")
            if bnd is None:
                warnings.warn(f"{anno_file} 无bndbox")
                bbox_cache[cache_key] = (0, 0, 9999, 9999)
                bbox_dict[img_rel_path] = (0, 0, 9999, 9999)
                continue
            xmin = int(bnd.find("xmin").text)
            ymin = int(bnd.find("ymin").text)
            xmax = int(bnd.find("xmax").text)
            ymax = int(bnd.find("ymax").text)
            if xmin >= xmax or ymin >= ymax:
                warnings.warn(f"{img_rel_path} 坐标无效")
                bbox_cache[cache_key] = (0, 0, 9999, 9999)
                bbox_dict[img_rel_path] = (0, 0, 9999, 9999)
                continue
            bbox_cache[cache_key] = (xmin, ymin, xmax, ymax)
            bbox_dict[img_rel_path] = (xmin, ymin, xmax, ymax)
        except Exception as e:
            warnings.warn(f"解析{anno_file}失败：{str(e)}")
            bbox_cache[cache_key] = (0, 0, 9999, 9999)
            bbox_dict[img_rel_path] = (0, 0, 9999, 9999)
    print(f"加载Stanford Dogs bounding box 共 {len(bbox_dict)} 张")
    return bbox_dict

class DogCropImageDataset(Dataset):
    def __init__(self, filename_list, root, bbox_dict, transform):
        self.filenames = filename_list
        self.root = os.path.join(root, "images", "Images")
        self.bbox_dict = bbox_dict
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        rel_path = self.filenames[idx]
        full_path = os.path.join(self.root, rel_path)
        img = Image.open(full_path).convert("RGB")
        x1, y1, x2, y2 = self.bbox_dict[rel_path]
        crop_img = img.crop((x1, y1, x2, y2))
        if self.transform is not None:
            crop_img = self.transform(crop_img)
        return crop_img, idx

def build_stanford_dog_gt(dataset_root: str):
    global test_data_cub
    img_root = os.path.join(dataset_root, "Images")
    list_dir = os.path.join(dataset_root, "lists")
    train_mat = os.path.join(list_dir, "train_list.mat")
    test_mat = os.path.join(list_dir, "test_list.mat")
    train_data = loadmat(train_mat)
    test_data = loadmat(test_mat)
    train_names = [item[0] for item in train_data["file_list"].flatten()]
    test_names = [item[0] for item in test_data["file_list"].flatten()]
    def get_cls(name):
        prefix = name.split("-")[0]
        num_str = prefix.lstrip("n")
        return int(num_str)
    db_img_names = train_names
    q_img_names = test_names
    imlist = db_img_names
    im2idx = {name: i for i, name in enumerate(imlist)}
    db_targets = [get_cls(name) for name in db_img_names]
    q_targets = [get_cls(name) for name in q_img_names]
    gnd = []
    db_cls_map = {name: get_cls(name) for name in db_img_names}
    for q_name in q_img_names:
        q_cl = get_cls(q_name)
        ok_idx = [im2idx[db_n] for db_n in db_img_names if db_cls_map[db_n] == q_cl]
        gnd.append({"ok": ok_idx, "junk": []})
    test_data_cub["db_target"] = np.array(db_targets)
    test_data_cub["query_target"] = np.array(q_targets)
    test_data_cub["filepath"] = imlist
    print(f"[StanfordDog] DB:{len(imlist)}, Query:{len(q_img_names)}")
    return imlist, q_img_names, gnd, test_names


def load_stanford_cars_bbox(root: str, all_img_names: list) -> dict:
    mat_train = os.path.join(root, "cars_train_annos.mat")
    mat_test_label = os.path.join(root, "cars_test_annos_withlabels.mat")
    mat_test_nolabel = os.path.join(root, "cars_test_annos.mat")
    train_bbox_map = {}
    test_bbox_map = {}

    if os.path.exists(mat_train):
        mat_data = loadmat(mat_train)
        train_annos = mat_data["annotations"][0]
        for anno in train_annos:
            fname = str(anno["fname"][0])
            x1 = int(anno["bbox_x1"][0, 0])
            y1 = int(anno["bbox_y1"][0, 0])
            x2 = int(anno["bbox_x2"][0, 0])
            y2 = int(anno["bbox_y2"][0, 0])
            train_bbox_map[fname] = (x1, y1, x2, y2)
    # 加载测试标注
    test_annos = None
    if os.path.exists(mat_test_label):
        mat_data = loadmat(mat_test_label)
        test_annos = mat_data["annotations"][0]
    elif os.path.exists(mat_test_nolabel):
        mat_data = loadmat(mat_test_nolabel)
        test_annos = mat_data["annotations"][0]
    else:
        warnings.warn("缺失测试标注文件")
    if test_annos is not None:
        for anno in test_annos:
            fname = str(anno["fname"][0])
            x1 = int(anno["bbox_x1"][0, 0])
            y1 = int(anno["bbox_y1"][0, 0])
            x2 = int(anno["bbox_x2"][0, 0])
            y2 = int(anno["bbox_y2"][0, 0])
            test_bbox_map[fname] = (x1, y1, x2, y2)

    return test_bbox_map


def build_stanford_cars_gt(dataset_root: str):
    global test_data_cub
    meta_mat = os.path.join(dataset_root, "cars_meta.mat")
    mat_train = os.path.join(dataset_root, "cars_train_annos.mat")
    mat_test = os.path.join(dataset_root, "cars_test_annos_withlabels.mat")
    if not os.path.exists(meta_mat) or not os.path.exists(mat_train) or not os.path.exists(mat_test):
        raise FileNotFoundError("缺失cars_meta.mat / cars_train_annos.mat / cars_test_annos_withlabels.mat")
    loadmat(meta_mat)
    train_data = loadmat(mat_train)
    train_annos = train_data["annotations"][0]
    test_data = loadmat(mat_test)
    test_annos = test_data["annotations"][0]

    train_names = []
    train_cls = []
    for anno in train_annos:
        train_names.append(str(anno["fname"][0]))
        train_cls.append(int(anno["class"][0,0]))
    test_names = []
    test_cls = []
    for anno in test_annos:
        test_names.append(str(anno["fname"][0]))
        test_cls.append(int(anno["class"][0,0]))
    imlist = train_names
    qimlist = test_names
    im2idx = {n:i for i,n in enumerate(imlist)}

    cls2db = defaultdict(list)
    for idx, c in enumerate(train_cls):
        cls2db[c].append(idx)
    gnd = []
    for c in test_cls:
        gnd.append({"ok": cls2db[c], "junk": []})
    test_data_cub["db_target"] = np.array(train_cls)
    test_data_cub["query_target"] = np.array(test_cls)
    test_data_cub["filepath"] = imlist
    print(f"[StanfordCars] DB:{len(imlist)}, Query:{len(qimlist)}")
    print(f"DB样本示例 {imlist[:2]}")
    print(f"Query样本示例 {qimlist[:2]}")

    return imlist, qimlist, gnd, test_names

class CarCropImageDataset(Dataset):
    def __init__(self, filename_list, root, bbox_dict, transform):
        self.filenames = filename_list
        self.root = root
        self.bbox_dict = bbox_dict
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        rel_path = self.filenames[idx]
        full = os.path.join(self.root, rel_path)
        img = Image.open(full).convert("RGB")
        if rel_path in self.bbox_dict:
            x1, y1, x2, y2 = self.bbox_dict[rel_path]
            crop_img = img.crop((x1, y1, x2, y2))
        else:
            crop_img = img
        if self.transform is not None:
            crop_img = self.transform(crop_img)
        return crop_img, idx