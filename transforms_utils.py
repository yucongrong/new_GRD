import os
import numpy as np
import torch
from sklearn.preprocessing import Normalizer


def pca_whitening_fit_torch(train_np: np.ndarray, pca_dim: int, eps: float = 1e-6, device="cpu"):
    X = torch.from_numpy(train_np).float().to(device)
    mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - mean
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    components = Vh[:pca_dim, :]
    explained_variance = (S ** 2) / (X.shape[0] - 1.0)
    explained_variance = explained_variance[:pca_dim]

    class TorchPCA:
        def __init__(self, comp_torch, mu_torch, evar_torch, eps_val):
            self.components = comp_torch
            self.mean = mu_torch
            self.explained_variance = evar_torch
            self.eps = eps_val
        def transform(self, X_np: np.ndarray) -> np.ndarray:
            Xt = torch.from_numpy(X_np).float()
            Xc = Xt - self.mean
            proj = Xc @ self.components.T
            return proj.cpu().numpy()
    return TorchPCA(components, mean, explained_variance, eps)


def save_torch_pca(pca_obj, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        components=pca_obj.components.cpu().numpy(),
        mean=pca_obj.mean.cpu().numpy(),
        explained_variance=pca_obj.explained_variance.cpu().numpy(),
        eps=np.array(pca_obj.eps),
        pca_mode="pws_entropy"
    )
    print(f" PCA‑Whitening参数已保存：{save_path}")


def load_torch_pca(load_path: str, device="cpu"):
    data = np.load(load_path)
    pca_mode = str(data.get("pca_mode", "pws_entropy"))
    if pca_mode == "pws_entropy":
        return None, "pws_entropy"
    comp_np = data["components"]
    mu_np = data["mean"]
    evar_np = data["explained_variance"]
    eps_val = float(data["eps"])
    components = torch.from_numpy(comp_np).float().to(device)
    mean = torch.from_numpy(mu_np).float().to(device)
    explained_variance = torch.from_numpy(evar_np).float().to(device)

    class TorchPCA:
        def __init__(self, comp_torch, mu_torch, evar_torch, eps_val):
            self.components = comp_torch
            self.mean = mu_torch
            self.explained_variance = evar_torch
            self.eps = eps_val
        def transform(self, X_np: np.ndarray) -> np.ndarray:
            Xt = torch.from_numpy(X_np).float()
            Xc = Xt - self.mean
            proj = Xc @ self.components.T
            return proj.cpu().numpy()
    print(f" PCA‑Whitening参数已加载：{load_path}")
    return TorchPCA(components, mean, explained_variance, eps_val), "standard"


def apply_pca_whitening(pca_obj, feat_np: np.ndarray, whiten: bool, eps: float = 1e-6) -> np.ndarray:
    feat_proj = pca_obj.transform(feat_np)
    if whiten:
        evar = pca_obj.explained_variance.cpu().numpy()
        # scale = np.sqrt(evar + eps)
        scale = (evar + eps) ** (1.0 / 3.0)
        feat_proj = feat_proj / scale
    norms = np.linalg.norm(feat_proj, axis=1, keepdims=True)
    feat_proj = feat_proj / (norms + eps)
    return feat_proj


# ========= PWs entropy whitening =========
def pws_fit(XText: np.ndarray, dim: int):
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    mu = np.mean(XText, axis=0).reshape(1, -1)
    LAM = np.cov(XText, rowvar=False)
    U, S_diag, Vt = np.linalg.svd(LAM, full_matrices=False)
    scoreTest = (XText - mu) @ U
    x_test = scoreTest[:, :dim]
    LAM_PCA = np.cov(x_test, rowvar=False)
    u, s_diag, vt = np.linalg.svd(LAM_PCA, full_matrices=False)
    s = np.diag(s_diag)
    xRot = x_test @ u
    epsilon = 1e-5
    lam = np.diag(s)
    sum_lam = np.sum(lam, axis=0, keepdims=True)
    p = lam / sum_lam
    p_flat = p.flatten()
    p_flat = p_flat[p_flat != 0]
    entropy = -np.sum(p_flat * np.log(p_flat))
    denominator = (s_diag ** (1.0 / entropy)) + epsilon
    diag_mat = np.diag(1.0 / denominator)
    X_whitened = xRot @ diag_mat
    normalizer = Normalizer(norm='l2')
    test_features_pca = normalizer.transform(X_whitened)
    return test_features_pca, U, u, mu, s, dim, entropy


def pws_transform(X_in: np.ndarray, U: np.ndarray, mu: np.ndarray, u: np.ndarray, s_diag: np.ndarray, entropy: float, dim: int, eps=1e-5):
    scoreTest = (X_in - mu) @ U
    x_test = scoreTest[:, :dim]
    xRot = x_test @ u
    denominator = (s_diag ** (1.0 / entropy)) + eps
    diag_mat = np.diag(1.0 / denominator)
    X_whitened = xRot @ diag_mat
    norms = np.linalg.norm(X_whitened, axis=1, keepdims=True)
    feat_out = X_whitened / (norms + eps)
    return feat_out


def save_pws_param(save_path: str, U, u, mu, s_mat, dim, entropy):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    s_diag = np.diag(s_mat)
    np.savez_compressed(
        save_path,
        U=U,
        u=u,
        mu=mu,
        s_diag=s_diag,
        dim=dim,
        entropy=entropy,
        pca_mode="pws_entropy"
    )
    print(f" PWs‑Entropy白化参数保存至 {save_path}")


def load_pws_param(load_path: str):
    data = np.load(load_path)
    return (
        data["U"],
        data["u"],
        data["mu"],
        data["s_diag"],
        int(data["dim"]),
        float(data["entropy"])
    )


# ========= Random Projection =========
def random_projection_get_matrix_torch(orig_dim: int, target_dim: int, seed: int, device="cpu") -> torch.Tensor:
    torch.manual_seed(seed)
    R_torch = torch.normal(mean=0.0, std=1.0, size=(orig_dim, target_dim), device=device)
    return R_torch


def apply_random_projection_torch(X_np: np.ndarray, R_torch: torch.Tensor, eps=1e-6) -> np.ndarray:
    X = torch.from_numpy(X_np).float()
    scale = 1.0 / torch.sqrt(torch.tensor(R_torch.shape[1], dtype=torch.float32))
    Z = X @ R_torch * scale
    norms = torch.norm(Z, dim=1, keepdim=True)
    Z = Z / (norms + eps)
    return Z.cpu().numpy()


def pca_centered_false_np(X):

    mu = np.mean(X, axis=0)
    X_cent = X - mu
    _, _, Vt = np.linalg.svd(X_cent, full_matrices=True)
    coeff = Vt.T
    score = X @ coeff
    return coeff, score, mu

def xPCAWhitening_XY_np(XTrain: np.ndarray, XTest: np.ndarray):

    XTrain = np.nan_to_num(XTrain, nan=0.0).astype(np.float32)
    XTest = np.nan_to_num(XTest, nan=0.0).astype(np.float32)
    coeff, scoreTrain, mu = pca_centered_false_np(XTrain)
    x = scoreTrain.T
    N = x.shape[1]
    sigma = (x @ x.T) / N
    s, v, _ = np.linalg.svd(sigma, full_matrices=True)
    scoreTest95 = (XTest - mu) @ coeff
    X = scoreTest95.T
    xRot = s.T @ X
    epsilon = np.float32(1e-5)
    scale = 1.0 / np.sqrt(np.diag(v) + epsilon)
    xPCAWhite = np.diag(scale) @ xRot
    features_pca = xPCAWhite.T
    norm = np.linalg.norm(features_pca, axis=1, keepdims=True)
    features_pca = features_pca / (norm + np.float32(1e-12))
    return features_pca

def simple_pca_project_np(X: np.ndarray, dim: int):
    X = X.astype(np.float32)
    mu = np.mean(X, axis=0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    coeff = Vt.T
    score = (X @ coeff)[:, :dim]
    return score

class GWLModel:
    def __init__(self, dw_stats, dim):
        self.dw_stats = dw_stats
        self.dim = dim

    def transform(self, XTest: np.ndarray) -> np.ndarray:
        XTrain_fit = self.dw_stats["XTrain_fit"].astype(np.float32)
        XTest = XTest.astype(np.float32)
        Xout = xPCAWhitening_XY_np(XTrain_fit, XTest)
        Yout = xPCAWhitening_XY_np(Xout, XTrain_fit)
        gain = np.concatenate([Xout, Yout], axis=1)
        X_proj = simple_pca_project_np(gain, self.dim)
        norm = np.linalg.norm(X_proj, axis=1, keepdims=True)
        X_proj = X_proj / (norm + np.float32(1e-12))
        return X_proj

def gwl_fit(XTrain: np.ndarray, XTest: np.ndarray, dim: int) -> GWLModel:
    assert XTrain.shape == XTest.shape, "GWL fit要求XTrain,XTest维度一致，本任务传入同一个db_feat"
    dw_stats = {
        "XTrain_fit": XTrain.astype(np.float32).copy()
    }
    return GWLModel(dw_stats, dim)
# ===================== PCA‑p‑whitening 幂白化（复刻MATLAB） =====================
def PCA_p_whitening(features: np.ndarray, dim: int, p: int = 3, eps: float = 1e-5) -> np.ndarray:

    norm = np.linalg.norm(features, axis=1, keepdims=True)
    features_data = features / (norm + 1e-12)
    features_data = np.nan_to_num(features_data, nan=0.0)


    norm2 = np.linalg.norm(features_data, axis=1, keepdims=True)
    x_train = features_data / (norm2 + 1e-12)
    x_train = np.nan_to_num(x_train, nan=0.0)

    x_train = x_train.T   # D × N
    mu = np.mean(x_train, axis=1, keepdims=True)
    x_train = x_train - mu

    n_sample = x_train.shape[1]
    sigma = (x_train @ x_train.T) / n_sample

    U, S, _ = np.linalg.svd(sigma, full_matrices=True)


    x_test = features_data.T
    mu_test = np.mean(x_test, axis=1, keepdims=True)
    x_test = x_test - mu_test

    xRot = U.T @ x_test

    diag_S = np.diag(S)
    scale = 1.0 / ((diag_S + eps) ** (1.0 / p))
    xPCAWhite = np.diag(scale) @ xRot


    features_data = xPCAWhite[:dim, :].T


    norm_out = np.linalg.norm(features_data, axis=1, keepdims=True)
    features_pca = features_data / (norm_out + 1e-12)
    return features_pca


class PCAPWhiteningModel:

    def __init__(self, U: np.ndarray, mu_train: np.ndarray, mu_test_ref: np.ndarray, S: np.ndarray, dim: int, p: int, eps: float):
        self.U = U
        self.mu_train = mu_train
        self.mu_test_ref = mu_test_ref
        self.S = S
        self.dim = dim
        self.p = p
        self.eps = eps

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N,D)输入原始特征
        return: (N, dim)
        """
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        features_data = X / (norm + 1e-12)
        features_data = np.nan_to_num(features_data, nan=0.0)

        norm2 = np.linalg.norm(features_data, axis=1, keepdims=True)
        x_test = features_data / (norm2 + 1e-12)
        x_test = np.nan_to_num(x_test, nan=0.0)

        x_test = x_test.T
        mu_test = np.mean(x_test, axis=1, keepdims=True)
        x_test = x_test - mu_test

        xRot = self.U.T @ x_test
        diag_S = np.diag(self.S)
        scale = 1.0 / ((diag_S + self.eps) ** (1.0 / self.p))
        xPCAWhite = np.diag(scale) @ xRot

        features_data = xPCAWhite[:self.dim, :].T
        norm_out = np.linalg.norm(features_data, axis=1, keepdims=True)
        features_pca = features_data / (norm_out + 1e-12)
        return features_pca


def pca_p_whitening_fit(train_feat: np.ndarray, dim: int, p: int = 3, eps: float = 1e-5) -> PCAPWhiteningModel:
    """
    fit阶段：只用图库特征训练，保存U,S,mu等统计量
    :param train_feat: 图库db_feat (N,D)
    :param dim:输出维度
    :param p:幂指数
    :param eps:epsilon
    :return:PCAPWhiteningModel对象，调用transform处理图库/查询
    """
    norm = np.linalg.norm(train_feat, axis=1, keepdims=True)
    features_data = train_feat / (norm + 1e-12)
    features_data = np.nan_to_num(features_data, nan=0.0)

    norm2 = np.linalg.norm(features_data, axis=1, keepdims=True)
    x_train = features_data / (norm2 + 1e-12)
    x_train = np.nan_to_num(x_train, nan=0.0)

    x_train = x_train.T
    mu_train = np.mean(x_train, axis=1, keepdims=True)
    x_train = x_train - mu_train

    n_sample = x_train.shape[1]
    sigma = (x_train @ x_train.T) / n_sample
    U, S, _ = np.linalg.svd(sigma, full_matrices=True)

    # 保存参考均值
    x_test_ref = features_data.T
    mu_test_ref = np.mean(x_test_ref, axis=1, keepdims=True)

    return PCAPWhiteningModel(U, mu_train, mu_test_ref, S, dim, p, eps)