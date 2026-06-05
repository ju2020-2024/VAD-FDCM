import torch
import torch.nn.functional as F


def coral_loss(f_v, f_a):
    # 计算CORAL损失
    # f_v, f_a: (B, T, D)
    b, t, d = f_v.shape
    f_v = f_v.reshape(-1, d)
    f_a = f_a.reshape(-1, d)
    cov_v = (f_v.T @ f_v) / (b * t - 1)
    cov_a = (f_a.T @ f_a) / (b * t - 1)
    return torch.mean((cov_v - cov_a) ** 2)


def cos_loss(fv, fa):
    # 计算余弦相似度损失
    cos = torch.nn.CosineSimilarity(dim=-1)
    return 1 - torch.mean(cos(fv, fa))
    # return torch.mean(cos(fv, fa))


def compute_csfd_loss(h_common, h_specific, h_out, h_input):
    # -----------------------------
    # 1. Orthogonality Loss (推荐)
    #    使用归一化 + inner product 的平方
    #    保证 loss >= 0 且梯度稳定
    # -----------------------------

    hc_norm = F.normalize(h_common, dim=-1)
    hs_norm = F.normalize(h_specific, dim=-1)
    L_orth = torch.mean(torch.sum(hc_norm * hs_norm, dim=-1) ** 2)

    # -----------------------------
    # 2. Sparsity Loss
    #    使 h_specific 稀疏，逼迫其提取"特解"
    # -----------------------------
    L_sparse = torch.mean(torch.abs(h_specific))

    # -----------------------------
    # 3. Residual Alignment Loss
    #    h_common + h_specific ≈ h_input
    # -----------------------------

    return L_orth, L_sparse
