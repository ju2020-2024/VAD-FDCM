import torch
import torch.nn as nn
import torch.nn.functional as F


class CSFD(nn.Module):
    """
    Common–Specific Feature Disentangling (CSFD)
    - Common = MLP
    - Specific = Gate * h
    - 输出 h_out = h_common + h_specific
    - 返回 h_out, h_common, h_specific, gate
    """

    def __init__(self, feat_dim=128):
        super().__init__()
        self.common_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.specific_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )

    def forward(self, h):
        h_common = self.common_mlp(h)
        gate = self.specific_gate(h)
        h_specific = gate * h
        h_out = h_common + h_specific
        return h_out, h_common, h_specific, gate


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, relu_rate=0.0, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation // 2,
                              dilation=dilation)
        self.relu = nn.LeakyReLU(negative_slope=relu_rate)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.downsample(x)


class TCNStack(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=3, relu_rate=0.0, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_c = input_dim if i == 0 else hidden_dim
            layers.append(
                TemporalConvBlock(in_c, hidden_dim, kernel_size=3,
                                  dilation=dilation, relu_rate=relu_rate, dropout=dropout)
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, C) -> permute to (B, C, T) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.net(x)
        return out.permute(0, 2, 1)


class AttentionFusion(nn.Module):
    def __init__(self, feat_dim, relu_rate=0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LeakyReLU(negative_slope=relu_rate),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x_cat = torch.cat([x1, x2], dim=-1)
        attn_weights = self.attn(x_cat)
        return attn_weights * x1 + (1 - attn_weights) * x2


class TemporalSpatialModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # TCN 输入维度 = 2048（UCF 特征维度）
        self.tcn_model = TCNStack(input_dim=2048, hidden_dim=256,
                                  num_layers=3, relu_rate=args.relu_rate, dropout=args.dropout)

        self.gru_model = nn.GRU(input_size=256, hidden_size=64,
                                batch_first=True, bidirectional=True)

        self.fc = nn.Linear(128, 1)

        # Fusion 保持接口（这里为可扩展保留）
        self.fusion = AttentionFusion(feat_dim=128, relu_rate=args.relu_rate)

        # CSFD 模块（分解 common / specific）
        # self.csfd = CSFD(feat_dim=128)

        # Confidence MIL 映射层（specific -> confidence scalar）
        self.conf_fc = nn.Linear(128, 1)

        # Temperature & gamma 控制 confidence-aware pooling
        # self.tau = args.tau
        # self.g = args.g

    def mil_scores(self, segment_scores, seq_len):
        segment_scores = segment_scores.squeeze(-1)  # (B, T)
        instance_scores = []
        for i in range(segment_scores.shape[0]):
            length = seq_len[i] if seq_len is not None else segment_scores.shape[1]
            k = max(1, int(length * 0.1))
            topk_vals, _ = torch.topk(segment_scores[i][:length], k=k, largest=True)
            weights = F.softmax(topk_vals, dim=0)
            tmp = torch.sum(topk_vals * weights).view(1)
            instance_scores.append(tmp)
        instance_scores = torch.cat(instance_scores, dim=0)
        return instance_scores

    # def confidence_mil(self, scores, h_specific, seq_len=None):
    #     """
    #     scores: (B, T)# 添加消融实验控制参数
    #     h_specific: (B, T, C)
    #     """
    #     B, T = scores.shape
    #     out_scores = []
    #
    #     for i in range(B):
    #         # 有效长度
    #         T_eff = seq_len[i].item() if seq_len is not None else T
    #
    #         s_i = scores[i][:T_eff]  # (Ti,)
    #         h_i = h_specific[i][:T_eff]  # (Ti, C)
    #
    #         # ===== 1. 固定 top-k，不改变 MIL 稳定性 =====
    #         k = max(1, int(T_eff * 0.1))
    #         topk_vals, topk_idx = torch.topk(s_i, k)
    #
    #         # ===== 2. 根据 h_specific 生成置信度 α =====
    #         # shape: (Ti, 1) → (Ti,)
    #         conf = torch.sigmoid(self.conf_fc(h_i)).squeeze(-1)
    #
    #         # 取出 top-k 部分的 α
    #         conf_topk = conf[topk_idx]
    #         lam = self.g
    #         weights_orig = F.softmax(topk_vals / self.tau, dim=0)
    #         weights_C = weights_orig * (1 + lam * conf_topk)
    #         weights_C = weights_C / (weights_C.sum() + 1e-6)
    #         out_scores.append(torch.sum(topk_vals * weights_C))
    #
    #     return torch.stack(out_scores, dim=0)

    def forward(self, x, seq_len):
        """
        x: (B, T, 2048)  -- visual-only input
        seq_len: tensor of effective lengths
        """
        # 直接使用输入（不截取）
        xv = x  # xv: (B, T, 2048)

        # TCN -> (B, T, 256)
        xv = self.tcn_model(xv)

        # BiGRU -> (B, T, 128)
        x1, _ = self.gru_model(xv)

        # frame-level score 1 (baseline)
        s1 = self.fc(x1)
        mil_s1 = self.mil_scores(s1, seq_len)

        # fusion (here fuse x1 with itself to keep API consistent)
        x_fused = self.fusion(x1, x1)
        x_fused_score = self.fc(x_fused)
        mil_s2 = self.mil_scores(x_fused_score, seq_len)

        # CSFD 分解（common / specific）
        # x3_out, h_common, h_specific, gate = self.csfd(x_fused)

        # frame-level score 3 (after CSFD)
        # s3 = self.fc(x3_out)
        # mil_s3 = self.confidence_mil(s3.squeeze(-1), h_specific, seq_len)

        # return {
        #     "frame_scores": [s1, s3],
        #     "video_scores": [mil_s1, mil_s3],
        #     # "csfd": [x_fused, x3_out, h_common, h_specific, gate, x]
        # }

        return {
            "frame_scores": [s1, x_fused_score],
            "video_scores": [mil_s1, mil_s2],
        }
