import torch
import torch.nn as nn
import torch.nn.functional as F


class CSFD(nn.Module):
    """
    Common–Specific Feature Disentangling (CSFD)
    最终稳定版 CSFD 模块（适配初版）
    - 无卷积
    - Common = MLP
    - Specific = Gate * h
    - 输出 h_out = h_common + h_specific
    - 附加返回项：h_common, h_specific, gate（用于计算三种 loss）
    """

    def __init__(self, feat_dim=128):
        super().__init__()

        # ====== Common branch：通解特征（MLP）======
        self.common_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        # ====== Specific branch：特解特征（Gate）======
        self.specific_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )

        # ====== Learnable balance gate: α ∈ (0,1) ======
        # 初始化为 0，让 h_out 初始更偏向 specific，通常更利于异常检测
        # self.alpha = nn.Parameter(torch.tensor(0.0))
        # self.linear=nn.Linear(256, 128)
        # self.bn=nn.BatchNorm1d(128)
        # self.relu=nn.ReLU(inplace=True)

    def forward(self, h):
        """
        输入：h (B, T, C)
        返回：
            h_out: 分解后的特征 (B, T, C)
            h_common: 通用特征 (B, T, C)
            h_specific: 特殊特征 (B, T, C)
            gate: 特解门控 (B, T, C)
        """

        # 通解
        h_common = self.common_mlp(h)

        # 特解
        gate = self.specific_gate(h)
        h_specific = gate * h  # 特解 = gate * 输入

        # alpha = torch.sigmoid(self.alpha)

        # 最终特征合成
        h_out = h_common + h_specific
        # h_out = alpha * h_common + (1 - alpha) * h_specific
        # h_out=torch.cat((h_common,h_specific),dim=-1)
        #
        # h_out=self.linear(h_out)
        # h_out = h_out.permute(0, 2, 1)
        # h_out=self.bn(h_out)
        # h_out = h_out.permute(0, 2, 1)
        # h_out=self.relu(h_out)

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
    def __init__(self, input_dim=1024, hidden_dim=128, num_layers=3, relu_rate=0.0, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_c = input_dim if i == 0 else hidden_dim
            layers.append(TemporalConvBlock(in_c, hidden_dim, kernel_size=3, dilation=dilation, relu_rate=relu_rate,
                                            dropout=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        out = self.net(x)
        return out.permute(0, 2, 1)  # (B, T, C)


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
        x_cat = torch.cat([x1, x2], dim=-1)  # [B, T, 2D]
        attn_weights = self.attn(x_cat)  # [B, T, 1]
        return attn_weights * x1 + (1 - attn_weights) * x2  # [B, T, D]


class TemporalSpatialModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tcn_model_1 = TCNStack(input_dim=1024, hidden_dim=256, num_layers=3, relu_rate=args.relu_rate,
                                    dropout=args.dropout)
        self.tcn_model_2 = TCNStack(input_dim=1152, hidden_dim=256, num_layers=3, relu_rate=args.relu_rate,
                                    dropout=args.dropout)

        self.gru_model = nn.GRU(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)
        self.fusion = AttentionFusion(feat_dim=128, relu_rate=args.relu_rate)

        self.csfd = CSFD(feat_dim=128)

        # ====== 新增：Confidence MIL 的映射层 ======
        # 用 128 维 specific → 1 个 confidence α
        self.conf_fc = nn.Linear(128, 1)

        # ====== 方案 A：温度系数 τ（Temperature Softmax） ======
        # 使 softmax 更平滑，减少 MIL 对排名的极端敏感性（最推荐）
        self.tau = args.tau  # τ > 1 -> 更平滑的注意力权重（可调 1.2~2.0） 默认1.5
        self.g = args.g
        self.ratioK = args.ratioK

    def forward(self, x, seq_len):
        xv = x[:, :, :1024]
        xa = x[:, :, 1024:]

        xv = self.tcn_model_1(xv)
        x1, _ = self.gru_model(xv)
        s1 = self.fc(x1)
        mil_s1 = self.mil_scores(s1, seq_len)

        x2_input = self.tcn_model_2(x)
        x2, _ = self.gru_model(x2_input)
        s2 = self.fc(x2)
        mil_s2 = self.mil_scores(s2, seq_len)

        x3 = self.fusion(x1, x2)

        # ===== 加入 CSFD 分解模块 =====
        x3_input = x3
        x3_out, h_common, h_specific, gate = self.csfd(x3)

        s3 = self.fc(x3_out)
        mil_s3 = self.confidence_mil(s3.squeeze(-1), h_specific, seq_len)

        return {
            "frame_scores": [s1, s2, s3],
            "video_scores": [mil_s1, mil_s2, mil_s3],
            "csfd": [x3_input, x3_out, h_common, h_specific, gate],
            "x": [xv, xa, x]
        }

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

    # ====== 新增：Confidence-Aware MIL，作为 Innovation 2 ======
    def confidence_mil(self, scores, h_specific, seq_len=None):
        """
        置信度感知的多实例学习（Confidence-aware Multiple Instance Learning）方法
        用于将帧级别的异常分数聚合为视频级别的异常分数
        
        参数：
            scores: (B, T) 形状的张量，B为批次大小，T为时间步长，包含每个帧的异常分数
            h_specific: (B, T, C) 形状的张量，C为特征维度，包含每个帧的特定特征表示
            seq_len: 可选参数，包含每个视频的有效帧长度的张量，用于处理不同长度的视频序列
        
        返回：
            (B,) 形状的张量，包含每个视频的置信度感知MIL异常分数
        """
        B, T = scores.shape  # 获取批次大小和时间步长
        out_scores = []  # 初始化输出分数列表

        # 遍历每个视频样本
        for i in range(B):
            # 计算当前视频的有效帧长度（使用实际长度或默认长度）
            T_eff = seq_len[i].item() if seq_len is not None else T

            # 获取当前视频的有效帧分数和特征
            s_i = scores[i][:T_eff]  # 当前视频的有效帧异常分数 (Ti,)
            h_i = h_specific[i][:T_eff]  # 当前视频的有效帧特定特征 (Ti, C)

            # ===== 1. 计算固定比例的top-k值，保持MIL的稳定性 =====
            k = max(1, int(T_eff * self.ratioK))  # k为有效帧长度的一定比例（至少为1）
            topk_vals, topk_idx = torch.topk(s_i, k)  # 获取top-k的异常分数值和对应的索引

            # ===== 2. 根据特定特征生成帧级别的置信度权重 =====
            # 使用全连接层将特定特征映射为置信度，并通过sigmoid归一化到[0,1]范围
            # 形状变化: (Ti, C) → (Ti, 1) → (Ti,)
            conf = torch.sigmoid(self.conf_fc(h_i)).squeeze(-1)

            # 取出top-k异常分数对应的置信度值
            conf_topk = conf[topk_idx]

            # ===== 3. 计算置信度感知的注意力权重 =====
            lam = self.g  # 置信度增益系数，用于控制置信度对权重的影响程度
            # 使用温度系数τ的softmax计算原始权重，τ>1使权重分布更平滑
            weights_orig = F.softmax(topk_vals / self.tau, dim=0)
            # 根据置信度调整权重：置信度越高，对应异常分数的权重越大
            weights_C = weights_orig * (1 + lam * conf_topk)
            # 归一化权重，确保权重和为1
            weights_C = weights_C / (weights_C.sum() + 1e-6)  # 加入小常数避免除零错误

            # 使用置信度感知权重对top-k异常分数进行加权求和，得到视频级异常分数
            out_scores.append(torch.sum(topk_vals * weights_C))

        # 将所有视频的异常分数堆叠为张量并返回
        return torch.stack(out_scores, dim=0)
