import numpy as np
import matplotlib.pyplot as plt

# 模拟数据
T_eff = 20  # 有效帧长度
frame_scores = np.random.uniform(0, 1, T_eff)  # 帧级异常分数
confidence = np.random.uniform(0.1, 0.9, T_eff)  # 帧级置信度
ratioK = 0.2  # top-k比例
k = max(1, int(T_eff * ratioK))  # 取top-4帧

# 排序并获取top-k
topk_indices = np.argsort(frame_scores)[-k:][::-1]  # top-k索引
topk_scores = frame_scores[topk_indices]  # top-k分数
topk_conf = confidence[topk_indices]  # top-k对应的置信度

# 绘制时序图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(T_eff), frame_scores, 'bo-', label='Frame Scores')
plt.plot(topk_indices, topk_scores, 'ro-', label=f'Top-{k} Scores')
plt.xlabel('Frame Index')
plt.ylabel('Anomaly Score')
plt.title('Frame-level Anomaly Scores with Top-k Selection')
plt.legend()

# 绘制置信度对权重的影响
tau = 1.5  # 温度系数
lam = 0.5  # 置信度增益系数

# 计算权重
weights_orig = np.exp(topk_scores / tau) / np.sum(np.exp(topk_scores / tau))
weights_C = weights_orig * (1 + lam * topk_conf)
weights_C = weights_C / np.sum(weights_C)  # 归一化

plt.subplot(1, 2, 2)
plt.bar(range(k), weights_orig, alpha=0.5, label='Original Weights')
plt.bar(range(k), weights_C, alpha=0.8, label='Confidence-aware Weights')
plt.xlabel(f'Top-{k} Frame Rank')
plt.ylabel('Weight')
plt.title('Comparison of Original vs Confidence-aware Weights')
plt.legend()

plt.tight_layout()
plt.savefig('confidence_mil_explanation.png', dpi=300)
plt.show()