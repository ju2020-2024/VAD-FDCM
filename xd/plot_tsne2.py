import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ===================== 自动匹配 infer3.py 的路径 =====================
NPY_FOLDER = "/home/stu2023/jj/project/vad2/xd/result/npy_s1_s3"
# ====================================================================

# 加载所有 npy，只保留 s1, s2, s3
all_s1 = []
all_s2 = []
all_s3 = []
all_labels = []

npy_files = [f for f in os.listdir(NPY_FOLDER) if f.endswith(".npy")]
print(f"找到 {len(npy_files)} 个分数文件")

for fname in npy_files:
    path = os.path.join(NPY_FOLDER, fname)
    data = np.load(path, allow_pickle=True).item()

    # 读取并展平成 1D 向量（修复关键！）
    s1 = data["s1"].ravel()
    s2 = data["s2"].ravel()
    s3 = data["s3"].ravel()
    label = data["label"]

    all_s1.append(s1)
    all_s2.append(s2)
    all_s3.append(s3)
    all_labels.append(np.full(len(s1), label))

# 合并所有帧
X_s1 = np.concatenate(all_s1, axis=0)
X_s2 = np.concatenate(all_s2, axis=0)
X_s3 = np.concatenate(all_s3, axis=0)
Y = np.concatenate(all_labels, axis=0)

print(f"总帧数量：{len(Y)}")
print(f"正常帧：{np.sum(Y==0)}, 异常帧：{np.sum(Y==1)}")

# ===================== 拼接成正确的 2D 矩阵 [N, 3] =====================
X_scores = np.column_stack([X_s1, X_s2, X_s3])

# ===================== t-SNE 降维 =====================
print("正在运行 t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
Z = tsne.fit_transform(X_scores)

# ===================== 绘图 =====================
plt.figure(figsize=(8, 6))

colors = ["#377eb8", "#e41a1c"]
labels = ["Normal", "Anomaly"]

plt.scatter(Z[Y==0,0], Z[Y==0,1], c=colors[0], s=5, alpha=0.6, label=labels[0])
plt.scatter(Z[Y==1,0], Z[Y==1,1], c=colors[1], s=5, alpha=0.6, label=labels[1])
plt.title("t-SNE of Frame Scores (s1+s2+s3)", fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("tsne_scores_combined.png", dpi=600, bbox_inches="tight")
plt.show()

print("✅ t-SNE 图已保存：tsne_scores_combined.png")