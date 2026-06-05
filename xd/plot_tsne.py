import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ===================== 你只需要确认路径 =====================
NPY_FOLDER = "/home/stu2023/jj/project/vad2/xd/result/npy"
# ==========================================================

# 加载所有 npy，合并成三大特征 + 标签
all_x3 = []
all_com = []
all_spe = []
all_labels = []

npy_files = [f for f in os.listdir(NPY_FOLDER) if f.endswith(".npy")]

print(f"找到 {len(npy_files)} 个视频特征文件")

for fname in npy_files:
    path = os.path.join(NPY_FOLDER, fname)
    data = np.load(path, allow_pickle=True).item()

    # 取出三个特征（取整个视频所有帧）
    x3 = data["x3_input"]
    com = data["h_common"]
    spe = data["h_specific"]
    label = data["label"]

    all_x3.append(x3)
    all_com.append(com)
    all_spe.append(spe)
    all_labels.append(np.full(len(x3), label))

# 合并成大矩阵（t-SNE 输入格式）
X_x3 = np.concatenate(all_x3, axis=0)
X_com = np.concatenate(all_com, axis=0)
X_spe = np.concatenate(all_spe, axis=0)
Y = np.concatenate(all_labels, axis=0)

print(f"总帧数量：{len(Y)}")
print(f"正常帧：{np.sum(Y == 0)}, 异常帧：{np.sum(Y == 1)}")

# ===================== t-SNE 降维 =====================
print("正在运行 t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

Z_x3 = tsne.fit_transform(X_x3)
Z_com = tsne.fit_transform(X_com)
Z_spe = tsne.fit_transform(X_spe)

# ===================== 绘图（论文三图格式） =====================
plt.figure(figsize=(18, 5))

# 配色：蓝色正常，红色异常
colors = ["#377eb8", "#e41a1c"]
labels = ["Normal", "Anomaly"]


def plot_single(z, y, title, idx):
    plt.subplot(1, 3, idx)
    plt.scatter(z[y == 0, 0], z[y == 0, 1], c=colors[0], s=3, alpha=0.6, label=labels[0])
    plt.scatter(z[y == 1, 0], z[y == 1, 1], c=colors[1], s=3, alpha=0.6, label=labels[1])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.xticks([])
    plt.yticks([])


plot_single(Z_x3, Y, "(a) Before Disentanglement", 1)
plot_single(Z_com, Y, "(b) Normal-shared Features", 2)
plot_single(Z_spe, Y, "(c) Anomaly-specific Features", 3)

plt.tight_layout()
plt.savefig("tsne_NAD.png", dpi=600, bbox_inches="tight")
plt.show()

print("✅ t-SNE 图已保存：tsne_NAD.png")
