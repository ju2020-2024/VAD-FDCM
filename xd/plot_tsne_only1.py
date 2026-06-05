import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

NPY_FOLDER = "/home/stu2023/jj/project/vad2/xd/result/npy"
SAVE_FOLDER = "/home/stu2023/jj/project/vad2/xd/result/Fig/npytsne_frame"
os.makedirs(SAVE_FOLDER, exist_ok=True)

FRAME_PER_SEG = 16
MAX_FRAMES = 2000  # 降低采样帧数，避免内存爆炸
PERPLEXITY = 30

npy_files = [f for f in os.listdir(NPY_FOLDER) if f.endswith(".npy")]
print(f"✅ 找到 {len(npy_files)} 个视频，开始批量绘制帧级 t-SNE...\n")


def plot_frame_tsne(video_data, video_name, save_path):
    try:
        x3_seg = video_data["x3_input"]
        com_seg = video_data["h_common"]
        spe_seg = video_data["h_specific"]
        frame_labels = video_data["frame_labels"]  # 应该是已经截断过的

        # 扩展特征到帧级
        x3_frame = np.repeat(x3_seg, repeats=FRAME_PER_SEG, axis=0)
        com_frame = np.repeat(com_seg, repeats=FRAME_PER_SEG, axis=0)
        spe_frame = np.repeat(spe_seg, repeats=FRAME_PER_SEG, axis=0)

        # 维度对齐（保险）
        min_len = min(x3_frame.shape[0], len(frame_labels))
        if x3_frame.shape[0] != len(frame_labels):
            print(f"  ⚠️ 对齐维度: 特征 {x3_frame.shape[0]} vs 标签 {len(frame_labels)} -> 使用 {min_len}")
            x3_frame = x3_frame[:min_len]
            com_frame = com_frame[:min_len]
            spe_frame = spe_frame[:min_len]
            frame_labels = frame_labels[:min_len]

        n_frames = min_len
        # 下采样
        if n_frames > MAX_FRAMES:
            idx = np.random.choice(n_frames, MAX_FRAMES, replace=False)
            x3_frame = x3_frame[idx]
            com_frame = com_frame[idx]
            spe_frame = spe_frame[idx]
            frame_labels = frame_labels[idx]
            n_frames = MAX_FRAMES
            print(f"  ⚡ 下采样至 {n_frames} 帧")

        # t-SNE 参数
        perp = min(PERPLEXITY, n_frames - 1) if n_frames > 1 else 1
        print(f"  t-SNE perplexity = {perp}")

        # 分别降维（使用 exact 方法避免内存问题）
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, method='exact')
        z_x3 = tsne.fit_transform(x3_frame)
        z_com = tsne.fit_transform(com_frame)
        z_spe = tsne.fit_transform(spe_frame)

        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = ["(a) Original Feature", "(b) Normal-shared Feature", "(c) Anomaly-specific Feature"]
        colors = ["#377eb8", "#e41a1c"]
        for ax, z, title in zip(axes, [z_x3, z_com, z_spe], titles):
            ax.scatter(z[frame_labels == 0, 0], z[frame_labels == 0, 1],
                       c=colors[0], s=6, alpha=0.5, label="Normal")
            ax.scatter(z[frame_labels == 1, 0], z[frame_labels == 1, 1],
                       c=colors[1], s=6, alpha=0.5, label="Anomaly")
            ax.set_title(title, fontweight='bold', fontsize=14)
            ax.legend(fontsize=12)
            ax.set_xticks([]);
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"  ❌ 绘图失败: {str(e)}")
        plt.close()
        return False


# 批量处理
for idx, video_name in enumerate(npy_files):
    video_path = os.path.join(NPY_FOLDER, video_name)
    try:
        data = np.load(video_path, allow_pickle=True).item()
        if "frame_labels" not in data:
            print(f"⚠️  {video_name} 缺少 frame_labels，请重新运行特征提取脚本")
            continue
        base_name = video_name.replace(".npy", "").replace("__vggish", "")
        save_path = os.path.join(SAVE_FOLDER, f"tsne_{base_name}.png")
        print(f"[{idx + 1}/{len(npy_files)}] 处理 {video_name} ...")
        success = plot_frame_tsne(data, video_name, save_path)
        if success:
            print(f"  ✅ 已保存: {save_path}\n")
        else:
            print(f"  ❌ 跳过\n")
    except Exception as e:
        print(f"❌ 加载失败 {video_name}: {str(e)}\n")
        continue

print(f"\n🎉 完成！图片保存至：{SAVE_FOLDER}")