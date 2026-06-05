import os
import torch
import numpy as np
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def build_predict_dict(pred, video_ranges, file_paths):
    """
    构建 predict_dict，用于绘图

    输入:
        pred: torch.Tensor, 所有视频的段预测结果（拼接后）
        video_ranges: list of (start_frame, end_frame)，每个视频的帧范围
        file_paths: list of str，测试集中每个视频的 RGB 或音频路径

    输出:
        predict_dict: list[dict], 可用于 anomap()
    """
    pred = pred.view(-1, 1).cpu()  # 确保 shape 为 [T_total, 1]
    predict_dict = []
    pointer = 0  # pred 中的位置

    for i, (start, end) in enumerate(video_ranges):
        num_frames = end - start + 1
        num_segments = num_frames // 16
        video_pred = pred[pointer:pointer + num_segments]
        pointer += num_segments

        # ✅ 统一处理 file_name
        raw_name = os.path.basename(file_paths[i])
        if raw_name.endswith('__vggish.npy'):
            file_name = raw_name.replace('__vggish.npy', '')
        elif raw_name.endswith('.npy'):
            file_name = raw_name.rsplit('__', 1)[0]
        else:
            file_name = raw_name

        predict_dict.append({
            "file_name": file_name,
            "pre_dict": video_pred
        })

    return predict_dict





import os
import json

def generate_xd_json_from_gt(gt_path, video_ranges, video_names, output_json):
    """
    根据 gt.npy 和视频帧范围生成 xd 格式的 ground truth json 文件

    Args:
        gt_path (np.ndarray): gt.npy 加载后的数组，形状为 [所有帧]
        video_ranges (List[Tuple[int, int]]): 每个视频的起止帧 (start, end)
        video_names (List[str]): 每个视频的原始路径，需从中提取 file_name
        output_json (str): 保存 json 的目标路径
    """
    gt = gt_path
    assert len(video_ranges) == len(video_names), "video_ranges 和 video_names 数量应一致"

    gt_dict = {}
    for (start, end), path in zip(video_ranges, video_names):
        # ✅ 统一处理 file_name
        raw_name = os.path.basename(path)
        if raw_name.endswith('__vggish.npy'):
            file_name = raw_name.replace('__vggish.npy', '')
        elif raw_name.endswith('.npy'):
            file_name = raw_name.rsplit('__', 1)[0]
        else:
            file_name = raw_name

        labels = gt[start:end + 1].tolist()
        gt_dict[file_name] = {
            "num_frames": len(labels),
            "labels": labels
        }

    with open(output_json, "w") as f:
        json.dump(gt_dict, f, indent=2)

    print(f"✅ Saved JSON to {output_json}")





def draw_tsne(features, labels, save_path='/home/stu2023/jj/project/fig/t-sne/tsne.pdf'):
    tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=30)
    tsne_result = tsne.fit_transform(features)

    labels = np.array(labels)
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], s=3, c='red', label='Abnormal')
    plt.scatter(tsne_result[labels==0, 0], tsne_result[labels==0, 1], s=3, c='blue', label='Normal')

    plt.legend()
    plt.title("t-SNE of Test Set Features")
    plt.savefig(save_path)
    plt.close()


def plot_alpha_heatmap(alpha, video_idx, save_dir="/home/stu2023/jj/project/fig/hotfig"):
    """
    alpha: ndarray, shape (T,) or (1, T) or (T, 1) — attention weights over time
    video_idx: int or str — used to name the output image file
    save_dir: str — directory to save the figure
    """
    # 确保 alpha 是一维数组
    alpha = np.squeeze(alpha)

    # 创建保存目录（若不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 画图
    plt.figure(figsize=(10, 3))
    plt.imshow(alpha[np.newaxis, :], aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight (α)')
    plt.yticks([])  # 不显示y轴刻度
    plt.xlabel('Time Step')
    plt.title(f'Attention Weights Heatmap (α) — Video {video_idx}')
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(save_dir, f'alpha_heatmap_video{video_idx}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"[✓] Heatmap saved to: {save_path}")


import numpy as np
import matplotlib.pyplot as plt
import os


def plot_feature_heatmaps(xv, xva, xf, video_idx, save_dir="/home/stu2023/jj/project/fig/hot_xvxfxva"):
    """
    Plot heatmaps for three features (xv, xva, xf) in a single figure.

    Parameters:
        xv, xva, xf: np.ndarray of shape (B, T, D)
            Feature matrices for visual, visual+audio, and fused respectively.
        video_idx: str or int
            Identifier for the video (used in filename).
        save_dir: str
            Directory where the heatmap image will be saved.
    """
    # Take mean over batch dimension
    xv = xv.mean(axis=0)  # shape: (T, D)
    xva = xva.mean(axis=0)
    xf = xf.mean(axis=0)

    os.makedirs(save_dir, exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

    feature_names = ['xv (Visual)', 'xva (Visual+Audio)', 'xf (Fused)']
    features = [xv, xva, xf]

    for ax, feat, name in zip(axs, features, feature_names):
        im = ax.imshow(feat.T, aspect='auto', cmap='viridis', origin='lower')
        ax.set_ylabel(name)
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02)

    axs[-1].set_xlabel("Time Step")
    fig.suptitle(f'Feature Heatmaps — Video {video_idx}')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(save_dir, f'feature_heatmaps_video{video_idx}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"[✓] Heatmap saved to: {save_path}")


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_heatmaps_with_attention(x1, x2, x3, alpha, video_idx="001", save_dir="/home/stu2023/jj/project/fig/fighot"):
    """
    绘制 x^v, x^va, x^f 的特征热力图，并在 x^f 上叠加 attention 曲线。
    """
    # 创建保存目录（若不存在）
    os.makedirs(save_dir, exist_ok=True)

    # squeeze alpha 成 (T,)
    alpha = np.squeeze(alpha)  # (T,)

    # 平均特征 over batch 维度，变成 (T, 128)
    x1 = np.mean(x1, axis=0)  # (T, 128)
    x2 = np.mean(x2, axis=0)
    x3 = np.mean(x3, axis=0)

    # 统一颜色范围
    vmin = min(x1.min(), x2.min(), x3.min())
    vmax = max(x1.max(), x2.max(), x3.max())

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # x^v
    im0 = axs[0].imshow(x1.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_title('x^v (Visual-only Feature)')
    axs[0].set_ylabel('Feature Dim')

    # x^va
    im1 = axs[1].imshow(x2.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_title('x^va (Visual+Audio Feature)')
    axs[1].set_ylabel('Feature Dim')

    # x^f with alpha
    im2 = axs[2].imshow(x3.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axs[2].set_title('x^f (Fused Feature with Attention α)')
    axs[2].set_ylabel('Feature Dim')
    axs[2].set_xlabel('Time Step')

    # Attention 曲线叠加在 x^f 上
    T = alpha.shape[0]
    axs[2].plot(np.arange(T), alpha * x3.shape[1], color='red', linewidth=2, label='Attention α')
    axs[2].legend(loc='upper right')

    # 加 colorbar
    fig.colorbar(im2, ax=axs, orientation='vertical', fraction=0.015, pad=0.01, label='Feature Value')

    # 保存
    save_path = os.path.join(save_dir, f'feature_heatmap_with_attention_video{video_idx}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Feature heatmap with attention saved to: {save_path}")

