from torch.utils.data import DataLoader
import torch
import numpy as np
import model
import datasets
from test import test
import option
import time
import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def repeat_segment_features(features, repeat_num=16):
    """
    将 segment-level feature 扩展到 frame-level

    输入:
        [B,T,C]

    输出:
        [B,T*repeat_num,C]
    """

    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()

    B, T, C = features.shape

    # 在时间维重复
    features = np.repeat(features, repeat_num, axis=1)

    return features


def visualize_tsne(
        features,
        labels=None,
        save_path='tsne.png',
        title='t-SNE Visualization',
        max_points=5000,
        perplexity=30,
        random_state=42):
    """
    features:
        Tensor or ndarray
        支持:
            [B,T,C]
            [N,T,C]
            [N,C]

    labels:
        frame-level label
        shape:
            [B,T]
            [N]

    """

    # =========================================================
    # tensor -> numpy
    # =========================================================
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()

    if labels is not None and torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()

    # =========================================================
    # reshape feature
    # =========================================================
    if len(features.shape) == 3:
        # [B,T,C] -> [B*T,C]
        B, T, C = features.shape
        features = features.reshape(B * T, C)

        if labels is not None:
            labels = labels.reshape(B * T)

    elif len(features.shape) == 2:
        # already [N,C]
        pass

    else:
        raise ValueError(f'Unsupported feature shape: {features.shape}')

    # =========================================================
    # 随机采样
    # =========================================================
    N = features.shape[0]

    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        features = features[idx]

        if labels is not None:
            labels = labels[idx]

    print(f't-SNE samples: {features.shape[0]}')

    # =========================================================
    # 标准化
    # =========================================================
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # =========================================================
    # t-SNE
    # =========================================================
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init='pca',
        learning_rate='auto'
    )

    tsne_features = tsne.fit_transform(features)

    # =========================================================
    # plot
    # =========================================================
    plt.figure(figsize=(8, 8))

    if labels is None:

        plt.scatter(
            tsne_features[:, 0],
            tsne_features[:, 1],
            s=5
        )

    else:

        normal_idx = labels == 0
        abnormal_idx = labels == 1

        plt.scatter(
            tsne_features[normal_idx, 0],
            tsne_features[normal_idx, 1],
            s=5,
            label='Normal',
            alpha=0.7
        )

        plt.scatter(
            tsne_features[abnormal_idx, 0],
            tsne_features[abnormal_idx, 1],
            s=5,
            label='Abnormal',
            alpha=0.7
        )

        plt.legend()

    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    print(f'Saved t-SNE to: {save_path}')

    plt.close()


def frame_to_segment_labels(gt, segment_size=16):
    total_segments = len(gt) // segment_size

    gt = gt[:total_segments * segment_size]

    gt = gt.reshape(total_segments, segment_size)

    segment_gt = np.max(gt, axis=1)

    return segment_gt


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    print('perform testing...')
    args = option.parser.parse_args()
    args.device = 'cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu'

    test_loader = DataLoader(datasets.AllDataset(args=args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model = model.TemporalSpatialModel(args)

    # 新增调试代码
    print("=== 调试信息 ===")
    print(f"传入的设备参数 args.device: {args.device}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"可用GPU编号: 0 ~ {torch.cuda.device_count() - 1}")
    print("================")

    model = model.to(args.device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in
         torch.load('ckpt/best_model.pkl', map_location=torch.device('cpu')).items()})
    gt = np.load(args.gt)
    st = time.time()

    pr_auc, pr_ap, out = test(test_loader, model, gt, args, False, False)

    """
    需要进行tsne画的特征
    """
    x_input = out["x_input"]
    x_out = out["x_out"]
    h_common = out["h_common"]
    h_specific = out["h_specific"]
    gate = out["gate"]
    x = out["x"]

    segment_gt = frame_to_segment_labels(gt)

    print(x_input.shape)
    print(x_out.shape)
    print(segment_gt.shape)
    visualize_tsne(
        x_input,
        labels=segment_gt,
        save_path="/home/stu2023/jj/project/vad2/xd/result/Fig/tsne_x_input.png",
        title=''
    )

    visualize_tsne(
        x_out,
        labels=segment_gt,
        save_path="/home/stu2023/jj/project/vad2/xd/result/Fig/tsne_x_out.png",
        title=''
    )

    visualize_tsne(
        h_common,
        labels=segment_gt,
        save_path="/home/stu2023/jj/project/vad2/xd/result/Fig/tsne_h_common.png",
        title=''
    )

    visualize_tsne(
        h_specific,
        labels=segment_gt,
        save_path="/home/stu2023/jj/project/vad2/xd/result/Fig/tsne_h_specific.png",
        title=''
    )
    visualize_tsne(
        x,
        labels=segment_gt,
        save_path="/home/stu2023/jj/project/vad2/xd/result/Fig/tsne_x.png",
        title=''
    )

    time_elapsed = time.time() - st
    print('AUC: {:.4f}   AP:  {:.4f} \n'.format(pr_auc, pr_ap))
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
