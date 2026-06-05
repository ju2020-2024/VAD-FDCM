import torch
from sklearn.metrics import auc, precision_recall_curve, average_precision_score, roc_curve
import numpy as np
from utils.tests import *
from utils.quality_fig import *


def plot_fig_zhe_xian(video_ranges,
                      file_paths,
                      save_root,
                      s_1,
                      s3,
                      gt):
    """
    绘制：
        1. without confidence 分数 (s_1)
        2. with confidence 分数 (s3)
        3. GT 粉色背景

    参数：
        video_ranges : [(start_frame, end_frame), ...]
        file_paths   : 视频路径列表
        save_root    : 保存目录
        s_1          : frame-level 分数
        s3           : frame-level 分数
        gt           : frame-level GT
    """

    os.makedirs(save_root, exist_ok=True)

    print("=" * 50)
    print("开始绘制折线图...")
    print(f"视频数量: {len(video_ranges)}")
    print("=" * 50)

    for idx, ((start, end), video_path) in enumerate(zip(video_ranges, file_paths)):

        # ---------------------------------------------------
        # 取当前视频数据
        # ---------------------------------------------------
        gt_video = gt[start:end + 1]

        s1_video = s_1[start:end + 1]
        s3_video = s3[start:end + 1]

        frames = np.arange(len(gt_video))

        # ---------------------------------------------------
        # 创建画布
        # ---------------------------------------------------
        plt.figure(figsize=(15, 4))

        # ---------------------------------------------------
        # 绘制GT背景（粉色）
        # ---------------------------------------------------
        in_abnormal = False
        abnormal_start = 0

        for i in range(len(gt_video)):

            # 异常开始
            if gt_video[i] == 1 and not in_abnormal:
                abnormal_start = i
                in_abnormal = True

            # 异常结束
            elif gt_video[i] == 0 and in_abnormal:
                plt.axvspan(
                    abnormal_start,
                    i,
                    color='pink',
                    alpha=0.35
                )
                in_abnormal = False

        # 如果异常持续到最后
        if in_abnormal:
            plt.axvspan(
                abnormal_start,
                len(gt_video),
                color='pink',
                alpha=0.35
            )

        # ---------------------------------------------------
        # 绘制两条曲线
        # ---------------------------------------------------
        plt.plot(
            frames,
            s1_video,
            linewidth=1.5,
            label='without confidence'
        )

        plt.plot(
            frames,
            s3_video,
            linewidth=1.5,
            label='with confidence'
        )

        # ---------------------------------------------------
        # 图像设置
        # ---------------------------------------------------
        video_name = os.path.basename(video_path)
        video_name = os.path.splitext(video_name)[0]

        plt.title(video_name, fontsize=14)

        plt.xlabel("Frame", fontsize=12)
        plt.ylabel("Anomaly Score", fontsize=12)

        plt.xlim([0, len(gt_video)])

        plt.ylim([0, 1])

        plt.legend()

        plt.tight_layout()

        # ---------------------------------------------------
        # 保存
        # ---------------------------------------------------
        save_path = os.path.join(
            save_root,
            f"{video_name}.png"
        )

        plt.savefig(save_path, dpi=300)

        plt.close()

        print(f"[{idx + 1}/{len(video_ranges)}] 保存: {save_path}")

    print("=" * 50)
    print("全部绘制完成")
    print("=" * 50)


def test(test_loader, model, gt, args, gt_path=False):
    model.eval()
    pred = torch.zeros(0).to(args.device)

    all_s1 = []
    all_s2 = []

    # gt对比图
    video_ranges = []
    current_frame = 0
    file_paths = test_loader.dataset.files

    # all_x_input = []
    # all_x_out = []
    # all_h_common = []
    # all_h_specific = []
    # all_gate = []
    # all_x = []

    is_plot_fig = False

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(args.device)

            out = model(inputs, None)
            s1, s2 = out["frame_scores"]

            segment_scores = (s1 + s2) / 2

            # s1_sigmoid = torch.sigmoid(s1)
            # s2_sigmoid = torch.sigmoid(s2)
            # s1_mean = torch.mean(s1_sigmoid, dim=0)
            # s2_mean = torch.mean(s2_sigmoid, dim=0)
            # all_s1.append(s1_mean.cpu())
            # all_s2.append(s2_mean.cpu())

            segment_scores = torch.sigmoid(segment_scores)
            segment_scores = torch.mean(segment_scores, 0)
            pred = torch.cat((pred, segment_scores))

            # x_input, x_out, h_common, h_specific, gate, x = out["csfd"]
            #
            # x_input = torch.mean(x_input, dim=0)
            # all_x_input.append(x_input.cpu())
            #
            # x_out = torch.mean(x_out, dim=0)
            # all_x_out.append(x_out.cpu())
            #
            # h_common = torch.mean(h_common, dim=0)
            # all_h_common.append(h_common.cpu())
            #
            # h_specific = torch.mean(h_specific, dim=0)
            # all_h_specific.append(h_specific.cpu())
            #
            # gate = torch.mean(gate, dim=0)
            # all_gate.append(gate.cpu())
            #
            # x = torch.mean(x, dim=0)
            # all_x.append(x.cpu())

            if gt_path:
                # gt对比图
                num_segments = segment_scores.shape[0]
                num_frames = num_segments * 16
                start_frame = current_frame
                end_frame = current_frame + num_frames - 1
                video_ranges.append((start_frame, end_frame))
                current_frame += num_frames

            if is_plot_fig:
                num_segments = segment_scores.shape[0]
                num_frames = num_segments * 16
                start_frame = current_frame
                end_frame = current_frame + num_frames - 1
                video_ranges.append((start_frame, end_frame))
                current_frame += num_frames

        pred_np = list(pred.cpu().detach().numpy())

        # all_s1 = torch.cat(all_s1, dim=0).numpy()
        # all_s2 = torch.cat(all_s2, dim=0).numpy()
        # all_s1 = np.repeat(all_s1, 16)
        # all_s2 = np.repeat(all_s2, 16)
        # plot_fig_zhe_xian(
        #     video_ranges,
        #     file_paths,
        #     "/home/stu2023/jj/fig_zhe_xian/ucf",
        #     all_s1,
        #     all_s2,
        #     gt
        # )

        precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred_np, 16))
        pr_auc = auc(recall, precision)
        fpr, tpr, threshold = roc_curve(list(gt), np.repeat(pred_np, 16))
        rec_auc = auc(fpr, tpr)

        # all_x_input = torch.cat(all_x_input, dim=0)
        # all_x_out = torch.cat(all_x_out, dim=0)
        # all_h_common = torch.cat(all_h_common, dim=0)
        # all_h_specific = torch.cat(all_h_specific, dim=0)
        # all_gate = torch.cat(all_gate, dim=0)
        # all_x = torch.cat(all_x, dim=0)

        if gt_path:
            # gt
            predict_dict = build_predict_dict(pred, video_ranges=video_ranges, file_paths=file_paths)
            generate_xd_json_from_gt(gt, video_ranges, file_paths,
                                     '/home/stu2023/jj/project/VAD_new_vision/list/ucf_test_gt.json')
            anomap(predict_dict, 'ucf', '/home/stu2023/jj/project/vad2/ucf/result/Fig/ucf_jiu_shi_yan_gt_png')

        # all_out = {
        #     "x_input": all_x_input,
        #     "x_out": all_x_out,
        #     "h_common": all_h_common,
        #     "h_specific": all_h_specific,
        #     "gate": all_gate,
        #     "x": all_x
        # }

        # return rec_auc, pr_auc, all_out
        return rec_auc, pr_auc, None
