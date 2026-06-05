import os
import json
import numpy as np
import matplotlib.pyplot as plt


def anomap(predict_dict, dataset, save_root):
    if os.path.exists(os.path.join(save_root, 'plot', dataset)) == 0:
        os.makedirs(os.path.join(save_root, 'plot', dataset))

    if dataset in ['sh', 'shanghai']:
        json_root = 'shanghaitech_ground_truth.testing.json'
        # json_root = '/home/stu2023/lyg/quality/shanghaitech_ground_truth.testing.json'
    elif dataset == 'ucf':
        json_root = 'ucf-crime_truth.testing.json'
        # json_root = '/home/stu2023/lyg/quality/ucf-crime_ground_truth.testing.json'
    elif dataset == 'xd':
        json_root = '/home/stu2023/jj/project/VAD_new_vision/list/xd_test_gt.json'
        # json_root = '/home/stu2023/lyg/quality/xd-violence_ground_truth.testing.json'

    # with open(file=os.path.join(save_root, 'list', json_root), mode='rb') as f:
    with open(file=os.path.join(save_root, json_root), mode='rb') as f:
        content = f.read()
        label_dict = json.loads(content)

    # for k, v in predict_dict.items():
    index =0
    for item in predict_dict:
        k, v = item['file_name'], item['pre_dict']
        predict_np = np.repeat(v.squeeze(-1).cpu().numpy(), 16)
        # label_np = np.array(label_dict[str(k[0])]['labels'])
        label_np = np.array(label_dict[k]['labels'])
        x = np.arange(len(predict_np))
        # plt.title(str(k[0]))
        # plt.title(k)

        # plt.title()
        plt.plot(x, predict_np, color='b', linewidth=1)
        plt.fill_between(x, label_np, where=label_np > 0.0, facecolor="red", alpha=0.3)
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xlabel('Frames')
        plt.ylabel('Anomaly scores')
        # plt.grid(True, linestyle='-.')
        # plt.legend().remove()
        # plt.show()
        # plt.savefig(os.path.join(save_root, 'plot', dataset, str(k[0]) + '.png'))
        plt.savefig(os.path.join(save_root, 'plot', dataset, f"{index:03d}_{k}.png"))
        # plt.savefig(os.path.join(save_root, 'plot', dataset, f"{index:03d}_{k}.svg"))
        # plt.savefig(os.path.join(save_root, 'plot_else/BilSTM', dataset, f"{index:03d}_{k}.pdf"))
        index += 1
        plt.close()