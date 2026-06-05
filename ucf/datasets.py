from torch.utils.data import Dataset, DataLoader
import torch
from utils.preprocess import *
import option
import os


class AllDataset(torch.utils.data.Dataset):

    def __init__(self, args, test_mode=False):

        self.test_mode = test_mode
        self.max_sequence_length = args.max_sequence_length
        # 这里的normal_flag是为了区分正常和异常样本，ucf的区分是‘Normal_Videos’
        self.normal_flag = 'Normal_Videos'

        # 判断是否是测试集
        if self.test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.rgb_list

        # 打开rgb_list_file文件，将其内容转换为列表
        self.list = list(open(self.rgb_list_file))
        
        # 保存文件路径列表，用于后续绘制视频图时添加视频名字
        if self.test_mode:
            self.files = [line.strip() for line in self.list]

    def __len__(self):
        # 返回数据集的长度
        return len(self.list)

    def __getitem__(self, index):

        # 如果包含normal_flag，标签为0.0，否则为1.0
        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0

        features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        features_not_pad = features_not_pad.mean(axis=1)
        if self.test_mode:
            return features_not_pad
        else:
            features_fused = process_feat(features_not_pad, self.max_sequence_length, is_random=False)
            return features_fused, label


if __name__ == '__main__':
    args = option.parser.parse_args()
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    test_data = AllDataset(args=args, test_mode=True)
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    i = 0
    for inputs in test_loader:
        inputs = inputs.to(args.device)
        print(inputs.shape)
        i += 1

    print(i)