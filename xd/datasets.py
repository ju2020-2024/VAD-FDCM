from torch.utils.data import Dataset, DataLoader
import torch
from utils.preprocess import *
import option
import os


class AllDataset(torch.utils.data.Dataset):

    def __init__(self, args, test_mode=False):

        self.test_mode = test_mode
        self.max_sequence_length = args.max_sequence_length
        self.normal_flag = '_label_A'
        self.Bool = args.database
        self.files = []

        if self.test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
            with open(self.audio_list_file, 'r') as f:
                self.audio_file_paths = [line.strip() for line in f.readlines()]
            self.files = self.audio_file_paths  # 这里才是真正的路径列表
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list

        if self.Bool == 'MIX':
            self.list = list(open(self.rgb_list_file))
            self.audio_list = list(open(self.audio_list_file))
        else:
            self.list = list(open(self.rgb_list_file))

    def __len__(self):

        return len(self.list)

    def __getitem__(self, index):

        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0

        if self.Bool == 'MIX':

            features_audio = np.array(np.load(self.audio_list[index // 5].strip('\n')), dtype=np.float32)
            try:
                features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            except Exception as e:
                print("Error at index:", index)
                print("FileAudio:", self.audio_list[index // 5])
                print("FileRGB:", self.list[index])
                raise e
            if features_not_pad.shape[0] == features_audio.shape[0]:
                features_fused = np.concatenate((features_not_pad, features_audio), axis=1)
            else:
                features_fused = np.concatenate((features_not_pad[:-1], features_audio), axis=1)
        else:
            features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features_fused = features_not_pad

        if self.test_mode:
            return features_fused
        else:
            features_fused = process_feat(features_fused, self.max_sequence_length, is_random=False)
            return features_fused, label


class NormalDataset(torch.utils.data.Dataset):
    def __init__(self, args_1, test_mode=False):
        self.test_mode = test_mode
        if self.test_mode:
            self.rgb_list_file = args_1.test_rgb_list
        else:
            self.rgb_list_file = args_1.rgb_list
        self.list = list(open(self.rgb_list_file))
        if self.test_mode:
            self.list = self.list[-1500:]
        else:
            self.list = self.list[-10245:]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        return features_not_pad


class AbnormalDataset(torch.utils.data.Dataset):
    def __init__(self, args_2, test_mode=False):
        self.test_mode = test_mode
        if self.test_mode:
            self.rgb_list_file = args_2.test_rgb_list
        else:
            self.rgb_list_file = args_2.rgb_list
        self.list = list(open(self.rgb_list_file))
        if self.test_mode:
            self.list = self.list[:2500]
        else:
            self.list = self.list[:9525]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)  # 加载数据
        return features_not_pad


if __name__ == '__main__':
    # path = "/home/stu2023/jj/Data/vggish-features/vggish-features/train/v=VHMi-j7W2gM__#1_label_B1-0-0__vggish.npy"
    #
    # x = np.load(path, allow_pickle=True)
    # print(type(x))
    # print(x.shape)
    root = "/home/stu2023/jj/Data/vggish-features/vggish-features/test"
    # root = "/home/stu2023/jj/Data/I3D/RGB/RGB"

    bad_files = []

    for dirpath, dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".npy"):
                path = os.path.join(dirpath, f)
                try:
                    np.load(path)
                except Exception as e:
                    print("BAD:", path, "| ERROR:", e)
                    bad_files.append(path)

    print("\nTotal bad files:", len(bad_files))
