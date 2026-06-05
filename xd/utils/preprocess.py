import numpy as np


def random_extract(feat, t_max): # 定义 random_extract 函数：从输入特征 feat 中随机抽取连续 t_max 长度的片段。
   r = np.random.randint(len(feat)-t_max) # 在 [0, len(feat)-t_max) 范围内随机生成一个整数 r，作为起始位置。
   return feat[r:r+t_max] # 返回从索引 r 到 r+t_max 的特征子序列。

def uniform_extract(feat, t_max): # 定义 uniform_extract 函数：从输入特征 feat 中均匀抽取 t_max 个采样点。
    # 使用 np.linspace 在 0 到 len(feat)-1 之间均匀生成 t_max 个采样点，并将索引转换为无符号16位整数。
   r = np.linspace(0, len(feat)-1, t_max, dtype=np.uint16)
   return feat[r, :] # 根据生成的索引 r，从 feat 中抽取对应行（假设 feat 为二维数组），返回均匀采样后的特征。

def pad(feat, min_len): # 定义 pad 函数：对特征 feat 进行填充，使得其第一维长度至少为 min_len。
    if np.shape(feat)[0] <= min_len: # 如果 feat 的第一维（例如帧数或序列长度）小于等于 min_len，则需要填充。
        # 使用 np.pad 在第一维末尾填充所需的行数，使总长度达到 min_len；
        # 第二维不做填充，填充值设为常数 0。
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else: # 如果 feat 的长度已经大于 min_len，则无需填充，直接返回原特征。
       return feat

def process_feat(feat, length, is_random=True):
    # 定义 process_feat 函数：对输入特征 feat 进行预处理，
    # 若长度大于指定 length，则根据 is_random 参数随机或均匀抽取；否则，对 feat 进行填充。

    if len(feat) > length: # 如果 feat 的长度超过指定的 length，则抽取一个子序列。
        if is_random: # 如果 is_random 为 True，则使用随机抽取的方式。
            return random_extract(feat, length)
        else:
            # 如果 is_random 为 False，则使用均匀抽取的方式。
            return uniform_extract(feat, length)
    else:
        # 如果 feat 的长度不足指定的 length，则调用 pad 函数对 feat 进行填充。
        return pad(feat, length)

def process_feat_1(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat