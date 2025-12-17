import torch
import numpy as np


class AverageMeter:
    """指标统计工具（平均损失、准确率等）"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class strLabelConverter(object):
    """标签与字符串转换工具（PDF第19-20页）"""

    def __init__(self, alphabet, ignore_case=False):
        self.ignore_case = ignore_case
        if self.ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # 添加blank字符（'-'）

        # 字符到索引的映射（0预留为blank）
        self.dict = {}
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + 1  # 字符索引从1开始

    def encode(self, text):
        """将字符串转换为模型训练用的标签（batch支持）"""
        if isinstance(text, str):
            text = [text]

        length = []
        result = []
        # 判断是否需要解码字节串
        decode_flag = isinstance(text[0], bytes)

        for item in text:
            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            # 每个字符转换为索引
            for char in item:
                result.append(self.dict[char])

        return torch.IntTensor(result), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        """将模型输出标签转换为字符串（处理blank和重复字符）"""
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "标签长度与声明长度不匹配"

            if raw:
                # 原始输出（不处理blank和重复）
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                # 处理blank（0）和连续重复字符
                char_list = []
                for i in range(length):
                    # 跳过blank和连续重复字符
                    if t[i] != 0 and not (i > 0 and t[i] == t[i - 1]):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # Batch模式
            assert t.numel() == length.sum(), "标签总长度与声明总长度不匹配"
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


def get_batch_label(dataset, idx):
    """从数据集中获取批次标签（PDF第18页）"""
    return [dataset[i][1] for i in idx]