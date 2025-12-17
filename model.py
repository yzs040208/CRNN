import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    """双向LSTM模块（PDF第15-16页）"""

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        # 双向LSTM
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # 全连接层（将双向LSTM的输出维度映射到类别数）
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # input shape: [T, N, nIn] (T:时间步, N:批次大小)
        recurrent, _ = self.rnn(input)  # [T, N, 2*nHidden]
        T, N, H = recurrent.size()

        # 重塑为全连接层输入格式
        t_rec = recurrent.view(T * N, H)  # [T*N, 2*nHidden]
        output = self.embedding(t_rec)  # [T*N, nOut]
        output = output.view(T, N, -1)  # [T, N, nOut]

        return output


class CRNN(nn.Module):
    """CRNN模型（CNN + 双向LSTM + CTC）（PDF第13-15页）"""

    def __init__(self, imgH, nc, nclass, nh, nrnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "imgH必须是16的倍数"

        # CNN参数（与PDF一致）
        ks = [3, 3, 3, 3, 3, 3, 2]  # 卷积核大小
        ps = [1, 1, 1, 1, 1, 1, 0]  # padding大小
        ss = [1, 1, 1, 1, 1, 1, 1]  # 步长
        nm = [64, 128, 256, 256, 512, 512, 512]  # 输出通道数

        # 构建CNN序列
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            """卷积+激活函数封装（支持批标准化）"""
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]

            # 添加卷积层
            cnn.add_module(f'conv{i}',
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

            # 批标准化（加速收敛）
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))

            # 激活函数
            if leakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        # 卷积层配置（与PDF一致，包含池化层）
        convRelu(0)  # 64x32xW
        cnn.add_module('pool0', nn.MaxPool2d(2, 2))  # 64x16x(W/2)

        convRelu(1)  # 128x16x(W/2)
        cnn.add_module('pool1', nn.MaxPool2d(2, 2))  # 128x8x(W/4)

        convRelu(2, batchNormalization=True)  # 256x8x(W/4)
        convRelu(3)  # 256x8x(W/4)
        cnn.add_module('pool2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x(W/4 +1)

        convRelu(4, batchNormalization=True)  # 512x4x(W/4 +1)
        convRelu(5)  # 512x4x(W/4 +1)
        cnn.add_module('pool3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x(W/4 +2)

        convRelu(6, batchNormalization=True)  # 512x1x(W/4 +2)（高度变为1，满足RNN输入）

        self.cnn = cnn

        # 构建RNN序列（2层双向LSTM）
        self.rnn = nn.Sequential(
            BidirectionalLSTM(nm[-1], nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # 1. CNN特征提取
        # input shape: [N, 1, 32, W] (N:批次大小, W:图片宽度)
        conv = self.cnn(input)  # [N, 512, 1, W'] (W':经过卷积池化后的宽度)
        b, c, h, w = conv.size()
        assert h == 1, "CNN输出高度必须为1"

        # 2. 维度转换（适配RNN输入格式）
        # RNN输入格式：[T, N, C] (T:时间步=W', N:批次, C:特征数=512)
        conv = conv.squeeze(2)  # [N, 512, W'] 移除高度维度
        conv = conv.permute(2, 0, 1)  # [W', N, 512]

        # 3. RNN序列预测
        output = self.rnn(conv)  # [T, N, nclass]

        # 4. 对数softmax（CTC损失要求）
        output = F.log_softmax(output, dim=2)

        return output