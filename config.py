import os
import torch

# 基础配置
class Config:
    # 设备配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型参数（关键：NCLASS改为生成的字符数+1，默认是72+1=73）
    IMG_H = 32  # 固定32，与数据集裁剪尺寸匹配
    NC = 1  # 灰度图，无需修改
    NCLASS = 73  # 替换为alphabet.txt字符数+1（运行生成代码后会提示具体值）
    NH = 256  # 无需修改
    NRNN = 2  # 无需修改

    # 训练参数
    TRAIN = {
        "END_EPOCH": 100,  # 可根据数据量调整（1000条样本建议50轮）
        "BATCH_SIZE": 32,  # 无需修改
        "LR": 0.001,  # 无需修改
        "MOMENTUM": 0.9,  # 无需修改
        "WEIGHT_DECAY": 1e-5  # 无需修改
    }

    # 数据路径（关键：匹配生成的数据集路径）
    DATA = {
        "TRAIN_ROOT": "./data/train",  # 训练集根目录（包含name/id子目录）
        "VAL_ROOT": "./data/val",  # 验证集根目录
        "TRAIN_LABEL": "./data/train_labels.txt",  # 训练集标签
        "VAL_LABEL": "./data/val_labels.txt",  # 验证集标签
        "ALPHABET": "./alphabet.txt"  # 自动生成的字符集
    }

    # 输出路径
    OUTPUT = {
        "CHECKPOINT_DIR": "./checkpoints",  # 模型保存路径
        "LOG_DIR": "./logs"  # TensorBoard日志路径
    }


# 创建输出目录
def init_dirs(config):
    os.makedirs(config.OUTPUT["CHECKPOINT_DIR"], exist_ok=True)
    os.makedirs(config.OUTPUT["LOG_DIR"], exist_ok=True)


# 实例化配置
cfg = Config()
init_dirs(cfg)