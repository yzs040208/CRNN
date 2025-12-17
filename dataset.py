import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class OCRDataset(Dataset):
    """OCR数据集（处理图片预处理和标签加载）（PDF第14页）"""

    def __init__(self, img_root, label_path, imgH=32, transform=None):
        self.img_root = img_root
        self.imgH = imgH
        self.transform = transform

        # 加载图片路径和标签
        self.img_paths, self.labels = self._load_labels(label_path)

    def _load_labels(self, label_path):
        """加载标签文件（格式：图片名 标签）"""
        img_paths = []
        labels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                if len(line) < 2:
                    continue
                img_name = line[0]
                label = ' '.join(line[1:])
                img_paths.append(os.path.join(self.img_root, img_name))
                labels.append(label)
        return img_paths, labels

    def _preprocess(self, img_path):
        """图片预处理（PDF第14页）：读取->灰度->二值化->高斯滤波->resize"""
        # 1. 读取图片
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")

        # 2. 转为灰度图
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. 二值化（OTSU自动阈值）
        _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. 高斯滤波（去噪）
        img_blur = cv2.GaussianBlur(img_bin, (3, 3), 0)

        # 5. 保持宽高比resize（高度固定为imgH）
        h, w = img_blur.shape
        scale = self.imgH / h
        new_w = int(w * scale)
        img_resized = cv2.resize(img_blur, (new_w, self.imgH), interpolation=cv2.INTER_LINEAR)

        # 6. 归一化（0-1）并转换为tensor
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float() / 255.0

        return img_tensor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # 图片预处理
        img = self._preprocess(img_path)

        # 应用额外transform（可选）
        if self.transform is not None:
            img = self.transform(img)

        return img, label