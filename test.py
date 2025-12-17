import torch
import cv2
import os
from model import CRNN
from utils import strLabelConverter
from config import cfg


def load_model(model_path):
    """加载训练好的模型"""
    model = CRNN(
        imgH=cfg.IMG_H,
        nc=cfg.NC,
        nclass=cfg.NCLASS,
        nh=cfg.NH,
        nrnn=cfg.NRNN
    ).to(cfg.DEVICE)

    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def preprocess_image(img_path):
    """图片预处理（与数据集一致）"""
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")

    # 灰度化->二值化->高斯滤波->resize
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_blur = cv2.GaussianBlur(img_bin, (3, 3), 0)

    # resize
    h, w = img_blur.shape
    scale = cfg.IMG_H / h
    new_w = int(w * scale)
    img_resized = cv2.resize(img_blur, (new_w, cfg.IMG_H), interpolation=cv2.INTER_LINEAR)

    # 归一化+tensor转换
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    return img_tensor.to(cfg.DEVICE)


def predict(img_path, model, converter):
    """预测单张图片"""
    img_tensor = preprocess_image(img_path)

    with torch.no_grad():
        preds = model(img_tensor)  # [T, 1, nclass]
        _, preds = preds.max(dim=2)  # [T, 1]
        preds = preds.transpose(1, 0).contiguous().view(-1)  # [T]

        # 解码（假设标签长度为预测结果的有效长度，这里简化处理）
        # 实际应用中可根据数据集统计设置合理长度
        pred_text = converter.decode(preds, torch.IntTensor([preds.size(0)]))

    return pred_text[0]


def main():
    # 1. 加载字符集和转换器
    with open(cfg.DATA["ALPHABET"], 'r', encoding='utf-8') as f:
        alphabet = f.read().strip()
    converter = strLabelConverter(alphabet)

    # 2. 加载模型
    model_path = os.path.join(cfg.OUTPUT["CHECKPOINT_DIR"], 'best_model.pth')
    model = load_model(model_path)
    print("模型加载成功！")

    # 3. 测试图片路径
    test_img_path = "./test_img.jpg"  # 替换为你的测试图片路径

    # 4. 预测
    pred_text = predict(test_img_path, model, converter)
    print(f"预测结果: {pred_text}")


if __name__ == "__main__":
    main()