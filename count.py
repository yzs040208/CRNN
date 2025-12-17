# 创建一个 generate_alphabet.py 文件，与 train.py 同级
import os
from config import cfg


def generate_alphabet():
    # 读取所有标签文件
    label_files = [
        cfg.DATA["TRAIN_LABEL"],
        cfg.DATA["VAL_LABEL"]
    ]

    all_chars = set()  # 用集合去重

    for label_file in label_files:
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"标签文件不存在: {label_file}")

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 标签格式：图片名 标签内容（例如：img1.jpg 123abc）
                # 提取标签内容（忽略图片名）
                parts = line.split()
                if len(parts) < 2:
                    continue
                label = ' '.join(parts[1:])  # 标签内容
                for char in label:
                    all_chars.add(char)

    # 将字符写入 alphabet.txt
    with open(cfg.DATA["ALPHABET"], 'w', encoding='utf-8') as f:
        f.write(''.join(sorted(all_chars)))  # 排序后写入，方便查看

    print(f"已生成 alphabet.txt，包含 {len(all_chars)} 个字符")
    print(f"请将 config.py 中的 NCLASS 修改为：{len(all_chars) + 1}")  # +1 是因为包含 blank 字符


if __name__ == "__main__":
    generate_alphabet()