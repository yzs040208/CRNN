import os


def process_labels_file(input_file, output_file=None):
    """
    处理标签文件，去除路径中的train/或val/前缀
    :param input_file: 输入文件路径
    :param output_file: 输出文件路径，默认覆盖原文件
    """
    if output_file is None:
        output_file = input_file  # 默认覆盖原文件

    processed_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 去除train/或val/前缀
            if line.startswith('train/'):
                processed_line = line.replace('train/', '', 1)
            elif line.startswith('val/'):
                processed_line = line.replace('val/', '', 1)
            else:
                processed_line = line  # 保持原格式（防止异常数据）
            processed_lines.append(processed_line)

    # 写入处理后的内容
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))

    print(f"文件处理完成：{input_file} -> {output_file}")


# 批量处理两个标签文件
if __name__ == "__main__":
    # 请根据实际文件路径修改（相对路径或绝对路径均可）
    file_paths = [
        "data/val_labels.txt",
        "data/train_labels.txt"
    ]

    # 处理所有文件（默认覆盖原文件）
    for file_path in file_paths:
        if os.path.exists(file_path):
            process_labels_file(file_path)
        else:
            print(f"警告：文件不存在 - {file_path}")