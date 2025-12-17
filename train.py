import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
from model import CRNN
from dataset import OCRDataset
from utils import AverageMeter, strLabelConverter, get_batch_label
from config import cfg


def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict):
    """训练函数（PDF第18页）"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (inp, labels) in enumerate(train_loader):
        # 记录数据加载时间
        data_time.update(time.time() - end)

        # 数据迁移到设备
        inp = inp.to(device)

        # 模型推理
        preds = model(inp)  # [T, N, nclass]

        # 标签编码
        text, length = converter.encode(labels)
        batch_size = inp.size(0)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)  # 每个样本的时间步长度

        # 计算CTC损失（注意：CTC_loss要求输入为log_softmax输出）
        loss = criterion(preds, text, preds_size, length)

        # 梯度清零+反向传播+参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        losses.update(loss.item(), batch_size)

        # 记录批次时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 打印日志（每100个batch）
        if (i + 1) % 100 == 0:
            print(f'Epoch: [{epoch + 1}/{config.TRAIN.END_EPOCH}] '
                  f'Batch: [{i + 1}/{len(train_loader)}] '
                  f'DataTime: {data_time.val:.3f}({data_time.avg:.3f}) '
                  f'BatchTime: {batch_time.val:.3f}({batch_time.avg:.3f}) '
                  f'Loss: {losses.val:.4f}({losses.avg:.4f})')

            # TensorBoard记录
            if writer_dict is not None:
                writer = writer_dict['writer']
                global_step = writer_dict['global_step']
                writer.add_scalar('train/loss', losses.val, global_step)
                writer_dict['global_step'] += 1


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict):
    """验证函数（计算准确率）"""
    losses = AverageMeter()
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (inp, labels) in enumerate(val_loader):
            inp = inp.to(device)

            # 模型推理
            preds = model(inp)
            text, length = converter.encode(labels)
            batch_size = inp.size(0)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)

            # 计算损失
            loss = criterion(preds, text, preds_size, length)
            losses.update(loss.item(), batch_size)

            # 预测结果解码
            _, preds = preds.max(dim=2)  # [T, N]
            preds = preds.transpose(1, 0).contiguous().view(-1)  # [N*T]
            pred_texts = converter.decode(preds, length)

            # 统计准确率
            for pred_text, true_text in zip(pred_texts, labels):
                if pred_text == true_text:
                    total_correct += 1
            total_samples += batch_size

    # 计算平均准确率
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f'Epoch: [{epoch + 1}/{config.TRAIN.END_EPOCH}] '
          f'Val Loss: {losses.avg:.4f} '
          f'Val Acc: {acc:.4f}')

    # TensorBoard记录
    if writer_dict is not None:
        writer = writer_dict['writer']
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/acc', acc, epoch)

    return acc


def main():
    # 1. 加载字符集
    with open(cfg.DATA["ALPHABET"], 'r', encoding='utf-8') as f:
        alphabet = f.read().strip()
    converter = strLabelConverter(alphabet)

    # 2. 构建数据集和数据加载器
    train_dataset = OCRDataset(
        img_root=cfg.DATA["TRAIN_ROOT"],
        label_path=cfg.DATA["TRAIN_LABEL"],
        imgH=cfg.IMG_H
    )
    val_dataset = OCRDataset(
        img_root=cfg.DATA["VAL_ROOT"],
        label_path=cfg.DATA["VAL_LABEL"],
        imgH=cfg.IMG_H
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN["BATCH_SIZE"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 3. 构建模型、损失函数、优化器
    model = CRNN(
        imgH=cfg.IMG_H,
        nc=cfg.NC,
        nclass=cfg.NCLASS,
        nh=cfg.NH,
        nrnn=cfg.NRNN
    ).to(cfg.DEVICE)

    # CTC损失函数（blank=0，与converter一致）
    criterion = torch.nn.CTCLoss(blank=0).to(cfg.DEVICE)

    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN["LR"],
        weight_decay=cfg.TRAIN["WEIGHT_DECAY"]
    )

    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 4. TensorBoard日志
    writer = SummaryWriter(cfg.OUTPUT["LOG_DIR"])
    writer_dict = {
        'writer': writer,
        'global_step': 0
    }

    # 5. 训练主循环（PDF第17页）
    best_acc = 0.0
    last_epoch = 0

    for epoch in range(last_epoch, cfg.TRAIN["END_EPOCH"]):
        # 训练一个epoch
        train(cfg, train_loader, train_dataset, converter, model, criterion, optimizer, cfg.DEVICE, epoch, writer_dict)

        # 学习率更新
        lr_scheduler.step()

        # 验证
        acc = validate(cfg, val_loader, val_dataset, converter, model, criterion, cfg.DEVICE, epoch, writer_dict)

        # 保存最优模型
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        checkpoint = {
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_acc': best_acc
        }

        # 保存checkpoint
        checkpoint_path = os.path.join(
            cfg.OUTPUT["CHECKPOINT_DIR"],
            f'checkpoint_epoch_{epoch + 1}_acc_{acc:.4f}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # 保存最优模型
        if is_best:
            best_checkpoint_path = os.path.join(cfg.OUTPUT["CHECKPOINT_DIR"], 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)

        print(f'Best Acc: {best_acc:.4f}')

    # 6. 清理资源
    writer.close()


if __name__ == "__main__":
    main()