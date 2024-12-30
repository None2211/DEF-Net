import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 确保输入通过sigmoid激活转换为概率
        assert inputs.size() == targets.size(), "The dimensions of inputs and targets must match."

        inputs = torch.sigmoid(inputs)

        # 展平除了批次维度之外的所有维度
        inputs_flat = inputs.view(inputs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # 计算交集
        intersection = (inputs_flat * targets_flat).sum(1)

        # 计算每个样本的dice系数
        dice = (2. * intersection + self.smooth) / (inputs_flat.sum(1) + targets_flat.sum(1) + self.smooth)

        # 返回1 - dice系数的平均值作为损失
        dice_loss = 1 - dice
        return dice_loss.mean()


def average_dice_coefficient(inputs, targets, smooth=1e-5):
    """
    Calculate the average Dice Coefficient for a batch of predictions and targets.

    Parameters:
    inputs (torch.Tensor): The predicted probabilities from the model (before sigmoid activation).
    targets (torch.Tensor): The ground truth labels.
    smooth (float): A small value to avoid division by zero and stabilize the division.

    Returns:
    float: The average Dice Coefficient for the batch.
    """
    assert inputs.size() == targets.size(), "Inputs and targets must have the same shape."

      # Ensure the inputs are in probability space.

    # Flatten the inputs and targets to compute the intersection and union across all dimensions except the batch dimension.
    inputs_flat = inputs.view(inputs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (inputs_flat * targets_flat).sum(dim=1)
    union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)

    return dice.mean().item()
if __name__ == '__main__':
    # 创建一个形状为 [3, 1, 128, 128]，其中所有值都是1的张量
    input = torch.zeros(3, 1, 4, 4)

    targets = torch.full((3, 1, 4, 4), 0.5)
    dice_loss_fn = BatchDiceLoss()

    # 计算 Dice Loss
    loss = dice_loss_fn(input, targets)

    print("Dice Loss:", loss.item())


