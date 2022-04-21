import torch
import torch.nn as nn


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, smooth=0, reduction='mean', do_sigmoid=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.do_sigmoid = do_sigmoid
        self.labels = ["Background", "Liver", "Tumour"]
        self.device = "cpu"

    def forward(self, pred, target, train=True):
        """This definition generalize to real valued pred and target vector.
        Parameters:
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        Returns:
        generalized_dice_loss:1-generalized_dice
        """
        if self.do_sigmoid:
            pred = torch.sigmoid(pred)

        assert pred.shape[0] == target.shape[0], "predict & target batch size do not match"
        final_dice = 1 - generalized_dice(pred, target, smooth=self.smooth, reduction=self.reduction)
        if train:
            ce = 0
            CE_L = torch.nn.BCELoss()
            for i in range(target.size(1)):
                ce = ce + CE_L(torch.sigmoid(pred[:, i, ...]), target[:, i, ...])
            final_dice = (0.7 * final_dice + 0.3 * ce) / (target.size(1))
        return final_dice


    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = GeneralizedDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


def generalized_dice(pred, target, smooth=0, reduction='mean'):
    """
        Computes dice coefficient given  a multi channel input and a multi channel target
        Assumes the input is a normalized probability, a result of a Softmax function.
        Assumes that the channel 0 in the input is the background
        Args:
             pred (torch.Tensor): NxCxSpatial pred tensor
             target (torch.Tensor): NxCxSpatial target tensor
        """
    pflat = pred.view(pred.shape[0], pred.shape[1], -1)
    tflat = target.view(target.shape[0], target.shape[1], -1)
    tflat_sum = tflat.sum(dim=2)
    w_tflat = 1/(torch.mul(tflat_sum,tflat_sum)).clamp(min=1e-6)
    w_tflat.requires_grad = False
    num = w_tflat*torch.sum(torch.mul(pflat, tflat), dim=2) + smooth
    den = w_tflat*(torch.sum(pflat+tflat, dim=2)) + smooth + 1e-6
    # We do not want to take into account the background
    # num = num[:, 1:]
    # den = den[:, 1:]
    dices = 2 * num.sum(dim=1) / den.sum(dim=1)

    if reduction == 'mean':
        return dices.mean()
    elif reduction == 'sum':
        return dices.sum()
    elif reduction == 'none':
        return dices
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))




class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["Background", "Liver", "Tumour"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        ce = 0
        CE_L = torch.nn.BCELoss()
        for i in range(1, target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
            ce = ce + CE_L(torch.sigmoid(inputs[:, i, ...]), target[:, i, ...])
        final_dice = (0.7 * dice + 0.3 * ce) / (target.size(1) - 1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class EDiceLoss_Val(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss_Val, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["Background", "Liver", "Tumour"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss_Val.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(1, target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
        final_dice = dice / (target.size(1)-1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


