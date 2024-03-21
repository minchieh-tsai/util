from torch import einsum
from scipy.ndimage.morphology import distance_transform_edt as edt

import numpy as np
import scipy.misc
import argparse
import os
import math
import torch
import torch.nn as nn


#%%metric

# weight dice(有考慮影像品質不好的部分)
def weight_dice_wayne(preds_var: np.array, masks_var: np.array, A4C_index = 3, bad_index = 1, margin = 10) -> float(): 
	mdice = np.array([])
	for i in range(masks_var.shape[0]):
		pred 	= (preds_var[i, :, :] == A4C_index)
		label 	= (masks_var[i, :, :] == A4C_index)
		fg_mask = (masks_var[i, :, :] == bad_index)
		
		bad = fg_mask.any()
		if bad:
			bg_mask = ~fg_mask
			bg_dist = edt(bg_mask)
			fg_dist = -edt(fg_mask)
			fg_dist[fg_dist < -margin] = -margin
			bg_dist[bg_dist > margin] = margin
			distance_map = fg_dist + bg_dist
			distance_map = (distance_map - distance_map.min())
			distance_map = distance_map / np.max(distance_map)
			weight_dice = dice_coeff(label * distance_map, pred)
			mdice = np.append(mdice, weight_dice)

		else:
			dice = dice_coeff(label, pred)
			mdice = np.append(mdice, dice)

		
# 		# for test
# 		plt.figure()
# 		plt.subplot(121)
# 		plt.imshow(label, cmap = 'gray')
# 		plt.title(f'index = {i}')
# 		plt.subplot(122)
# 		plt.imshow(pred, cmap = 'gray')
# 		if bad:
#  			plt.title(f'weight_dice = {weight_dice:.3f}, bad')
# 		else:
#  			plt.title(f'dice = {dice:.3f}, good')
 			
# 		if bad:
#  			plt.figure()
#  			plt.subplot(121)
#  			plt.imshow(bg_mask, cmap = 'gray')
#  			plt.title(f'index = {i}')
#  			plt.subplot(122)
#  			plt.imshow(distance_map, cmap = 'gray')
			
		
	return mdice.mean()

# weight dice使用的(有考慮影像品質不好的部分)
def dice_coeff(pred, target):
	smooth = 0
# 	num = pred.size(0)
# 	pred = pred[:,0,:,:]
# 	target = target[:,0,:,:]
# 	m1 = pred.view(num, -1).float()  # Flatten
# 	m2 = target.view(num, -1).float()  # Flatten
	intersection = (pred * target).sum().astype('float64')
# 	print(2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
	return  (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# 只計算左心室(第三類)的dice
def dice_score(pred, labels,n_classes,smooth=1e-5 ,show=False):
    # 3是左心室那類
    i = 3
    mdice = np.array([])
    tp_pred = pred == i
    tp_label = labels == i
    tp_fp = np.sum(tp_pred)
    tp_fn = np.sum(tp_label)
    tp = np.sum((tp_pred) * (tp_label))

    dice = (2 * tp + smooth) / (tp_fp + tp_fn + smooth)
    mdice = np.append(mdice, dice)
    if show:
        print('class #%d : %1.5f'%(i, dice))
    
    if (mdice == np.array([])):
        mean = 0
    else:
        mean = dice.mean()
        
    if show:
        print('\nmean_dice: %f\n' % mean)

    return mean

# 計算4個類別的mean dice
def mean_dice_score(pred, labels,n_classes, show=False):
    
    mean_iou = np.array([])
    for i in range(n_classes):
        tp_pred = pred == i
        tp_label = labels == i
        tp_fp = np.sum(tp_pred)
        tp_fn = np.sum(tp_label)
        tp = np.sum((tp_pred) * (tp_label))
        if (tp_fp + tp_fn - tp)!=0:
            iou = 2 * tp / (tp_fp + tp_fn)
            mean_iou = np.append(mean_iou, iou)
        if show:
            print('class #%d : %1.5f'%(i, iou))
    
    if (mean_iou == np.array([])):
        mean = 0
    else:
        mean = mean_iou.mean()
        
    if show:
        print('\nmean_dice: %f\n' % mean)

    return mean

# 計算4個類別的miou
def mean_iou_score(pred, labels, n_classes, show=False):

    # 有幾類
    mean_iou = np.array([])
    for i in range(n_classes):
        tp_pred = pred == i
        tp_label = labels == i
        tp_fp = np.sum(tp_pred)
        tp_fn = np.sum(tp_label)
        tp = np.sum((tp_pred) * (tp_label))
        if (tp_fp + tp_fn - tp)!=0:
            iou = tp / (tp_fp + tp_fn - tp)
            mean_iou = np.append(mean_iou, iou)
        if show:
            print('class #%d : %1.5f'%(i, iou))
    
    if (mean_iou == np.array([])):
        mean = 0
    else:
        mean = mean_iou.mean()
        
    if show:
        print('\nmean_iou: %f\n' % mean)

    return mean


#%%losses

# CE計算loss時使用
def CrossEntropy2d(outputs, labels, criterion, n_classes):
    outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()
    outputs = outputs.view(-1, n_classes)
    labels = labels.transpose(1, 2).transpose(2, 3).contiguous()
    labels = labels.view(-1)
    # print(outputs.size())
    # print(labels.size())
    loss = criterion(outputs, labels.long())
    return loss



class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        net_output = torch.nn.functional.softmax(net_output, dim=1, _stacklevel=3, dtype=None)
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
            
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * einsum("bcxy, bcxy->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", y_onehot))
        
        # intersection: torch.Tensor = einsum("bcxy, bcxy->bc", net_output, y_onehot)
        # union: torch.Tensor = (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", y_onehot))
        divided: torch.Tensor = - 2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor , device , debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()


        pred_error = (pred - target) ** 2
        distance = (pred_dt ** self.alpha + target_dt ** self.alpha).to(device)

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss

#%%util

# 計算每個step的平均
class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v