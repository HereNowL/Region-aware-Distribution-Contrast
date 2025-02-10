import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=0.95):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss
    
class ComputeLoss(nn.Module):
    def __init__(self):
        super(ComputeLoss, self).__init__()
        self.comp_edge_loss = BalancedCrossEntropyLoss()
    
    def forward(self):
        pass

    def compute_supervision(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3=None, x_output3=None):
        # Compute supervised task-specific loss for all tasks when all task labels are available

        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()
        # semantic loss: depth-wise cross entropy
        x_pred1 = F.log_softmax(x_pred1, dim=1) 
        loss1 = F.nll_loss(x_pred1, x_output1, ignore_index=-1)

        # depth loss: l1 norm
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask) / torch.nonzero(binary_mask).size(0)
        if x_pred3 == None:
            return [loss1, loss2]

        # normal loss: dot product
        # binary mark to mask out undefined pixel space
        binary_mask_3 = (torch.sum(x_output3, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).cuda()

        x_pred3 = x_pred3 / torch.norm(x_pred3, p=2, dim=1, keepdim=True)
        loss3 = 1 - torch.sum((x_pred3 * x_output3) * binary_mask_3) / torch.nonzero(binary_mask_3).size(0)

        return [loss1, loss2, loss3]
    
    def compute_distill_loss(self, s_pred_s, t_pred_s, s_pred_d, t_pred_d, s_pred_n, t_pred_n):
        # semantic distill
        loss1 = self.comp_semantic_distill_loss(s_pred_s, t_pred_s.detach())

        loss2 = self.comp_depth_distill_loss(s_pred_d, t_pred_d.detach())

        loss3 = self.comp_normal_distill_loss(s_pred_n, t_pred_n.detach())

        return [loss1, loss2, loss3]
    
    def comp_semantic_distill_loss(self, y_s, y_t, T=5):
        p_s = F.log_softmax(y_s / T, dim=1)
        p_t = F.softmax(y_t / T, dim=1)
        # p_s = y_s
        # p_t = y_t
        # p_s = p_s / (p_s.pow(2).sum(1) + 1e-6).sqrt().view(p_s.size(0), 1, p_s.size(2), p_s.size(3))
        # p_t = p_t / (p_t.pow(2).sum(1) + 1e-6).sqrt().view(p_t.size(0), 1, p_t.size(2), p_t.size(3))
        # p_s = F.softmax(p_s / T, dim=1)
        # p_t = F.softmax(p_t / T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='mean') * (T**4) / y_s.shape[0]
        # loss = F.cross_entropy(p_s, p_t.detach(), size_average=False) / y_s.shape[0]
        # loss = (p_s - p_t).pow(2).sum(1).mean()
        return loss 
    
    def comp_depth_distill_loss(self, y_s, y_t, T=2):
        # p_s = F.log_softmax(y_s/T, dim=1)
        # p_t = F.softmax(y_t/T, dim=1)
        p_s = y_s
        p_t = y_t
        p_s = p_s / (p_s.pow(2).sum(1) + 1e-6).sqrt().view(p_s.size(0), 1, p_s.size(2), p_s.size(3))
        p_t = p_t / (p_t.pow(2).sum(1) + 1e-6).sqrt().view(p_t.size(0), 1, p_t.size(2), p_t.size(3))
        # loss = F.kl_div(p_s, p_t, size_average=False) / y_s.shape[0]
        # loss = F.cross_entropy(p_s, p_t, size_average=False) / y_s.shape[0]
        loss = (p_s - p_t).pow(2).sum(1).mean()
        # loss = F.mse_loss(p_s, p_t, size_average=False) / y_s.shape[0]
        # loss = F.cross_entropy(p_s, p_t, size_average=False) / y_s.shape[0]
        return loss 
    
    def comp_normal_distill_loss(self, y_s, y_t, T=5):
        # p_s = y_s / torch.norm(y_s, p=2, dim=1, keepdim=True)
        # p_t = y_t / torch.norm(y_t, p=2, dim=1, keepdim=True)
        # p_s = F.log_softmax(p_s/T, dim=1)
        # p_t = F.softmax(p_t/T, dim=1)
        p_s = y_s
        p_t = y_t
        p_s = p_s / (p_s.pow(2).sum(1) + 1e-6).sqrt().view(p_s.size(0), 1, p_s.size(2), p_s.size(3))
        p_t = p_t / (p_t.pow(2).sum(1) + 1e-6).sqrt().view(p_t.size(0), 1, p_t.size(2), p_t.size(3))
        # loss = F.kl_div(p_s, p_t, size_average=False) / y_s.shape[0]
        # loss = F.cross_entropy(p_s, p_t, size_average=False) / y_s.shape[0]
        loss = (p_s - p_t).pow(2).sum(1).mean()
        # loss = F.mse_loss(p_s, p_t, size_average=False) / y_s.shape[0]
        # loss = F.cross_entropy(p_s, p_t, size_average=False) / y_s.shape[0]
        return loss 
    

