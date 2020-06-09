#!/bin/env python
import torch
import torch.nn as nn


class filter_pruner:
    def __init__(self, layer):
        self.layer = layer

    def cal_mask_l1(self, ratio):
        layer = self.layer
        filters = self.layer.ori_weight.shape[0]
        w_abs = self.layer.ori_weight.abs()
        w_sum = w_abs.view(filters, -1).sum(1)
        count = filters - int( filters * ratio )
        if count > 0:
            threshold = torch.topk(w_sum.view(-1), count, largest=False)[0].max()
        else:
            # don't prune this layer
            threshold = torch.min(w_sum) - 1.0
            # print(threshold)
            
        mask_weight = torch.gt(w_sum, threshold)[:, None, None, None].expand_as(layer.weight).type_as(layer.w_mask).detach()
        need_b_mask = True if hasattr(layer, 'bias') and layer.bias is not None else False
        mask_bias = torch.gt(w_sum, threshold).type_as(layer.b_mask).detach() if need_b_mask else None
        return mask_weight, mask_bias

    def cal_mask_l2(self, ratio):
        layer = self.layer
        filters = self.layer.ori_weight.shape[0]
        w = self.layer.ori_weight.view(filters, -1)
        w_l2 = torch.sqrt((w ** 2).sum(dim=1)) 
        count = filters - int( filters * ratio )
        if count > 0: 
            threshold = torch.topk(w_l2.view(-1), count, largest=False)[0].max()
        else:
            threshold = torch.min(w_l2) - 1.0
        mask_weight = torch.gt(w_l2, threshold)[:, None, None, None].expand_as(layer.weight).type_as(layer.w_mask).detach()
        need_b_mask = True if hasattr(layer, 'bias') and layer.bias is not None else False
        mask_bias = torch.gt(w_l2, threshold).type_as(layer.b_mask).detach() if need_b_mask else None
        return mask_weight, mask_bias