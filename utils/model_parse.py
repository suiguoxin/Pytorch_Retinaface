#!/bin/env python
import torch
import torch.nn as nn
import torchvision

#OP_Names = ['Conv1d', 'Conv2d', 'Conv3d', 'Linear', 'BatchNorm2d']
OP_Names = ['Conv1d', 'Conv2d', 'Conv3d']
OP_Types = [ getattr(nn, name) for name in OP_Names]

class model_parser:
    def __init__(self, model):
        self.model = model
        self.target_layer = []
    
    def parse_model(self):
        """Parse the model find the layers that can compress"""
        for name, sub_model in self.model.named_modules():
            for op_type in OP_Types:
                if isinstance(sub_model, op_type):
                    sub_model.name = name
                    self.target_layer.append(sub_model)
                    # print(name, sub_model)
        return self.target_layer

class mask_decorater:
    def __init__(self, model):
        # super(mask_decorater, self).__init__()
        self.model = model
        # assert hasattr(model, 'weight')
        parser = model_parser(self.model)
        self.target_layer = parser.parse_model() 
        self.create_mask()

    def create_mask(self):
        for layer in self.target_layer:
            device = layer.weight.device
            layer.old_forward = layer.forward
            layer.register_buffer('w_mask', torch.ones(layer.weight.shape).to(device))
            layer.register_buffer('ori_weight', layer.weight.data.clone().detach())
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.register_buffer('b_mask', torch.ones(layer.bias.shape).to(device))
                layer.register_buffer('ori_bias', layer.bias.data.clone().detach())


    def update_mask(self, layer, weight_mask, bias_mask=None):
        layer.w_mask.data.copy_(weight_mask.data)
        layer.weight.data.copy_(layer.ori_weight.data)
        layer.weight.data.mul_(layer.w_mask.data)
        
        if bias_mask is not None and hasattr(layer, 'bias') and layer.bias is not None:
            layer.b_mask.data.copy_(bias_mask.data)
            layer.bias.data.copy_(layer.ori_bias.data)
            layer.bias.data.mul_(layer.b_mask.data)


    @property
    def length(self):
        return len(self.target_layer)