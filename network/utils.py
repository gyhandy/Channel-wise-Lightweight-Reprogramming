import torch
import torch.nn as nn

from network.ReGhos_Block import *

def fix_BN(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.BatchNorm2d):
            model._modules[name].eval()
            for param in model._modules[name].parameters():
                param.requires_grad = False
        fix_BN(layer)
    return model

def train_BN(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.BatchNorm2d):
            model._modules[name].train()
            for param in model._modules[name].parameters():
                param.requires_grad = True
        train_BN(layer)
    return model

def add_parameters(params, model, layer_type):
    for name, layer in model.named_children():
        if isinstance(layer, layer_type):
            params += list(layer.parameters())
        params = add_parameters(params, layer, layer_type)
    return params

def add_ghostnet(model, model_type="base"):
    for name, layer in model.named_children():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            if model_type == "base":
                model._modules[name] = ReGhos_Block(layer)
            if model_type == "reduced_v1":
                # reduced ghostnet, no linear transformation for 1*1 convolutional
                if layer.kernel_size != (1,1):
                    model._modules[name] = ReGhos_Block(layer)
            if model_type == "reduced_v2":
                # reduced ghostnet, single scale was used for 1*1 convolutional
                if layer.kernel_size == (1,1):
                    model._modules[name] = ReGhos_Block(layer, 1)
                else:
                    model._modules[name] = ReGhos_Block(layer)
        else:
            add_ghostnet(layer, model_type)
    return model

def add_ghost_a(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            model._modules[name] = Ghost_Block_Combine(module, 3)
        else:
            add_ghost_a(module)
    return model


def add_residule(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            model._modules[name] = Residule_block(module)
        else:
            add_residule(module)
    return model

if __name__ == "__main__":
    pass