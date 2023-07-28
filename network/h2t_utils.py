import os
import math

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms

class feature_extraction(nn.Module):
    def __init__(self, model, target_size):
        super(feature_extraction,self).__init__()
        
        self.target_size = target_size
        
        self.model = model
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.name_list = []
        self.name_list.append('relu')
        for i, j in enumerate([3, 4, 6, 3]):
            for k in range(j):
                for l in range(3):
                    self.name_list.append(f'layer{i+1}.{k}.relu{l+1}')
        self.name_list.append('avgpool')
        
        self.embedding_size = {}
        self.feature_shape = {}
        self.activation = []
        self.forward_hook()
    
    def getActivation(self):
        # the hook signature
        def hook(model, input, output):
            self.activation.append(output)
        return hook

    def forward_hook(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.modules.activation.ReLU):
                layer.register_forward_hook(self.getActivation())
            elif isinstance(layer, nn.modules.pooling.AdaptiveAvgPool2d):
                layer.register_forward_hook(self.getActivation())

    def flatten_and_concat(self):
        concat_vector_list = []
        assert len(self.activation) == len(self.name_list)
        for i in range(len(self.name_list)):
            name = self.name_list[i]
            feature_map = self.activation[i]
            if 'relu' in name:
                avg_out_siz = int(math.sqrt(self.target_size / feature_map.shape[1]))
                feature = F.adaptive_avg_pool2d(feature_map, output_size=(avg_out_siz, avg_out_siz))
                self.feature_shape[name] = feature.shape
                flatten_vector = torch.flatten(feature, 1)
                self.embedding_size[name] = flatten_vector.shape[-1]
                concat_vector_list.append(flatten_vector)
            elif 'pool' in name:
                feature = feature_map
                self.feature_shape[name] = feature.shape
                flatten_vector = torch.flatten(feature, 1)
                self.embedding_size[name] = flatten_vector.shape[-1]
                concat_vector_list.append(flatten_vector)
        concat_vector = torch.concat(concat_vector_list, axis = 1)
        return concat_vector
    
    def forward(self, x):
        self.activation = []
        out = self.model(x)
        # concat_vector_list = []
        concat_vector = self.flatten_and_concat()
        return concat_vector

def embed_dataset(feature_extraction_model, data_loader, device):
    feature_list = []
    label_list = []
    for input, label, _ in data_loader:
        input = input.to(device)
        batch_feature = feature_extraction_model(input)
        batch_feature = batch_feature.to('cpu')
        feature_list.append(batch_feature)
        label_list.append(label)
    embed_feature = torch.concat(feature_list, axis = 0)
    labels = torch.concat(label_list, axis = 0)
    return embed_feature, labels

if __name__ == '__main__':
    pass