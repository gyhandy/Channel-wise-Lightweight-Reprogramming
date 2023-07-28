import torch
import torch.nn as nn
import torchvision.models as models
import timm

from network.utils import *
from network.ReGhos_Block import *

class Res_Baseline(nn.Module):
    def __init__(self, num_classes, finetune=False, scratch=False):
        super(Res_Baseline,self).__init__()

        self.num_classes = num_classes
        
        if scratch:
            self.model = models.resnet50()
        else:
            # self.model = models.resnet50(pretrained=True)
            self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

        for param in self.model.parameters():
            param.requires_grad = finetune
        if finetune:
            self.model.train()
        else:
            self.model.eval()
        self.model.fc = torch.nn.Linear(2048, num_classes)
        if finetune:
            self.reg_params = list(self.model.parameters())
        else:
            self.reg_params = list(self.model.fc.parameters())
        self.noreg_params = []

    def forward(self, x):
        x = self.model(x)
        return x

class Xception_Baseline(nn.Module):
    def __init__(self, num_classes, finetune=False, scratch=False):
        super(Xception_Baseline,self).__init__()

        self.num_classes = num_classes
        
        if scratch:
            self.model = timm.create_model('xception')
        else:
            self.model = timm.create_model('xception',pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = finetune
        if finetune:
            self.model.train()
        else:
            self.model.eval()
        self.model.fc = torch.nn.Linear(2048, num_classes)
        if finetune:
            self.reg_params = list(self.model.parameters())
        else:
            self.reg_params = list(self.model.fc.parameters())
        self.noreg_params = []

    def forward(self, x):
        x = self.model(x)
        return x

class Res_Ghost(nn.Module):
    def __init__(self, num_class, model_type="base"):
        super(Res_Ghost, self).__init__()
        self.num_class = num_class

        self.model = models.resnet18(pretrained=True)
        # self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.model = train_BN(self.model)

        self.model.fc = torch.nn.Linear(512, self.num_class)
        self.reg_params = list(getattr(self.model, "fc").parameters())
        self.noreg_params = []
        self.reg_params = add_parameters(self.reg_params, self.model, nn.BatchNorm2d)

        self.model = add_ghostnet(self.model, model_type)
        self.reg_params = add_parameters(self.reg_params, self.model, ReGhos_Block)

    def forward(self, x):
        x = self.model(x)
        return x

class Xception_Ghost(nn.Module):
    def __init__(self, num_class):
        super(Xception_Ghost, self).__init__()
        self.num_class = num_class

        self.model = timm.create_model('xception',pretrained=True)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()

        self.model = train_BN(self.model)

        self.model.fc = torch.nn.Linear(2048, self.num_class)

        self.reg_params = list(getattr(self.model, "fc").parameters())
        self.noreg_params = []
        self.reg_params = add_parameters(self.reg_params, self.model, nn.BatchNorm2d)

        self.model = add_ghostnet(self.model)

        self.reg_params = add_parameters(self.reg_params, self.model, ReGhos_Block)

    def forward(self, x):
        x = self.model(x)
        return x

class Res_Ghost_A(nn.Module):
    def __init__(self, num_class):
        super(Res_Ghost_A, self).__init__()

        self.num_class = num_class
    
        self.model = models.resnet18(pretrained=True)
        # self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model = train_BN(self.model)
    
        fc_in_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_in_feature, self.num_class)

        self.reg_params = list(getattr(self.model, "fc").parameters())
        self.noreg_params = []
        self.reg_params = add_parameters(self.reg_params, self.model, nn.BatchNorm2d)
    
        self.model = add_ghost_a(self.model)

        self.reg_params = add_parameters(self.reg_params, self.model, Ghost_Block_Combine_base)
        self.noreg_params = add_parameters(self.noreg_params, self.model, Feat_Choice_Layer)

    def forward(self, x):
        x = self.model(x)
        return x

class li18(nn.Module):
    def __init__(self, num_class):
        super(li18, self).__init__()

        self.num_class = num_class
    
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
    
        fc_in_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_in_feature, self.num_class)

        self.reg_params = list(getattr(self.model, "fc").parameters())
        self.noreg_params = []
    
        self.model = add_residule(self.model)

        self.reg_params = add_parameters(self.reg_params, self.model, Residule_block)

    def forward(self, x):
        x = self.model(x)
        return x
    
class li50(nn.Module):
    def __init__(self, num_class):
        super(li50, self).__init__()

        self.num_class = num_class
    
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
    
        fc_in_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_in_feature, self.num_class)

        self.reg_params = list(getattr(self.model, "fc").parameters())
        self.noreg_params = []
    
        self.model = add_residule(self.model)

        self.reg_params = add_parameters(self.reg_params, self.model, Residule_block)

    def forward(self, x):
        x = self.model(x)
        return x