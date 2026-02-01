import torch
import torch.nn as nn
import torch.nn.functional as F
import render

#TODO create a PointNet (Point-wise MLPs) Model
#TODO PointNet with knn with pointnet2_ops for knn gpu support
#TODO look into using transformers and PointNet maybe?


def initialize_weight(model):

    for m in model.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)