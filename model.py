import torch
import torch.nn as nn
import torch.nn.functional as F
import render

#TODO create a PointNet (Point-wise MLPs) Model
#TODO PointNet with knn with pointnet2_ops for knn gpu support
#TODO look into using transformers and PointNet maybe?

class Generator(nn.Module):

    def  __init__(self, latent_dim, features, num_points, out_dim):
        super(Generator, self).__init__()

    def forward(self, x):
        return x

class PC_Critic(nn.Module):

    def  __init__(self, features, num_points, in_dim):
        super(PC_Critic, self).__init__()

        self.point_wise_mpls = nn.Sequential(
            self._Point_Wise_MLP(in_dim, features),
            self._Point_Wise_MLP(features, features*2),
            self._Point_Wise_MLP(features*2, features*16)
        )

        self.linear_layers = nn.Sequential(
            self._Linear_Block(features*32, features*16),
            self._Linear_Block(features*16, features*8),
            self._Linear_Block(features*8, features*4),
            self._Linear_Block(features*4, 1)

        )

    # Also called shared MLP
    @staticmethod
    def _Point_Wise_MLP(in_channels, out_channels, use_batch_norm=True):
        
        layers = [
            nn.Conv1d(in_channels, out_channels, 1)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels)) 
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _Linear_Block(in_channels, out_channels, use_batch_norm=True):

        layers = [
            nn.Linear(in_channels, out_channels)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=0.25))
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.point_wise_mpls(x)
        out_max = torch.max(out, 2)[0]
        out_avg = torch.mean(out, 2)
        out = torch.cat([out_max, out_avg], dim=1)
        print(out.shape)
        out = self.linear_layers(out)
        return out


if __name__ == '__main__':
    
    point_cloud = torch.randn(1, 6, 1024)
    print(point_cloud.shape)
    critic = PC_Critic(32, 1024, 6)
    out = critic(point_cloud)
    print(out.shape)




def initialize_weight(model):

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)