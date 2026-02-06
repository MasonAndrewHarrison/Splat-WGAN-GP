import torch
import torch.nn as nn
import torch.nn.functional as F
import render

#TODO create a PointNet (Point-wise MLPs) Model
#TODO PointNet with knn with pointnet2_ops for knn gpu support
#TODO look into using transformers or EdgeConv?

class Generator(nn.Module):

    def  __init__(self, latent_dim, features, num_points, out_dim):
        super(Generator, self).__init__()

        self.num_points = num_points
        self.features = features

        self.linear_layers = nn.Sequential(
            self._Linear_Block(latent_dim, features*4, use_layer_norm=False),
            self._Linear_Block(features*4, features*16, use_layer_norm=False),
            self._Linear_Block(features*16, features*32, use_layer_norm=False),
            self._Linear_Block(features*32, features*64, use_layer_norm=False),
            self._Linear_Block(features*64, num_points * features, use_layer_norm=False)
        ) 

        self.point_wise_mpls = nn.Sequential(
            self._Point_Wise_MLP(features, features*2),
            self._Point_Wise_MLP(features*2, features*4),
            self._Point_Wise_MLP(features*4, features*8),
        )

        self.final_layer = nn.Sequential(
            nn.Conv1d(features*8, out_dim, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _Linear_Block(in_channels, out_channels, use_layer_norm=True):

        layers = [
            nn.Linear(in_channels, out_channels)
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        #layers.append(nn.Dropout(p=0.05))
        return nn.Sequential(*layers)

    # Also called shared MLP
    @staticmethod
    def _Point_Wise_MLP(in_channels, out_channels, use_instance_norm=True):
        
        layers = [
            nn.Conv1d(in_channels, out_channels, 1)
        ]
        if use_instance_norm:
            layers.append(nn.InstanceNorm1d(out_channels, affine=True)) 
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.linear_layers(x)
        out = out.reshape(-1, self.features, self.num_points)
        out = self.point_wise_mpls(out)
        out = self.final_layer(out)
        return out

class PC_Critic(nn.Module):

    def  __init__(self, features, in_dim):
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
            nn.Linear(features*4, 1)
        )

    # Also called shared MLP
    @staticmethod
    def _Point_Wise_MLP(in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.InstanceNorm1d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _Linear_Block(in_channels, out_channels):

        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.point_wise_mpls(x)
        out_max = torch.max(out, 2)[0]
        out_avg = torch.mean(out, 2)
        out = torch.cat([out_max, out_avg], dim=1)
        out = self.linear_layers(out)
        return out

def init_generator(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if hasattr(m, 'final_layer'):
                nn.init.normal_(m.weight, 0.0, 0.02) 
            else:
                nn.init.xavier_normal_(m.weight, gain=0.5) 
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0 if m.bias is not None else None)

def init_critic(model):

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
    
    latent = torch.randn(1, 128)
    generator = Generator(128, features=32, num_points=1024, out_dim=6)
    critic = PC_Critic(features=32, in_dim=6)

    init_generator(generator)
    init_critic(critic)

    point_cloud = generator(latent)
    print(point_cloud.shape)
    
    out = critic(point_cloud)
    print(out.shape)


