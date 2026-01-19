import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, knn_graph
import render

class Generator(nn.Module):
    def __init__(self, latent_dim, features, num_points, out_dim):
        super(Generator, self).__init__()
        self.features = features

        self.layer1 = nn.Sequential(
            
            nn.Linear(latent_dim, num_points * features),
            nn.Unflatten(1, (features, num_points)),
        )

    def _graphConvBlock(self, x, in_dim, out_dim, k, final_layer=False):

        B, C, N = x.shape

        pos = x[:, :3, :]
        pos = pos.permute(0, 2, 1)
        pos_flat = pos.reshape(B * N, 3)

        batch = torch.cat([
            torch.full((N,), i, device=x.device, dtype=torch.long)
            for i in range(B)
        ])

        edge_index = knn_graph(pos_flat, k=k, batch=batch)
        x_flat = x.permute(0, 2, 1).reshape(B * N, C)
    
        mlp = nn.Sequential(
            nn.Linear(in_dim*2, self.features),
            nn.ReLU(),
            nn.Linear(self.features, out_dim),
        ).to(x.device)

        edge_conv = EdgeConv(mlp, aggr="max")
        out_flat =  edge_conv(x_flat, edge_index)

        if final_layer is False:
            return out_flat.view(B, out_dim, N)
        else:
            return out_flat.view(B, N, out_dim)


    def forward(self, x):

        x = self.layer1(x)
        print(x.shape)
        x = self._graphConvBlock(x, 3, 6, k=20)
        print(x.shape)
        return self._graphConvBlock(x, 6, 12, k=20, final_layer=True)


if __name__ == "__main__":

    latent = torch.randn(10, 100)

    generator = Generator(100, 3, 100, 3)
    out = generator(latent)


    print(out.shape)















