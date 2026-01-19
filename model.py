import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.edge_conv1 = EdgeConv(self._create_mlp(3*2, features, 6), aggr="max")
        self.edge_conv2 = EdgeConv(self._create_mlp(6*2, features, 6), aggr="max")

    def _create_mlp(self, in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def _graphConvBlock(self, x, edge_conv, k, final_layer=False):

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
        out_flat = edge_conv(x_flat, edge_index)

        if final_layer is False:
            return out_flat.view(B, -1, N)
        else:
            return out_flat.view(B, N, -1)


    def forward(self, x):

        x = self.layer1(x)
        print(x.shape)
        x = self._graphConvBlock(x, self.edge_conv1, k=20)
        print(x.shape)
        x = self._graphConvBlock(x, self.edge_conv2, k=20, final_layer=True)
        print(x.shape)
        output = torch.cat([
            x[:, :, :3],
            F.sigmoid(x[:, :, 3:]),
        ], dim=2)
        return output


if __name__ == "__main__":

    latent = torch.randn(10, 100)

    generator = Generator(100, 3, 700, 3)
    out = generator(latent)

    out = out[1, :, :]
    print(out)
    print(out.shape)
    #render.show_model(out)















