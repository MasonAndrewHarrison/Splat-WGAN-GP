import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, knn_graph



class SimpleGraphGenerator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k=8):
        super().__init__()
        self.k = k

        self.edge_conv = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        )

    def forward(self, x):
        edge_index = knn_graph(x, k=self.k)
        x = self.edge_conv(x, edge_index)
        return x

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    x = torch.randn(1024, 3).to(device)
    model = SimpleGraphGenerator(3, 32, 3).to(device)
    out = model(x)






class Generator(nn.Module):
    def __init__(self, latent_dim, features, num_points, k=20):
        super(Generator, self).__init__()

        self.k = k

        self.edge_conv = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * features)
            )
        )

        self.gen = nn.Sequential(
            
            nn.Linear(latent_dim, )
        )

    def _graphConvBlock(self, in_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(in_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )


    def forward(self, x):

        edge_index = knn_graph(x, k=k)
        x = self.gen(x, edge_index)
        return x 





















class SimpleGraphGenerator(nn.Module):
    """
    Simple point cloud generator using graph convolutions
    
    Flow:
    Random noise → Initial points → Graph Conv layers → Final point cloud
    """
    
    def __init__(self, latent_dim=128, num_points=512):
        super().__init__()
        self.num_points = num_points
        
        # Step 1: Expand latent code to initial point features
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_points * 64)  # Each point gets 64 features
        )
        
        # Step 2: Graph convolutions to refine points
        # Each EdgeConv takes [point_features, neighbor_features] → new_features
        
        self.conv1 = EdgeConv(
            nn.Sequential(
                nn.Linear(64 * 2, 128),  # 64*2 because we concat point + neighbor
                nn.ReLU(),
                nn.Linear(128, 64)
            ),
            aggr='max'
        )
        
        self.conv2 = EdgeConv(
            nn.Sequential(
                nn.Linear(64 * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            ),
            aggr='max'
        )
        
        # Step 3: Convert features to xyz coordinates
        self.to_xyz = nn.Linear(32, 3)
        
    def forward(self, z):
        """
        Args:
            z: Random noise vector (batch_size, latent_dim)
            
        Returns:
            points: Generated point cloud (batch_size, num_points, 3)
        """
        batch_size = z.size(0)
        
        # Step 1: Create initial point features
        x = self.fc(z)  # (batch, num_points * 64)
        x = x.view(batch_size, self.num_points, 64)  # (batch, num_points, 64)
        
        # Process each point cloud in the batch
        all_points = []
        
        for i in range(batch_size):
            features = x[i]  # (num_points, 64)
            
            # Step 2a: Build graph based on features
            # We use features to find neighbors (not xyz yet!)
            edge_index = knn_graph(features, k=16)
            
            # Step 2b: Apply first graph convolution
            features = self.conv1(features, edge_index)  # (num_points, 64)
            
            # Rebuild graph with updated features
            edge_index = knn_graph(features, k=16)
            
            # Step 2c: Apply second graph convolution
            features = self.conv2(features, edge_index)  # (num_points, 32)
            
            # Step 3: Convert to xyz coordinates
            points = self.to_xyz(features)  # (num_points, 3)
            points = torch.tanh(points)  # Normalize to [-1, 1]
            
            all_points.append(points)
        
        return torch.stack(all_points)