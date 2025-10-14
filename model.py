import torch
import warnings
import torch.nn.functional as F
from torch_geometric.nn import GATConv
warnings.filterwarnings('ignore')

class GATFraudDetector(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=64, heads=4, num_gnn_layers=3, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.tmp_out = hidden_dim // heads
        
        # Node and Edge embeddings
        self.node_emb = torch.nn.Linear(num_node_features, self.hidden_dim)
        self.edge_emb = torch.nn.Linear(num_edge_features, self.hidden_dim)
        
        # Multiple GAT layers with residual connections
        self.gat_layers = torch.nn.ModuleList()
        self.edge_mlps = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        # GNN layers
        for _ in range(self.num_gnn_layers):
            self.gat_layers.append(GATConv(
                in_channels=self.hidden_dim,
                out_channels=self.tmp_out,
                heads=self.heads,
                edge_dim=self.hidden_dim,
                dropout=self.dropout,
                concat=True,
                add_self_loops=True
            ))
            self.edge_mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                )
            )
            self.layer_norms.append(torch.nn.LayerNorm(self.hidden_dim))

        # Classifier with multiple layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dest = edge_index
        
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        
        # Apply GAT layers
        for i, (gat, edge_mlp, ln) in enumerate(zip(self.gat_layers[:-1], self.edge_mlps, self.layer_norms)):
            x_new = F.relu(ln(gat(x, edge_index, edge_attr)))
            x = x + x_new # Residual Connection
            edge_attr = edge_attr + edge_mlp(torch.cat([x[src], x[dest], edge_attr], dim=-1))
        
        # Edge-level prediction: concatenate source node, destination node and edge features
        edge_features = torch.cat([x[src], x[dest], edge_attr], dim=-1)
        
        return self.classifier(edge_features).squeeze()
