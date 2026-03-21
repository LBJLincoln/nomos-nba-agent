import torch
import torch.nn as nn
from models.graph_net import NBAGraphNet

class DropoutWrapper(nn.Module):
    def __init__(self, model: NBAGraphNet, dropout: float = 0.5):
        super(DropoutWrapper, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_tab: torch.Tensor, node_features: torch.Tensor, adj_matrices: torch.Tensor) -> torch.Tensor:
        x = self.model(x_tab, node_features, adj_matrices)
        x = self.dropout(x)
        return x
