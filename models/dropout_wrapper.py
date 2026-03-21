import torch
import torch.nn as nn
from models.graph_net import NBAGraphNet
from features.dynamic_dropout import DynamicDropout

class DropoutWrapper(nn.Module):
    def __init__(self, model: NBAGraphNet, initial_dropout: float = 0.5, final_dropout: float = 0.1, num_epochs: int = 100):
        super(DropoutWrapper, self).__init__()
        self.model = model
        self.dynamic_dropout = DynamicDropout(initial_dropout, final_dropout, num_epochs)

    def forward(self, x_tab: torch.Tensor, node_features: torch.Tensor, adj_matrices: torch.Tensor, epoch: int) -> torch.Tensor:
        x = self.model(x_tab, node_features, adj_matrices)
        x = self.dynamic_dropout(x, epoch)
        return x
