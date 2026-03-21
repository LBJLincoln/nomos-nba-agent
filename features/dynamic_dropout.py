import torch
import torch.nn as nn
from typing import Optional

class DynamicDropout(nn.Module):
    """Dropout that decreases over training epochs"""
    def __init__(self, initial_dropout: float = 0.5, final_dropout: float = 0.1, num_epochs: int = 100):
        super(DynamicDropout, self).__init__()
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.num_epochs = num_epochs

    def forward(self, x: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        if epoch is None or epoch >= self.num_epochs:
            current_dropout = self.final_dropout
        else:
            progress = epoch / self.num_epochs
            current_dropout = self.initial_dropout - (self.initial_dropout - self.final_dropout) * progress
        return nn.functional.dropout(x, p=current_dropout, training=self.training)
