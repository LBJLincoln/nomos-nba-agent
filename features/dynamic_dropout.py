import torch
import torch.nn as nn

class DynamicDropout(nn.Module):
    def __init__(self, initial_dropout: float = 0.5, final_dropout: float = 0.1, num_epochs: int = 100):
        super(DynamicDropout, self).__init__()
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.num_epochs = num_epochs
        self.dropout = nn.Dropout(p=initial_dropout)

    def forward(self, x: torch.Tensor, epoch: int) -> torch.Tensor:
        current_dropout = self.initial_dropout - (self.initial_dropout - self.final_dropout) * (epoch / self.num_epochs)
        self.dropout.p = current_dropout
        return self.dropout(x)
