import torch
import torch.nn as nn

class TildeQLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        q1 = (y_pred - y_true) ** 2
        q2 = torch.abs(torch.fft.fft(y_pred, dim=-1) - torch.fft.fft(y_true, dim=-1)) ** 2
        q = self.alpha * q1 + self.beta * q2.mean(dim=-1)
        loss = q.sum()
        return loss
