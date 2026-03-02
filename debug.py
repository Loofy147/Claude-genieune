import torch
import torch.nn as nn

class Head(nn.Module):
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        b, t, d = x.size()
        return x

head = Head()
x = torch.randn(2, 32, 512)
head(x)
