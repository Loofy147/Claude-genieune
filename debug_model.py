import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Head(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q = nn.Linear(d_model, d_head)
        self.k = nn.Linear(d_model, d_head)
        self.v = nn.Linear(d_model, d_head)
        self.d_head = d_head

    def forward(self, x, mask=None):
        print(f"Head input x shape: {x.shape}")
        b, t, d = x.size()
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            print(f"Mask shape: {mask.shape}, scores shape: {scores.shape}")
            scores = scores.masked_fill(mask == 0, -1e9)
            print(f"Scores after mask shape: {scores.shape}")
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.head = Head(d_model, d_model)

    def forward(self, x, mask=None):
        return self.head(x, mask)

d_model = 256
t = 32
b = 2
x = torch.randn(b, t, d_model)
mask = torch.tril(torch.ones((t, t))).view(1, 1, t, t)
model = Block(d_model)
out = model(x, mask)
print(f"Output shape: {out.shape}")
