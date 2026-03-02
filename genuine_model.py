import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GenuineAttentionHead(nn.Module):
    def __init__(self, d_model, d_head, dropout=0.1):
        super().__init__()
        self.d_head = d_head
        self.q = nn.Linear(d_model, d_head)
        self.k = nn.Linear(d_model, d_head)
        self.v = nn.Linear(d_model, d_head)
        self.dropout = nn.Dropout(dropout)
        self.current_entropy = None

    def forward(self, x, mask=None):
        # x: (batch, seq, d_model)
        q = self.q(x) # (b, t, d_head)
        k = self.k(x) # (b, t, d_head)
        v = self.v(x) # (b, t, d_head)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            # BROADCAST FIX: ensure mask is compatible with scores (b, t, t)
            # scores: (b, t, t), mask: (1, 1, t, t) -> (b, t, t)
            # mask.squeeze(1) gives (1, t, t) which broadcasts correctly with (b, t, t)
            m = mask.squeeze(1) if mask.dim() == 4 else mask
            scores = scores.masked_fill(m == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        with torch.no_grad():
            eps = 1e-10
            h = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
            self.current_entropy = h

        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        return out

class GenuineTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.heads = nn.ModuleList([GenuineAttentionHead(d_model, self.d_head, dropout) for _ in range(n_heads)])
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        ln_x = self.ln1(x)
        head_outputs = [head(ln_x, mask) for head in self.heads]
        attn_out = torch.cat(head_outputs, dim=-1)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class GenuineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=8, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([GenuineTransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_emb.weight = self.head.weight

    def forward(self, tokens, labels=None):
        b, t = tokens.size()
        # mask: (1, 1, t, t)
        mask = torch.tril(torch.ones((t, t), device=tokens.device)).view(1, 1, t, t)
        x = self.token_emb(tokens) + self.pos_emb[:, :t, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

def compute_genuineness_regularization(model, lambda_var=0.1, lambda_col=0.05):
    total_var = 0
    total_col = 0
    head_count = 0
    for layer in model.layers:
        for head in layer.heads:
            if head.current_entropy is None: continue
            var_h = torch.var(head.current_entropy, dim=-1).mean()
            total_var += var_h
            diffs = head.current_entropy[:, 1:] - head.current_entropy[:, :-1]
            collapses = torch.relu(-diffs - 0.1).sum(dim=-1).mean()
            total_col += collapses
            head_count += 1
    if head_count == 0: return torch.tensor(0.0, device=model.pos_emb.device)
    return -(lambda_var * (total_var / head_count) + lambda_col * (total_col / head_count))

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GenuineTransformer(vocab_size=1000).to(device)
    x = torch.randint(0, 1000, (2, 32)).to(device)
    logits, loss = model(x, labels=x)
    reg = compute_genuineness_regularization(model)
    print(f"Verified: Logits={logits.shape}, Base Loss={loss.item():.4f}, Gen Reg={reg.item():.4f}")
