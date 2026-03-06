import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from genuine_model import GenuineTransformer, compute_genuineness_regularization
from datetime import datetime
import os

class SyntheticReasoningDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        if idx % 2 == 0:
            p1, p2 = np.random.randint(10, 500), np.random.randint(501, 1000)
            filler = np.random.randint(1001, self.vocab_size // 2, size=self.seq_len - 3)
            data = np.concatenate([[p1, p2], filler, [p1]])
            target = np.roll(data, -1)
            target[-1] = p2
        else:
            pattern = np.random.randint(self.vocab_size // 2, self.vocab_size, size=4)
            data = np.tile(pattern, self.seq_len // 4 + 1)[:self.seq_len]
            target = np.roll(data, -1)
            target[-1] = data[0]

        return torch.tensor(data, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Improved Model on {device}...")
    vocab_size = 50257

    model = GenuineTransformer(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        max_seq_len=64
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    # Increased num_samples and batch_size
    dataset = SyntheticReasoningDataset(vocab_size, seq_len=64, num_samples=2000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    gen_states = []

    # Increased number of epochs for much more time
    for epoch in range(10):
        epoch_loss = 0
        epoch_reg = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, base_loss = model(x, labels=y)
            gen_reg = compute_genuineness_regularization(model, lambda_var=2.0, lambda_col=1.0)
            total_loss = base_loss + gen_reg
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += base_loss.item()
            epoch_reg += -gen_reg.item()
            gen_states.append(-gen_reg.item())
        print(f"Epoch {epoch+1}: Base Loss={epoch_loss/len(loader):.4f}, Gen Signal={epoch_reg/len(loader):.4f}")

    torch.save(model.state_dict(), "genuine_transformer_v1.pt")
    print("Improved Model saved.")

if __name__ == "__main__":
    train()
