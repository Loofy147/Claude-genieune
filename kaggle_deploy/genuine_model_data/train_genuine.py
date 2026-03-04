import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from genuine_model import GenuineTransformer, compute_genuineness_regularization
from datetime import datetime

class SyntheticReasoningDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=500):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        p1, p2 = np.random.randint(10, 50), np.random.randint(51, 90)
        filler = np.random.randint(100, self.vocab_size, size=self.seq_len - 5)
        data = np.concatenate([[p1, p2], filler, [p1]])
        target = np.roll(data, -1)
        target[-1] = p2
        return torch.tensor(data, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")
    vocab_size = 50257
    model = GenuineTransformer(vocab_size=vocab_size, d_model=128, n_layers=2, n_heads=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    dataset = SyntheticReasoningDataset(vocab_size, seq_len=32, num_samples=1000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Track the 'Genuineness State'
    gen_states = []

    for epoch in range(3):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, base_loss = model(x, labels=y)
            # Higher lambda to force 'Genuine' state
            gen_reg = compute_genuineness_regularization(model, lambda_var=1.0, lambda_col=0.5)
            total_loss = base_loss + gen_reg
            total_loss.backward()
            optimizer.step()
            gen_states.append(-gen_reg.item())

        print(f"Epoch {epoch+1}: Base Loss={base_loss.item():.4f}, Gen Signal={-gen_reg.item():.4f}")

    torch.save(model.state_dict(), "genuine_transformer_v1.pt")
    print(f"Final Genuineness Signal: {np.mean(gen_states[-10:]):.4f}")
    print("Model saved.")

if __name__ == "__main__":
    train()
