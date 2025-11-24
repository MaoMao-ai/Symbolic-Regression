# symbolic_regression_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sympy as sp
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from sympy import preorder_traversal

# ---------------------- 1. Utils for expression and data ----------------------
OPERATORS = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log']
TERMINALS = ['x'] + [str(i) for i in range(-5, 6)]
TOKENS = OPERATORS + TERMINALS
TOKEN2IDX = {tok: i for i, tok in enumerate(TOKENS)}
IDX2TOKEN = {i: tok for tok, i in TOKEN2IDX.items()}


def generate_random_expr(max_depth=3):
    if max_depth == 0 or random.random() < 0.3:
        return random.choice(TERMINALS)
    op = random.choice(OPERATORS)
    if op in ['sin', 'cos', 'exp', 'log']:
        return f"{op}({generate_random_expr(max_depth-1)})"
    else:
        return f"({generate_random_expr(max_depth-1)} {op} {generate_random_expr(max_depth-1)})"


def expr_to_tokens(expr: str) -> List[str]:
    expr = sp.sympify(expr)
    return list(map(str, preorder_traversal(expr)))


def tokens_to_tensor(tokens: List[str], max_len: int) -> torch.Tensor:
    idxs = [TOKEN2IDX.get(tok, 0) for tok in tokens]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    return torch.tensor(idxs[:max_len])


def eval_expr(expr: str, x_vals: np.ndarray) -> np.ndarray:
    x = sp.Symbol('x')
    try:
        sympy_expr = sp.sympify(expr)
        f = sp.lambdify(x, sympy_expr, modules='numpy')
        y = f(x_vals)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"[WARN] Invalid expr: {expr}, error: {e}")
        y = np.zeros_like(x_vals)
    return y



# ---------------------- 2. Dataset ----------------------
class ExprDataset(Dataset):
    def __init__(self, n_samples=1000, max_len=30):
        self.samples = []
        self.max_len = max_len
        self.x_vals = np.linspace(-5, 5, 100)
        for _ in range(n_samples):
            expr = generate_random_expr()
            tokens = expr_to_tokens(expr)
            y_vals = eval_expr(expr, self.x_vals)
            self.samples.append((tokens, y_vals, expr))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, y_vals, expr = self.samples[idx]
        return tokens_to_tensor(tokens, self.max_len), torch.tensor(y_vals, dtype=torch.float32), expr


# ---------------------- 3. VAE ----------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.rnn(emb)
        h = h[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_dim=32, hidden_dim=128, max_len=30):
        super().__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.max_len = max_len

    def forward(self, z):
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        inputs = h0.repeat(self.max_len, 1, 1).permute(1, 0, 2)
        out, _ = self.rnn(inputs)
        logits = self.out(out)
        return logits


class VAE(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        logits = self.decoder(z)
        return logits, mu, logvar

    def loss(self, x, logits, mu, logvar):
        recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), ignore_index=0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl


# ---------------------- 4. Training ----------------------
def train_vae():
    dataset = ExprDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = VAE(vocab_size=len(TOKENS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0
        for x, y_vals, _ in dataloader:
            logits, mu, logvar = model(x)
            loss = model.loss(x, logits, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

    return model, dataset


if __name__ == '__main__':
    train_vae()
