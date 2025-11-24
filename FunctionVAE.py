import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sympy as sp
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# --- 1. Dataset 类：从表达式中构造系数向量 ---
class ExpressionDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.expressions = df["Formula"].tolist()

        self.symbols = [sp.Symbol(f'x_{i+1}') for i in range(9)]
        self.basis = self._build_basis(self.expressions, self.symbols)

        self.vectors = []
        for expr_str in tqdm(self.expressions, desc="Parsing expressions"):
            try:
                expr = sp.sympify(expr_str)
                coeffs = [float(expr.coeff(b)) for b in self.basis]
            except Exception:
                coeffs = [0.0] * len(self.basis)
            self.vectors.append(coeffs)

        self.vectors = torch.tensor(self.vectors, dtype=torch.float32)
        self.mean = self.vectors.mean(dim=0, keepdim=True)
        self.std = self.vectors.std(dim=0, keepdim=True)
        self.vectors = (self.vectors - self.mean) / (self.std + 1e-8)

    def _build_basis(self, expr_list, symbols):
        terms = set()
        for expr_str in tqdm(expr_list, desc="Building basis"):
            try:
                expr = sp.sympify(expr_str)
                for term in expr.as_ordered_terms():
                    _, muls = term.as_coeff_mul(*symbols)
                    terms.add(sp.Mul(*muls))
            except Exception:
                continue
        basis = sorted(terms, key=lambda t: str(t))
        return basis

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]

# --- 2. 更深的 VAE ---
class DeepFunctionVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.fc_mu_logvar = nn.Linear(32, latent_dim * 2)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu_logvar = self.fc_mu_logvar(h)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# --- 3. 损失函数 ---
def vae_loss(recon_x, x, mu, logvar, beta=0.01):
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# --- 4. 从系数向量还原表达式 ---
def coefficients_to_expression(coeffs, basis, round_to_int=False):
    if round_to_int:
        coeffs = [int(np.clip(round(c), -5, 5)) for c in coeffs]
    return sum(c * b for c, b in zip(coeffs, basis))

# --- 5. 主训练与测试流程 ---
def main():
    dataset = ExpressionDataset("GeneratedFeynmanSubset.csv")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_dim = dataset.vectors.shape[1]

    model = DeepFunctionVAE(input_dim=input_dim, latent_dim=32)
    train_vae(model, loader, dataset, epochs=60)

# --- 6. 训练函数 ---
def train_vae(model, loader, dataset, epochs=60, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            beta = min(1.0, epoch / (epochs * 0.3)) * 0.01
            loss = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{epoch+1}/{epochs}] Loss: {total_loss / len(loader):.4f}")

    # --- 可视化重建效果 ---
    model.eval()
    x_vals = np.linspace(0.1, 5.0, 1000)
    x = sp.Symbol('x', real=True, positive=True)
    with torch.no_grad():
        plt.figure(figsize=(12, 16))
        for i in range(5):
            inp = dataset[i].unsqueeze(0).to(device)
            recon, _, _ = model(inp)

            true_coef = inp.cpu() * dataset.std + dataset.mean
            recon_coef = recon.cpu() * dataset.std + dataset.mean

            f_true = coefficients_to_expression(true_coef.squeeze().tolist(), dataset.basis, round_to_int=True)
            f_recon = coefficients_to_expression(recon_coef.squeeze().tolist(), dataset.basis, round_to_int=True)

            print(f"Original Function {i+1}:")
            sp.pprint(f_true)
            print(f"\nReconstructed Function {i+1}:")
            sp.pprint(f_recon)
            print("=" * 80)

            y_true = [float(f_true.subs(x, val)) for val in x_vals]
            y_recon = [float(f_recon.subs(x, val)) for val in x_vals]

            plt.subplot(5, 1, i + 1)
            plt.plot(x_vals, y_true, label='Original')
            plt.plot(x_vals, y_recon, '--', label='Reconstructed')
            plt.title(f"Sample {i + 1}")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
