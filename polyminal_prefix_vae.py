import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 生成随机五阶多项式并计算值
def generate_polynomial(num_samples=10000, x_range=(-1, 1), num_points=100):
    X = np.linspace(x_range[0], x_range[1], num_points)
    coefficients = np.random.uniform(-1, 1, size=(num_samples, 6))
    degree_weights = np.exp(-np.arange(6) * 0.5)
    coefficients *= degree_weights
    return coefficients, X

# Min-Max 归一化
def prefix_encode(coefficients):
    min_val = coefficients.min(axis=0, keepdims=True)
    max_val = coefficients.max(axis=0, keepdims=True)
    normalized_coeffs = 2 * (coefficients - min_val) / (max_val - min_val) - 1
    return np.flip(normalized_coeffs, axis=1).copy()

# 生成数据
num_samples = 10000
coefficients, X = generate_polynomial(num_samples)
prefix_encoded_coefficients = prefix_encode(coefficients)
prefix_encoded_coefficients = torch.tensor(prefix_encoded_coefficients, dtype=torch.float32)

class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 32)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

vae = VAE()
optimizer = optim.AdamW(vae.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

num_epochs = 500
batch_size = 128
dataset = torch.utils.data.TensorDataset(prefix_encoded_coefficients)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_losses, kl_losses, recon_losses = [], [], []

for epoch in range(num_epochs):
    vae.train()
    epoch_loss, epoch_kl, epoch_recon = 0, 0, 0
    beta = min(0.1, epoch / 100.0)

    for batch in dataloader:
        batch = batch[0]
        optimizer.zero_grad()
        reconstructed, mean, logvar = vae(batch)
        
        recon_loss = 0.8 * F.mse_loss(reconstructed, batch, reduction="mean") + 0.2 * F.l1_loss(reconstructed, batch, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_kl += kl_loss.item()
        epoch_recon += recon_loss.item()
    
    scheduler.step(epoch_loss / len(dataloader))
    train_losses.append(epoch_loss / len(dataloader))
    kl_losses.append(epoch_kl / len(dataloader))
    recon_losses.append(epoch_recon / len(dataloader))

# 生成测试数据
test_coefficients, _ = generate_polynomial(1)
test_prefix_encoded = prefix_encode(test_coefficients)
test_prefix_encoded = torch.tensor(test_prefix_encoded, dtype=torch.float32).unsqueeze(0)

vae.eval()
with torch.no_grad():
    encoded_mean, encoded_logvar = vae.encode(test_prefix_encoded)
    encoded_z = vae.reparameterize(encoded_mean, encoded_logvar)
    decoded_coefficients = vae.decode(encoded_z).numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Total Loss')
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')

plt.subplot(1, 2, 2)
plt.plot(X, np.polyval(test_coefficients[0][::-1], X), label="Original", linestyle="dashed")
plt.plot(X, np.polyval(decoded_coefficients[0][::-1], X), label="Reconstructed", linestyle="solid")
plt.legend()
plt.title("Original vs Reconstructed Polynomial")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()