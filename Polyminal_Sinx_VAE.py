import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sympy as sp
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Generate polynomial data
def generate_polynomial():
    x = sp.Symbol('x')
    terms = []
    terms.append(round(random.uniform(-1, 1), 2))

    for degree in range(1, 7):
        max_coef = 1.0 / (2 ** degree)
        coef = round(random.uniform(-max_coef, max_coef), 3)
        terms.append(coef * x**degree)

    sin_coef = round(random.uniform(-1, 1), 3)
    terms.append(sin_coef * sp.sin(x))

    return sum(terms)

# Dataset class
class CoefficientDataset(Dataset):
    def __init__(self, polynomials):
        self.coefficients = []
        x = sp.Symbol('x')

        for poly in tqdm(polynomials, desc="Extracting coefficients"):
            expanded = sp.expand(poly)
            coeffs = [expanded.coeff(x, i) for i in range(7)]
            coeffs = [float(c) if c != 0 else 0.0 for c in coeffs]
            sin_coef = float(expanded.coeff(sp.sin(x))) if expanded.coeff(sp.sin(x)) else 0.0
            coeffs.append(sin_coef)
            self.coefficients.append(coeffs)

        self.coefficients = torch.tensor(self.coefficients, dtype=torch.float32)

        self.mean = self.coefficients.mean(dim=0, keepdim=True)
        self.std = self.coefficients.std(dim=0, keepdim=True)
        self.coefficients = (self.coefficients - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.coefficients)

    def __getitem__(self, idx):
        return self.coefficients[idx]

# VAE model
class Polynomial_Sinx_VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32), 
            nn.ReLU(),
        )
        
        # Output mu and logvar
        self.fc_mu_logvar = nn.Linear(32, latent_dim * 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32), 
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),

            nn.Linear(64, 8),
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


# VAE loss function
def vae_loss(recon_x, x, mu, logvar, beta=0.01):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# Training function
def train_vae(model, train_loader, num_epochs=20, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Save model
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Load model
def load_model(model_path="polynomial_sinx_vae_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Polynomial_Sinx_VAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# Convert coefficients to polynomial
def coefficients_to_polynomial(coeffs):
    x = sp.Symbol('x')
    terms = [coef * x**i for i, coef in enumerate(coeffs[:-1])]
    sin_coef = coeffs[-1]
    if abs(sin_coef) > 1e-6:
        terms.append(sin_coef * sp.sin(x))
    return sum(terms)

    
# Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_polynomials = 5000
    latent_dim = 8
    num_epochs = 100
    batch_size = 16

    polynomials = [generate_polynomial() for _ in range(num_polynomials)]
    coeff_dataset = CoefficientDataset(polynomials)
    train_loader = DataLoader(coeff_dataset, batch_size=batch_size, shuffle=True)

    model = Polynomial_Sinx_VAE(latent_dim=latent_dim)
    train_vae(model, train_loader, num_epochs=num_epochs)

    save_model(model, 'polynomial_sinx_vae_model.pth')
    
    model.eval()
    with torch.no_grad():
        plt.figure(figsize=(12, 20))
        x_vals = np.linspace(-1, 1, 100)
        for i in range(5):
            orig_coeffs = coeff_dataset[i].unsqueeze(0).to(device)
            recon_coeffs, _, _ = model(orig_coeffs)

            orig_poly = coefficients_to_polynomial(orig_coeffs.squeeze())
            recon_poly = coefficients_to_polynomial(recon_coeffs.squeeze())

            orig_y = np.array([float(orig_poly.subs('x', xi)) for xi in x_vals])
            recon_y = np.array([float(recon_poly.subs('x', xi)) for xi in x_vals])

            plt.subplot(5, 1, i+1)
            plt.plot(x_vals, orig_y, label=f'Original: {sp.simplify(orig_poly)}')
            plt.plot(x_vals, recon_y, '--', label=f'Reconstructed: {sp.simplify(recon_poly)}')
            plt.legend()
            plt.title(f'Sample {i+1}')
            plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
