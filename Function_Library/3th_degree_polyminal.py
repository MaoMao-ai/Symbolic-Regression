import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sympy as sp
from scipy.integrate import quad
from datetime import datetime
import pickle
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from itertools import combinations
import math
import random
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_polynomial():
    x = sp.Symbol('x')
    terms = []
    const_term = round(random.uniform(-1, 1), 2)
    terms.append(const_term)

    for degree in range(1, 4):
        max_coef = 1.0 / (2 ** degree)
        coef = round(random.uniform(-max_coef, max_coef), 3)
        terms.append(coef * x**degree)

    return sum(terms)

def compute_l2_distance(f, g, domain=(-5, 5)):
    x = sp.Symbol('x')
    f_func = sp.lambdify(x, f, 'numpy')
    g_func = sp.lambdify(x, g, 'numpy')

    diff_squared = lambda x: (f_func(x) - g_func(x))**2
    distance, _ = quad(diff_squared, domain[0], domain[1])
    return np.sqrt(distance)

def compute_distances_chunk(args):
    chunk_idx, pairs_chunk, polynomials = args
    results = []

    pbar = tqdm(pairs_chunk,
                desc=f'Chunk {chunk_idx + 1}',
                position=chunk_idx + 1,
                leave=False)

    for i, j in pbar:
        try:
            dist = compute_l2_distance(polynomials[i], polynomials[j])
            results.append((i, j, dist))
        except Exception as e:
            print(f"\nError computing distance for pair ({i}, {j}): {e}")
            results.append((i, j, float('nan')))

    pbar.close()
    return results

def compute_distances(polynomials, num_processes=24):
    n = len(polynomials)
    distances = torch.zeros(n, n)

    pairs = list(combinations(range(n), 2))
    total_pairs = len(pairs)

    print(f"\nTotal number of pairs to compute: {total_pairs}")

    chunk_size = math.ceil(total_pairs / num_processes)
    pair_chunks = [pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]

    chunk_args = [
        (idx, chunk, polynomials)
        for idx, chunk in enumerate(pair_chunks)
    ]

    print(f"\nComputing pairwise distances using {num_processes} processes...")
 
    main_pbar = tqdm(total=total_pairs,
                    desc='Overall progress',
                    position=0,
                    leave=True)

    completed_pairs = 0
    with Pool(num_processes) as pool:
        for chunk_results in pool.imap(compute_distances_chunk, chunk_args):
            for i, j, dist in chunk_results:
                distances[i, j] = dist
                distances[j, i] = dist
                completed_pairs += 1
                main_pbar.update(1)

    main_pbar.close()
    print("\nDistance computation completed!")

    distances.fill_diagonal_(0)

    valid_distances = distances[~torch.isnan(distances)]
    print("\nDistance Statistics:")
    print(f"Mean: {valid_distances.mean():.4f}")
    print(f"Min: {valid_distances.min():.4f}")
    print(f"Max: {valid_distances.max():.4f}")
    print(f"Std: {valid_distances.std():.4f}")

    return distances

class PolynomialDataset(Dataset):
    def __init__(self, num_polynomials, num_threads=24, skip_init=False):
        self.polynomials = []
        self.distances = None

        if not skip_init:
            print("\nGenerating polynomials...")
            for _ in tqdm(range(num_polynomials)):
                poly = generate_polynomial()
                self.polynomials.append(poly)

            self.distances = compute_distances(self.polynomials, num_processes=num_threads)

    def __len__(self):
        return len(self.polynomials)

    def __getitem__(self, idx):
        return self.polynomials[idx], self.distances[idx]

class CoefficientDataset(Dataset):
    def __init__(self, polynomials):
        self.coefficients = []
        self.distances = None
        x = sp.Symbol('x')

        print("Extracting coefficients...")
        for poly in tqdm(polynomials):
            expanded = sp.expand(poly)
            coeffs = [expanded.coeff(x, i) for i in range(4)]
            coeffs = [float(c) if c != 0 else 0.0 for c in coeffs]
            self.coefficients.append(coeffs)

        self.coefficients = torch.tensor(self.coefficients, dtype=torch.float32)

        #normalization
        self.mean = self.coefficients.mean(dim=0, keepdim=True)
        self.std = self.coefficients.std(dim=0, keepdim=True)
        self.coefficients = (self.coefficients - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.coefficients)

    def __getitem__(self, idx):
        return self.coefficients[idx]

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.conv2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.1)
        
        # 如果输入和输出维度不同，添加shortcut连接
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.elu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += self.shortcut(identity)
        out = F.elu(out)
        
        return out

class PolynomialVAE(nn.Module):
    def __init__(self, latent_dim=12):
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器 (Encoder)
        self.encoder_input = nn.Linear(4, 64)
        self.encoder_block1 = ResNetBlock(64, 64)
        self.encoder_block2 = ResNetBlock(64, 32)
        self.encoder_block3 = ResNetBlock(32, 16)
        self.encoder_dropout = nn.Dropout(0.1)

        # 提取均值和方差
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)

        # 解码器 (Decoder)
        self.decoder_input = nn.Linear(latent_dim, 32)
        self.decoder_block1 = ResNetBlock(32, 32)
        self.decoder_block2 = ResNetBlock(32, 64)
        self.decoder_block3 = ResNetBlock(64, 128)
        self.decoder_block4 = ResNetBlock(128, 256)
        self.decoder_dropout = nn.Dropout(0.1)
        self.decoder_output = nn.Linear(256, 4)

    def encode(self, x):
        x = self.encoder_input(x)
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_dropout(x)
        return self.fc_mu(x), self.fc_var(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)
        x = self.decoder_block4(x)
        x = self.decoder_dropout(x)
        return self.decoder_output(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=0.005):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def train_vae(model, train_loader, num_epochs=400, lr=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch)

            # 监控 mu 和 logvar
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}: mu mean {mu.mean().item():.4f}, logvar mean {logvar.mean().item():.4f}")

            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]["lr"]:.6f}')


def save_model(model, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': model.latent_dim
    }, filename)
    print(f"\nModel saved to {filename}")

def load_model(filename):
    checkpoint = torch.load(filename)
    model = PolynomialVAE(latent_dim=checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nModel loaded from {filename}")
    return model

def plot_reconstruction(model, dataset, num_samples=5, domain=(-5, 5)):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            orig_coeffs = dataset[i].to(device)
            recon_coeffs, _, _ = model(orig_coeffs.unsqueeze(0))

            orig_poly = coefficients_to_polynomial(orig_coeffs)
            recon_poly = coefficients_to_polynomial(recon_coeffs[0])

            x = np.linspace(domain[0], domain[1], 200)
            orig_y = np.array([float(orig_poly.subs('x', xi)) for xi in x])
            recon_y = np.array([float(recon_poly.subs('x', xi)) for xi in x])


            ax.plot(x, orig_y, 'b-', label='Original')
            ax.plot(x, recon_y, 'r--', label='Reconstructed')

            distance = compute_l2_distance(orig_poly, recon_poly)
            ax.set_title(f'Sample {i+1} (L2 Distance: {distance:.4f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True)

            print(f"\nSample {i+1}:")
            print(f"Original: {orig_poly}")
            print(f"Reconstructed: {recon_poly}")

        plt.tight_layout()
        plt.show()

def coefficients_to_polynomial(coeffs):
    x = sp.Symbol('x')
    terms = []
    coeffs = coeffs.detach().cpu().numpy() if isinstance(coeffs, torch.Tensor) else coeffs
    for i, coeff in enumerate(coeffs):
        if abs(coeff) > 1e-6:
            terms.append(coeff * x**i)
    return sum(terms)

def sample_from_latent(model, num_samples=5):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim)
        coeffs = model.decode(z)

        plt.figure(figsize=(12, 8))

        polynomials = []
        for i in range(num_samples):
            poly = coefficients_to_polynomial(coeffs[i])
            polynomials.append(poly)
            x = np.linspace(-5, 5, 200)
            y = np.array([float(poly.subs('x', xi)) for xi in x])
            plt.plot(x, y, label=f'Sample {i+1}')
            print(f"\nSample {i+1}: {poly}")

        plt.title('Sampled Polynomials from Latent Space')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

        return polynomials

def interpolate_polynomials(model, poly1_coeffs, poly2_coeffs, steps=10):
    model.eval()
    with torch.no_grad():
        mu1, _ = model.encode(poly1_coeffs.unsqueeze(0))
        mu2, _ = model.encode(poly2_coeffs.unsqueeze(0))

        interpolated = []
        plt.figure(figsize=(12, 8))

        for alpha in torch.linspace(0, 1, steps):
            z = alpha * mu1 + (1 - alpha) * mu2
            coeffs = model.decode(z)
            poly = coefficients_to_polynomial(coeffs[0])
            interpolated.append(poly)

            x = np.linspace(-5, 5, 100)
            y = np.array([float(poly.subs('x', xi)) for xi in x])
            plt.plot(x, y, label=f'α={alpha:.2f}')

        plt.title('Interpolated Polynomials')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

        return interpolated

def save_dataset(dataset, filename):
    data = {
        'polynomials': dataset.polynomials,
        'distances': dataset.distances,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nDataset saved to {filename}")

def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    dataset = PolynomialDataset(0, skip_init=True)
    dataset.polynomials = data['polynomials']
    dataset.distances = data['distances']
    print(f"\nLoaded dataset from {filename}")
    return dataset

def main():
    # Updated parameters
    num_polynomials = 2000  # Increased dataset size
    latent_dim = 8  # Reduced latent dimension
    num_epochs = 200
    batch_size = 64

    # Rest of the main function remains the same
    dataset_file = 'Datasets/3th_degree_polynomial_dataset.pkl'
    if os.path.exists(dataset_file):
        dataset = load_dataset(dataset_file)
    else:
        dataset = PolynomialDataset(num_polynomials)
        save_dataset(dataset, dataset_file)

    coeff_dataset = CoefficientDataset(dataset.polynomials)
    train_loader = DataLoader(coeff_dataset, batch_size=batch_size, shuffle=True)

    model = PolynomialVAE(latent_dim=latent_dim)
    train_vae(model, train_loader, num_epochs=num_epochs)

    save_model(model, 'Model_Library/3th_degree_polynomial_vae_model.pt')
    plot_reconstruction(model, coeff_dataset)
    sample_from_latent(model)

if __name__ == "__main__":
    main()


