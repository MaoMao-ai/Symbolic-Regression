import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.integrate import dblquad
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

x_sym, y_sym = sp.Symbol('x'), sp.Symbol('y')

# ----------------------------
# 函数生成与L2距离计算
# ----------------------------
def generate_function():
    a = random.uniform(-2, 2)
    b = random.uniform(-3, 3)
    c = random.uniform(-3, 3)
    d = random.uniform(-2, 2)
    return a, b, c, d, a * sp.sin(b * x_sym) * sp.cos(c * y_sym) + d

def compute_l2_distance(f, g, domain=(-5, 5)):
    f_func = sp.lambdify((x_sym, y_sym), f, 'numpy')
    g_func = sp.lambdify((x_sym, y_sym), g, 'numpy')
    
    def diff_squared(y, x):  # 注意dblquad的积分顺序是 y, x
        return (f_func(x, y) - g_func(x, y)) ** 2

    result, _ = dblquad(diff_squared, domain[0], domain[1],
                        lambda x: domain[0], lambda x: domain[1])
    return np.sqrt(result)
def compute_distances_chunk(args):
    chunk_idx, pairs_chunk, functions = args
    results = []

    pbar = tqdm(pairs_chunk,
                desc=f'Chunk {chunk_idx + 1}',
                position=chunk_idx + 1,
                leave=False)

    for i, j in pbar:
        try:
            dist = compute_l2_distance(functions[i], functions[j])
            results.append((i, j, dist))
        except Exception as e:
            print(f"\nError computing distance for pair ({i}, {j}): {e}")
            results.append((i, j, float('nan')))

    pbar.close()
    return results

def compute_distances(functions, num_processes=24):
    n = len(functions)
    distances = torch.zeros(n, n)

    pairs = list(combinations(range(n), 2))
    total_pairs = len(pairs)

    print(f"\nTotal number of pairs to compute: {total_pairs}")

    chunk_size = math.ceil(total_pairs / num_processes)
    pair_chunks = [pairs[i:i + chunk_size] for i in range(0, total_pairs, chunk_size)]

    chunk_args = [
        (idx, chunk, functions)
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

class FunctionDataset(Dataset):
    def __init__(self, num_functions, num_threads=24, skip_init=False):
        self.functions = []
        self.distances = None

        if not skip_init:
            print("\nGenerating Functions...")
            for _ in tqdm(range(num_functions)):
                poly = generate_function()
                self.functions.append(poly)

            self.distances = compute_distances(self.functions, num_processes=num_threads)

    def __len__(self):
        return len(self.functions)

    def __getitem__(self, idx):
        return self.functions[idx], self.distances[idx]

class CoefficientDataset(Dataset):
    def __init__(self, functions):
        self.coefficients = []
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        print("Extracting coefficients...")
        for func in tqdm(functions):
            if isinstance(func, tuple):
                # 如果是从generate_function直接获得的数据
                a, b, c, d, _ = func
                coeffs = [a, b, c, d]
            else:
                try:
                    # 提取偏移项 d
                    d = float(func.subs({x: 0, y: 0}))
                    
                    # 使用网格采样点
                    x_points = np.linspace(-5, 5, 20)
                    y_points = np.linspace(-5, 5, 20)
                    X, Y = np.meshgrid(x_points, y_points)
                    
                    # 计算函数值
                    values = np.array([[float(func.subs({x: xi, y: yi})) 
                                      for xi in x_points] for yi in y_points])
                    
                    # 提取振幅 a
                    max_val = np.max(values)
                    min_val = np.min(values)
                    a = (max_val - min_val) / 2
                    
                    # 使用FFT分析频率 b 和 c
                    fft_x = np.fft.fft2(values)
                    freqs_x = np.fft.fftfreq(len(x_points))
                    freqs_y = np.fft.fftfreq(len(y_points))
                    
                    # 获取主要频率
                    peaks_x = np.argsort(np.abs(fft_x))[-2:]
                    peaks_y = np.argsort(np.abs(fft_x))[-2:]
                    b = abs(freqs_x[peaks_x[0]]) * 10
                    c = abs(freqs_y[peaks_y[0]]) * 10
                    
                    coeffs = [a, b, c, d]
                except Exception as e:
                    print(f"Error extracting coefficients: {e}")
                    coeffs = [
                        random.uniform(-2, 2),   # a
                        random.uniform(-3, 3),   # b
                        random.uniform(-3, 3),   # c
                        random.uniform(-2, 2)    # d
                    ]
            
            self.coefficients.append(coeffs)

        self.coefficients = torch.tensor(self.coefficients, dtype=torch.float32)
        
        # 修改参数范围
        self.ranges = {
            'a': (-2, 2),
            'b': (-3, 3),
            'c': (-3, 3),
            'd': (-2, 2)
        }
        
        # 归一化所有参数
        for i, (param, (min_val, max_val)) in enumerate(self.ranges.items()):
            self.coefficients[:, i] = torch.tanh(self.coefficients[:, i] / max_val)

        global COEFF_STATS
        COEFF_STATS = self.ranges

    def __len__(self):
        return len(self.coefficients)

    def __getitem__(self, idx):
        return self.coefficients[idx]

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm1 = nn.BatchNorm1d(out_features)
        self.norm2 = nn.BatchNorm1d(out_features)
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        out += self.shortcut(identity)
        return F.relu(out)

class FunctionVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器 (Encoder) - 输入维度改为4
        self.encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        # 解码器 (Decoder) - 输出维度改为4
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 4),  # 输出4个参数
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=0.1, epoch=0):
    # 动态权重
    freq_weight = min(2.0 + epoch/100, 4.0)  # 随着训练进行增加频率项的权重
    
    # MSE损失
    recon_loss_a = F.mse_loss(recon_x[:, 0], x[:, 0])
    recon_loss_b = F.mse_loss(recon_x[:, 1], x[:, 1]) * freq_weight
    recon_loss_c = F.mse_loss(recon_x[:, 2], x[:, 2]) * freq_weight
    recon_loss_d = F.mse_loss(recon_x[:, 3], x[:, 3])
    
    # L1损失
    l1_loss = 0.1 * (torch.abs(recon_x - x).mean())
    
    # 周期性损失（针对频率参数）
    periodic_loss = 0.1 * (torch.sin(recon_x[:, 1] - x[:, 1]).abs().mean() + 
                          torch.sin(recon_x[:, 2] - x[:, 2]).abs().mean())
    
    # 总重建损失
    recon_loss = (recon_loss_a + recon_loss_b + recon_loss_c + 
                  recon_loss_d + l1_loss + periodic_loss)
    
    # 自适应beta
    beta_t = beta * (1.0 + epoch/200)  # 随训练增加KL项的权重
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta_t * kl_loss

def train_vae(model, train_loader, num_epochs=800, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 使用OneCycleLR调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    best_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}')
            
def save_model(model, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'latent_dim': model.latent_dim
    }, filename)
    print(f"\nModel saved to {filename}")

def load_model(filename):
    checkpoint = torch.load(filename)
    model = FunctionVAE(latent_dim=checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nModel loaded from {filename}")
    return model

def plot_reconstruction(model, dataset, num_samples=5):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            orig_coeffs = dataset[i].to(device)
            recon_coeffs, _, _ = model(orig_coeffs.unsqueeze(0))

            orig_poly = coefficients_to_function(orig_coeffs)
            recon_poly = coefficients_to_function(recon_coeffs[0])

            # 绘制原始函数
            plot_2d_function(orig_poly, ax, f'Sample {i+1} (Original)', 'b', '-')
            # 绘制重建函数
            plot_2d_function(recon_poly, ax, f'Sample {i+1} (Reconstructed)', 'r', '--')

            distance = compute_l2_distance(orig_poly, recon_poly)
            ax.set_title(f'Sample {i+1} (L2 Distance: {distance:.4f})')

            print(f"\nSample {i+1}:")
            print(f"Original: {orig_poly}")
            print(f"Reconstructed: {recon_poly}")

        plt.tight_layout()
        plt.show()

COEFF_STATS = {
    'a': (-2, 2),
    'b': (-3, 3),
    'c': (-3, 3),
    'd': (-2, 2)
}

def coefficients_to_function(coeffs):
    if isinstance(coeffs, torch.Tensor):
        coeffs = coeffs.detach().cpu().numpy()
    a = COEFF_STATS['a'][1] * np.arctanh(np.clip(coeffs[0], -0.95, 0.95))
    b = COEFF_STATS['b'][1] * np.arctanh(np.clip(coeffs[1], -0.95, 0.95))
    c = COEFF_STATS['c'][1] * np.arctanh(np.clip(coeffs[2], -0.95, 0.95))
    d = COEFF_STATS['d'][1] * np.arctanh(np.clip(coeffs[3], -0.95, 0.95))
    return a * sp.sin(b * x_sym) * sp.cos(c * y_sym) + d

def plot_2d_function(func, ax, title, color='b', linestyle='-'):
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda x, y: float(func.subs({x_sym: x, y_sym: y})))(X, Y)
    ax.contour(X, Y, Z, levels=20, colors=color, linestyles=linestyle)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)

def interpolate_polynomials(model, poly1_coeffs, poly2_coeffs, steps=10):
    model.eval()
    with torch.no_grad():
        mu1, _ = model.encode(poly1_coeffs.unsqueeze(0))
        mu2, _ = model.encode(poly2_coeffs.unsqueeze(0))

        fig, axes = plt.subplots(1, steps, figsize=(3 * steps, 4))

        for i, alpha in enumerate(torch.linspace(0, 1, steps)):
            z = alpha * mu1 + (1 - alpha) * mu2
            coeffs = model.decode(z)
            func = coefficients_to_function(coeffs[0])
            plot_2d_function(func, axes[i], f'α={alpha:.2f}', 'r')

        plt.tight_layout()
        plt.show()

# ----------------------------
# sample_from_latent 修复版
# ----------------------------
def sample_from_latent(model, num_samples=5):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]

        z = torch.randn(num_samples, model.latent_dim).to(device)
        coeffs = model.decode(z)

        for i in range(num_samples):
            func = coefficients_to_function(coeffs[i])
            plot_2d_function(func, axes[i], f'Sample {i + 1}', 'r')
            print(f"\nSample {i+1}:\nGenerated function: {func}")

        plt.tight_layout()
        plt.show()

def save_dataset(dataset, filename):
    data = {
        'functions': dataset.functions,
        'distances': dataset.distances,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nDataset saved to {filename}")

def load_dataset(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    dataset = FunctionDataset(0, skip_init=True)
    dataset.functions = data['functions']
    dataset.distances = data['distances']
    print(f"\nLoaded dataset from {filename}")
    return dataset

def main():
    # 初始化全局变量
    global COEFF_STATS
    COEFF_STATS = None

    num_functions = 3000  # 增加数据集大小
    latent_dim = 32     # 增加潜在空间维度
    num_epochs = 500    # 增加训练轮数
    batch_size = 128    # 调整批次大小

    # 创建或加载数据集
    dataset_file = 'Datasets/sin_x_cos_y_dataset.pkl'
    if os.path.exists(dataset_file):
        print(f"\nFound existing dataset at {dataset_file}")
        dataset = load_dataset(dataset_file)
    else:
        print("\nCreating new dataset...")
        functions = []
        for _ in tqdm(range(num_functions)):
            a, b, c, d, func = generate_function()
            functions.append((a, b, c, d, func))
        dataset = FunctionDataset(0, skip_init=True)
        dataset.functions = functions
        save_dataset(dataset, dataset_file)

    # 创建系数数据集
    coeff_dataset = CoefficientDataset(dataset.functions)

    train_loader = DataLoader(
        coeff_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = FunctionVAE(latent_dim=latent_dim)
    train_vae(model, train_loader, num_epochs=num_epochs)

    save_model(model, 'Model_Library/sin_x_cos_y_vae_model.pt')
    plot_reconstruction(model, coeff_dataset, num_samples=5)
    sample_from_latent(model, num_samples=5)

if __name__ == "__main__":
    main()