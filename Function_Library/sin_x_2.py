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


def generate_function():
    """ 生成 a * sin(b * x^2) + c 形式的函数，使用更精确的参数生成 """
    x = sp.Symbol('x')
    # 直接返回参数和函数，避免信息丢失
    a = random.uniform(-2, 2)
    b = random.uniform(-3, 3)
    c = random.uniform(-2, 2)
    return a, b, c, a * sp.sin(b * x**2) + c

def compute_l2_distance(f, g, domain=(-5, 5)):
    x = sp.Symbol('x')
    f_func = sp.lambdify(x, f, 'numpy')
    g_func = sp.lambdify(x, g, 'numpy')

    diff_squared = lambda x: (f_func(x) - g_func(x))**2
    distance, _ = quad(diff_squared, domain[0], domain[1])
    return np.sqrt(distance)

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

# ----------------------------
# 数据集相关部分修改
# ----------------------------
class CoefficientDataset(Dataset):
    def __init__(self, functions):
        self.coefficients = []
        x = sp.Symbol('x')

        print("Extracting coefficients...")
        for func in tqdm(functions):
            if isinstance(func, tuple):
                # 如果是从generate_function直接获得的数据
                a, b, c, _ = func
                coeffs = [a, b, c]
            else:
                # 如果是从文件加载的数据，需要提取系数
                try:
                    # 提取常数项 c
                    c = float(func.subs(x, 0))
                    
                    # 使用更密集的采样点
                    sample_points = np.linspace(-5, 5, 50)
                    values = [float(func.subs(x, p)) for p in sample_points]
                    derivatives = [float(sp.diff(func, x).subs(x, p)) for p in sample_points]
                    
                    # 提取振幅 a
                    max_val = max(values)
                    min_val = min(values)
                    a = (max_val - min_val) / 2
                    
                    # 提取频率 b
                    # 使用FFT来估计频率
                    fft = np.fft.fft(values)
                    freqs = np.fft.fftfreq(len(values))
                    peak_idx = np.argmax(np.abs(fft[1:])) + 1
                    b = freqs[peak_idx] * 20  # 缩放因子
                    
                    # 使用导数信息验证和调整b
                    max_deriv = max(abs(d) for d in derivatives)
                    b_from_deriv = max_deriv / (2 * abs(a)) if abs(a) > 1e-6 else 3.0
                    
                    # 综合两种方法
                    b = (b + b_from_deriv) / 2
                    
                    coeffs = [a, b, c]
                except Exception as e:
                    print(f"Error extracting coefficients: {e}")
                    coeffs = [random.uniform(-2, 2), random.uniform(-3, 3), random.uniform(-2, 2)]
            
            self.coefficients.append(coeffs)

        self.coefficients = torch.tensor(self.coefficients, dtype=torch.float32)
        
        # 使用改进的归一化方法
        self.a_range = (-2, 2)
        self.b_range = (-3, 3)
        self.c_range = (-2, 2)
        
        # 使用双曲正切归一化
        self.coefficients[:, 0] = torch.tanh(self.coefficients[:, 0] / self.a_range[1])
        self.coefficients[:, 1] = torch.tanh(self.coefficients[:, 1] / self.b_range[1])
        self.coefficients[:, 2] = torch.tanh(self.coefficients[:, 2] / self.c_range[1])

        global COEFF_STATS
        COEFF_STATS = {
            'a_range': self.a_range,
            'b_range': self.b_range,
            'c_range': self.c_range
        }

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

class FunctionVAE(nn.Module):
    def __init__(self, latent_dim=32):  # 增加潜在空间维度
        super().__init__()
        self.latent_dim = latent_dim

        # 编码器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(3, 128),
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

        # 均值和方差
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)

        # 解码器 (Decoder)
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
            
            nn.Linear(128, 3),
            nn.Tanh()  # 确保输出在合理范围内
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

def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    # 分别计算每个参数的重建损失
    recon_loss_a = F.mse_loss(recon_x[:, 0], x[:, 0])
    recon_loss_b = F.mse_loss(recon_x[:, 1], x[:, 1]) * 2.0  # 增加b的权重
    recon_loss_c = F.mse_loss(recon_x[:, 2], x[:, 2])
    
    # 添加L1损失以促进稀疏性
    l1_loss = 0.1 * (torch.abs(recon_x - x).mean())
    
    # 总重建损失
    recon_loss = recon_loss_a + recon_loss_b + recon_loss_c + l1_loss
    
    # KL散度损失
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

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

            orig_poly = coefficients_to_function(orig_coeffs)
            recon_poly = coefficients_to_function(recon_coeffs[0])

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

# ----------------------------
# 反归一化函数修改
# ----------------------------
def coefficients_to_function(coeffs):
    """
    将归一化的系数转换回函数形式
    """
    x = sp.Symbol('x')
    
    if isinstance(coeffs, torch.Tensor):
        coeffs = coeffs.detach().cpu().numpy()
    
    # 使用反双曲正切进行反归一化
    a = float(COEFF_STATS['a_range'][1] * np.arctanh(np.clip(coeffs[0], -0.99, 0.99)))
    b = float(COEFF_STATS['b_range'][1] * np.arctanh(np.clip(coeffs[1], -0.99, 0.99)))
    c = float(COEFF_STATS['c_range'][1] * np.arctanh(np.clip(coeffs[2], -0.99, 0.99)))
    
    return a * sp.sin(b * x**2) + c


def sample_from_latent(model, num_samples=5):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)  # 直接在正确的设备上创建
        coeffs = model.decode(z)

        plt.figure(figsize=(12, 8))

        polynomials = []
        for i in range(num_samples):
            poly = coefficients_to_function(coeffs[i])
            polynomials.append(poly)
            x = np.linspace(-5, 5, 200)
            y = np.array([float(poly.subs('x', xi)) for xi in x])
            plt.plot(x, y, label=f'Sample {i+1}')
            print(f"\nSample {i+1}: {poly}")

        plt.title('Sampled Functions from Latent Space')
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
            poly = coefficients_to_function(coeffs[0])
            interpolated.append(poly)

            x = np.linspace(-5, 5, 100)
            y = np.array([float(poly.subs('x', xi)) for xi in x])
            plt.plot(x, y, label=f'α={alpha:.2f}')

        plt.title('Interpolated Functions')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

        return interpolated

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

    num_functions = 5000  # 增加数据集大小以获得更多样的函数
    latent_dim = 32  # 增加潜在空间维度以容纳更多变化
    num_epochs = 800
    batch_size = 256

    # 创建或加载数据集
    dataset_file = 'Datasets/sin(x^2)_dataset.pkl'
    if os.path.exists(dataset_file):
        print(f"\nFound existing dataset at {dataset_file}")
        dataset = load_dataset(dataset_file)
    else:
        print("\nCreating new dataset...")
        functions = []
        for _ in tqdm(range(num_functions)):
            a, b, c, func = generate_function()
            functions.append((a, b, c, func))
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

    save_model(model, 'Model_Library/sin(x^2)_vae_model.pt')
    plot_reconstruction(model, coeff_dataset, num_samples=5)
    sample_from_latent(model, num_samples=5)

if __name__ == "__main__":
    main()