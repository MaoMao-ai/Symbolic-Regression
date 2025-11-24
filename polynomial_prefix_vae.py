import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from Prefix_func_re import TOKEN_TO_ID, ID_TO_TOKEN, infix_to_prefix, encode_expression

def prefix_to_infix(prefix_tokens):
    """将前缀表达式转换为中序表达式"""
    operators = {'+', '-', '*', '/', '^'}
    stack = []
    
    # 从右向左扫描前缀表达式
    for token in reversed(prefix_tokens.split()):
        if token in operators:
            # 操作符需要两个操作数
            if len(stack) < 2:
                return "Invalid Expression"
            a = stack.pop()
            b = stack.pop()
            # 根据操作符构建中序表达式
            if token in {'+', '-'}:
                expr = f"({a} {token} {b})"
            elif token == '*':
                expr = f"{a} * {b}"
            elif token == '/':
                expr = f"{a} / {b}"
            else:  # power
                expr = f"{a}^{b}"
            stack.append(expr)
        else:
            # 操作数直接入栈
            stack.append(token)
    
    return stack[0] if stack else "Invalid Expression"

def plot_polynomials(polynomials, x_range=(-5, 5)):
    """绘制多个多项式的对比图"""
    try:
        import numpy as np
        from sympy import sympify, symbols, lambdify
        
        x = symbols('x')
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        
        # 创建一个大图
        plt.figure(figsize=(15, 10))
        
        # 设置颜色和样式
        colors = ['b', 'r', 'g', 'm', 'c']
        styles = ['-', '--', ':', '-.', '-']
        
        # 计算所有函数的最大最小值以设置y轴范围
        all_y_vals = []
        
        # 绘制每个多项式
        for i, (orig_expr, recon_expr) in enumerate(polynomials):
            try:
                # 将表达式转换为sympy表达式
                original = sympify(orig_expr.replace('^', '**'))
                reconstructed = sympify(recon_expr.replace('^', '**'))
                
                # 创建lambda函数以便快速计算
                f_original = lambdify(x, original)
                f_reconstructed = lambdify(x, reconstructed)
                
                # 计算y值
                y_original = f_original(x_vals)
                y_reconstructed = f_reconstructed(x_vals)
                
                # 添加到所有y值列表
                all_y_vals.extend(y_original)
                all_y_vals.extend(y_reconstructed)
                
                # 绘制函数
                plt.plot(x_vals, y_original, 
                        color=colors[i % len(colors)], 
                        linestyle=styles[0], 
                        label=f'Original {i+1}: {orig_expr}', 
                        alpha=0.7)
                plt.plot(x_vals, y_reconstructed, 
                        color=colors[i % len(colors)], 
                        linestyle=styles[1], 
                        label=f'Reconstructed {i+1}: {recon_expr}', 
                        alpha=0.7)
                
            except Exception as e:
                print(f"Error plotting polynomial {i+1}: {e}")
                continue
        
        # 设置图形属性
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Original vs Reconstructed Polynomials')
        plt.xlabel('x')
        plt.ylabel('y')
        
        # 添加零线
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # 调整显示范围
        y_min = min(all_y_vals)
        y_max = max(all_y_vals)
        margin = (y_max - y_min) * 0.1
        plt.ylim(y_min - margin, y_max + margin)
        
        # 调整布局以适应图例
        plt.tight_layout()
        
        plt.show()
        return True
            
    except Exception as e:
        print(f"Error in plot_polynomials: {e}")
        return False

class ImprovedPrefixVAE(nn.Module):
    def __init__(self, vocab_size=7, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, 64),  # 7 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),  # 64 -> 32
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),  # 32 -> 16
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 32)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),  # 32 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),  # 64 -> 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),  # 128 -> 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)  # 256 -> 7
        )

    def encode(self, x):
        x = x.long()  # 确保输入是LongTensor类型
        x = F.one_hot(x, num_classes=30).float()  # Convert to one-hot
        # 将30维度映射到7维度
        x = x.mean(dim=1)  # Average over sequence length
        x = x @ torch.eye(30, 7).to(x.device)  # 线性投影到7维
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        return mu  # Using deterministic latent space

    def decode(self, z, seq_length):
        h = self.decoder_input(z)
        output = self.decoder(h)
        # 将7维度映射回30维度
        output = output @ torch.eye(7, 30).to(output.device)  # 线性投影到30维
        return output.unsqueeze(1).repeat(1, seq_length, 1)  # Repeat for sequence length

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x.shape[1]), mu, logvar

def load_improved_model(model_path="vae_model_improved.pt"):
    """加载改进版VAE模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ImprovedPrefixVAE(latent_dim=checkpoint['latent_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def process_polynomial(polynomial_expr, model, device):
    """处理多项式表达式，通过VAE进行编码和解码"""
    # 转换为prefix表达式
    prefix_seq = infix_to_prefix(polynomial_expr)
    print(f"Prefix expression: {' '.join(prefix_seq)}")
    
    # 编码为tensor
    encoded_input = encode_expression(prefix_seq).to(device)
    
    # 通过VAE进行编码和解码
    with torch.no_grad():
        # 编码到潜在空间
        mu, logvar = model.encode(encoded_input.unsqueeze(0))
        z = model.reparameterize(mu, logvar)
        
        # 从潜在空间解码
        output = model.decode(z, seq_length=20)
        
        # 使用softmax获取token概率
        probs = F.softmax(output / 0.6, dim=-1)
        tokens = probs.argmax(dim=-1).cpu().numpy()[0]
        
        # 转换回表达式
        reconstructed_prefix = ' '.join([ID_TO_TOKEN.get(t, '?') for t in tokens])
        # 转换为中序表达式
        reconstructed_infix = prefix_to_infix(reconstructed_prefix)
        
    return {
        'original_expr': polynomial_expr,
        'prefix_expr': ' '.join(prefix_seq),
        'reconstructed_prefix': reconstructed_prefix,
        'reconstructed_infix': reconstructed_infix,
        'latent_vector': z.cpu().numpy()
    }

def main():
    # 加载模型
    print("Loading model...")
    model, device = load_improved_model()
    print("Model loaded successfully!")
    
    # 测试多项式
    test_polynomials = [
        "x^2 + 3*x - 5",
        "2*x^3 - 4*x^2 + x - 1",
        "x^4 - 2*x^2 + 1",
        "x^3 + 2*x^2 - x + 1",
        "x^5 - 3*x^3 + 2*x"
    ]
    
    print("\nProcessing polynomials...")
    results = []
    
    for poly in test_polynomials:
        print("\n" + "="*50)
        print(f"Original polynomial: {poly}")
        
        result = process_polynomial(poly, model, device)
        results.append((poly, result['reconstructed_infix']))
        
        print(f"Prefix form: {result['prefix_expr']}")
        print(f"Reconstructed prefix: {result['reconstructed_prefix']}")
        print(f"Reconstructed infix: {result['reconstructed_infix']}")
        print(f"Latent vector shape: {result['latent_vector'].shape}")
        print("="*50)
    
    # 绘制所有多项式的对比图
    print("\nGenerating comparison plot...")
    plot_polynomials(results)

if __name__ == "__main__":
    main() 