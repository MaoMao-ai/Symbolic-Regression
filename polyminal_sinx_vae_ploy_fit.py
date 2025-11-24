import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sympy as sp
import random
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from Polyminal_Sinx_VAE import Polynomial_Sinx_VAE, load_model, coefficients_to_polynomial
def target_function():
    """Generate a random target polynomial (including sine term)"""
    x = sp.Symbol('x')
    # Reduce coefficient range to avoid too large function values
    terms = [round(random.uniform(-1, 1), 3) * x ** degree for degree in range(7)]
    # Add sine term
    sin_coef = round(random.uniform(-1, 1), 3)
    terms.append(sin_coef * sp.sin(x))
    
    return sum(terms)

def sample_target_data(target_poly, num_samples=500, domain=(-10, 10)):
    """Sample data points from target polynomial"""
    x = sp.Symbol('x')
    f_func = sp.lambdify(x, target_poly, 'numpy')
    x_samples = np.linspace(domain[0], domain[1], num_samples)
    y_samples = np.array([f_func(xi) for xi in x_samples])
    return torch.tensor(x_samples, dtype=torch.float32), torch.tensor(y_samples, dtype=torch.float32)

def initialize_latent_vector(vae_model):
    """Initialize latent vector with controlled randomness for better stability"""
    latent_vector = torch.randn(1, vae_model.latent_dim, dtype=torch.float32) * 0.5
    latent_vector.requires_grad_(True)
    return latent_vector

def compute_loss(vae_model, latent_vector, x_samples, y_samples, target_poly):
    """Compute total loss including MSE, regularization and smoothness constraints"""
    pred_coeffs = vae_model.decode(latent_vector).squeeze(0)
    
    # Separate polynomial coefficients and sine coefficient
    poly_coeffs = pred_coeffs[:-1]  # First 7 coefficients for polynomial
    sin_coef = pred_coeffs[-1]      # Last coefficient for sine term
    
    # Calculate polynomial part predictions
    x_np = x_samples.cpu().numpy()
    vander_matrix = np.vander(x_np, N=7, increasing=True)
    vander_tensor = torch.from_numpy(vander_matrix).float().to(x_samples.device)
    poly_pred = vander_tensor @ poly_coeffs
    
    # Calculate sine part predictions
    sin_pred = sin_coef * torch.sin(x_samples)
    
    # Combine predictions
    pred_y_samples = poly_pred + sin_pred
    
    # MSE loss
    mse_loss = F.mse_loss(pred_y_samples, y_samples)
    
    # L2 regularization to control latent vector magnitude
    l2_reg = 0.001 * torch.norm(latent_vector, p=2)  # Reduced regularization weight
    
    # Smoothness constraint
    smoothness_loss = 0.0001 * torch.norm(torch.diff(pred_y_samples), p=2)  # Reduced smoothness weight
    
    # Total loss
    total_loss = mse_loss + l2_reg + smoothness_loss
    return total_loss

def optimize_latent_vector(vae_model, x_samples, y_samples, target_poly, num_steps=2000):
    """Optimize latent vector to match target function"""
    latent_vector = initialize_latent_vector(vae_model)
    optimizer = torch.optim.Adam([latent_vector], lr=0.001)  # Increased initial learning rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # Adjusted learning rate decay

    def closure():
        optimizer.zero_grad()
        loss = compute_loss(vae_model, latent_vector, x_samples, y_samples, target_poly)
        loss.backward()
        return loss

    best_loss = float('inf')
    best_vector = latent_vector.clone()

    for step in range(num_steps):
        # Reduce learning rate gradually for better convergence
        if step >= num_steps - 100 and optimizer.param_groups[0]['lr'] > 0.0001:
            optimizer.param_groups[0]['lr'] *= 0.95

        # Switch to LBFGS optimizer for final fine-tuning
        if step == num_steps - 50:
            print("\n🔄 Switching optimizer to LBFGS for fine-tuning!")
            optimizer = torch.optim.LBFGS([latent_vector], lr=0.1, max_iter=20)

        loss = optimizer.step(closure)
        
        # Save best results
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_vector = latent_vector.clone()

        latent_vector.data = torch.clamp(latent_vector.data, -2, 2)  # Reduced clamp range

        if step % 100 == 0:  # Reduced print frequency
            grad_norm = latent_vector.grad.norm().item()
            print(f"Step {step}: Loss = {loss.item():.4f}, Grad Norm = {grad_norm:.6f}")

        if step < num_steps - 50:  # Only update scheduler before switching to LBFGS
            scheduler.step()

    return best_vector.detach()  # Return best result

def plot_results(x_samples, y_samples, target_poly, init_poly, optimal_poly):
    """Plot comparison results"""
    plt.figure(figsize=(12, 6))
    x = np.linspace(-10, 10, 500)  # Adjusted plot range
    
    # Calculate function values
    y_target = np.array([float(target_poly.subs('x', xi)) for xi in x])
    y_init = np.array([float(init_poly.subs('x', xi)) for xi in x])
    y_optimal = np.array([float(optimal_poly.subs('x', xi)) for xi in x])
    
    # Plot functions
    plt.plot(x, y_target, 'b-', label="Target Function", linewidth=2)
    plt.plot(x, y_init, 'g--', label="Initial Function", linewidth=2, alpha=0.8)
    plt.plot(x, y_optimal, 'r-.', label="Optimized Function", linewidth=2)
    plt.scatter(x_samples.numpy(), y_samples.numpy(), color='black', 
               label="Sample Points", zorder=3, alpha=0.6, s=20)
    
    # Set plot properties
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Target vs Initial vs Optimized Function")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add function expressions
    plt.text(0.02, 0.98, f'Target: {sp.simplify(target_poly)}', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
    plt.text(0.02, 0.94, f'Initial: {sp.simplify(init_poly)}',
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
    plt.text(0.02, 0.90, f'Optimized: {sp.simplify(optimal_poly)}', 
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
    
    # Auto-adjust y-axis range to fit all curves
    all_y = np.concatenate([y_target, y_init, y_optimal])
    y_min, y_max = np.min(all_y), np.max(all_y)
    margin = (y_max - y_min) * 0.1  # Add 10% margin
    plt.ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function: Load model, generate target function, optimize latent vector and visualize results"""
    print("Loading model...")
    vae_model, device = load_model("polynomial_sinx_vae_model.pth")
    vae_model.eval()

    # Generate random target polynomial
    target_poly = target_function()
    print(f"\nTarget function: {target_poly}")

    # Sample data points from target polynomial
    x_samples, y_samples = sample_target_data(target_poly)

    # Initialize latent vector and get initial polynomial
    init_latent_vector = initialize_latent_vector(vae_model)
    init_coeffs = vae_model.decode(init_latent_vector)
    init_poly = coefficients_to_polynomial(init_coeffs[0].detach())
    print(f"\nInitial function: {init_poly}")

    print("\nStarting optimization...")
    # Optimize latent vector to fit target polynomial
    optimal_latent_vector = optimize_latent_vector(vae_model, x_samples, y_samples, target_poly)
    optimal_coeffs = vae_model.decode(optimal_latent_vector)
    optimal_poly = coefficients_to_polynomial(optimal_coeffs[0].detach())
    
    print(f"\nOptimized polynomial: {optimal_poly}")

    # Plot results
    plot_results(x_samples, y_samples, target_poly, init_poly, optimal_poly)
    
    # Print function comparison at the end
    print("\n" + "="*50)
    print("Function Comparison:")
    print("-"*50)
    print(f"Target Function:     {sp.simplify(target_poly)}")
    print(f"Initial Function:     {sp.simplify(init_poly)}")
    print(f"Optimized Function:   {sp.simplify(optimal_poly)}")
    print("="*50)

if __name__ == "__main__":
    main() 