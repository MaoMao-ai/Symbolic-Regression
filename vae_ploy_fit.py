import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sympy as sp
import random
import matplotlib.pyplot as plt
from VAE import PolynomialVAE, load_model, coefficients_to_polynomial


def target_function():
    """Randomly generates a target 6th-degree polynomial."""
    x = sp.Symbol('x')
    terms = [round(random.uniform(-1, 1), 3) * x ** degree for degree in range(7)]
    return sum(terms)


def sample_target_data(target_poly, num_samples=200, domain=(-1, 1)):
    """Generates sample data points for the target polynomial."""
    x = sp.Symbol('x')
    f_func = sp.lambdify(x, target_poly, 'numpy')
    x_samples = np.linspace(domain[0], domain[1], num_samples)
    y_samples = np.array([f_func(xi) for xi in x_samples])
    return torch.tensor(x_samples, dtype=torch.float32), torch.tensor(y_samples, dtype=torch.float32)


def initialize_latent_vector(vae_model):
    """Initializes the latent vector with controlled randomness for better stability."""
    latent_vector = torch.randn(1, vae_model.latent_dim, dtype=torch.float32) * 0.5
    latent_vector.requires_grad_(True)
    return latent_vector


def compute_loss(vae_model, latent_vector, x_samples, y_samples):
    """Computes the total loss, including MSE, gradient loss, second derivative loss, and L2 regularization."""
    pred_coeffs = vae_model.decode(latent_vector).squeeze(0)

    # Compute predicted y-values using the polynomial coefficients
    X_vander = torch.vander(x_samples, N=7, increasing=True)
    pred_y_samples = X_vander @ pred_coeffs
    mse_loss = F.mse_loss(pred_y_samples, y_samples)

    # L2 regularization to control latent vector magnitude
    l2_reg = 0.01 * torch.norm(latent_vector, p=2)


    # Compute the total loss
    total_loss = mse_loss + l2_reg
    return total_loss


def optimize_latent_vector(vae_model, x_samples, y_samples, num_steps=200):
    """Optimizes the latent vector to minimize loss and match the target function."""
    latent_vector = initialize_latent_vector(vae_model)
    optimizer = torch.optim.Adam([latent_vector], lr=0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    def closure():
        optimizer.zero_grad()
        loss = compute_loss(vae_model, latent_vector, x_samples, y_samples)
        loss.backward()
        return loss

    for step in range(num_steps):
        # Reduce learning rate gradually for better convergence
        if step >= num_steps - 70 and optimizer.param_groups[0]['lr'] > 0.0001:
            optimizer.param_groups[0]['lr'] *= 0.9

        # Switch to LBFGS optimizer for final fine-tuning
        if step == num_steps - 50:
            print("\n🔄 Switching optimizer to LBFGS for fine-tuning!")
            optimizer = torch.optim.LBFGS([latent_vector], lr=0.1, max_iter=20)

        loss = optimizer.step(closure)
        latent_vector.data = torch.clamp(latent_vector.data, -3, 3)

        grad_norm = latent_vector.grad.norm().item()
        print(f"Step {step}: Loss = {loss.item():.4f}, Grad Norm = {grad_norm:.6f}")

        if grad_norm is None or grad_norm < 1e-6:
            print("⚠️ Warning: Gradient too small, stopping optimization early!")
            break
        scheduler.step()

    return latent_vector.detach()


def main():
    """Main function to load the model, generate a target function, optimize the latent vector, and visualize results."""
    vae_model = load_model("vae_model_improved.pt")
    vae_model.eval()

    # Generate a random target polynomial
    target_poly = target_function()
    print(f"\nTarget Function: {target_poly}")

    # Sample data points from the target polynomial
    x_samples, y_samples = sample_target_data(target_poly, num_samples=200)

    # Initialize the latent vector and obtain the initial polynomial
    init_latent_vector = initialize_latent_vector(vae_model)
    init_coeffs = vae_model.decode(init_latent_vector)
    init_poly = coefficients_to_polynomial(init_coeffs[0])

    # Optimize the latent vector to fit the target polynomial
    optimal_latent_vector = optimize_latent_vector(vae_model, x_samples, y_samples)
    optimal_coeffs = vae_model.decode(optimal_latent_vector)
    optimal_poly = coefficients_to_polynomial(optimal_coeffs[0])
    print(f"\nOptimized Polynomial: {optimal_poly}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    x = np.linspace(-1, 1, 100)
    y_target = np.array([float(target_poly.subs('x', xi)) for xi in x])
    y_optimal = np.array([float(optimal_poly.subs('x', xi)) for xi in x])

    plt.plot(x, y_target, 'b-', label="Target Function", linewidth=2)
    y_init = np.array([float(init_poly.subs('x', xi)) for xi in x])
    plt.plot(x, y_init, 'g--', label="Initial Function", linewidth=2)
    plt.plot(x, y_optimal, 'r-.', label="Optimized Function", linewidth=2)
    plt.scatter(x_samples, y_samples, color='black', label="Sample Points", zorder=3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Target vs Optimized Function")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()