
def compute_loss(vae_model, latent_vector, x_samples, y_samples, target_poly):
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


def optimize_latent_vector(vae_model, x_samples, y_samples, target_poly, num_steps=200):
    """Optimizes the latent vector to minimize loss and match the target function."""
    latent_vector = initialize_latent_vector(vae_model)
    optimizer = torch.optim.Adam([latent_vector], lr=0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    def closure():
        optimizer.zero_grad()
        loss = compute_loss(vae_model, latent_vector, x_samples, y_samples, target_poly)
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