import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from difflib import SequenceMatcher
import os

# Define token mappings
TOKEN_TO_ID = {
    '+': 1, '-': 2, '*': 3, '/': 4, '^': 5,  
    'x': 6, 'y': 7, 'z': 8,  
    '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18,  
    '√': 19, 'log': 20, 'sin': 21, 'cos': 22,  
    'π': 23, 'e': 24  
}
ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}

# Function to generate random expressions
def generate_expression():
    ops = ['+', '-', '*', '/', '^']
    variables = ['x', 'y', 'z']
    numbers = [str(i) for i in range(10)]
    functions = ['sin', 'cos', 'log', '√']
    constants = ['π', 'e']

    expr = []
    length = random.randint(4, 8)
    for _ in range(length):
        choice = random.choice(['var', 'num', 'op', 'func', 'const'])
        if choice == 'var':
            expr.append(random.choice(variables))
        elif choice == 'num':
            expr.append(random.choice(numbers))
        elif choice == 'op' and len(expr) >= 2:
            expr.insert(random.randint(0, len(expr) - 1), random.choice(ops))
        elif choice == 'func' and len(expr) >= 1:
            expr.insert(0, random.choice(functions))
        elif choice == 'const':
            expr.append(random.choice(constants))
    
    return ' '.join(expr)

# Generate 90,000 training and 10,000 validation expressions
train_expressions = [generate_expression() for _ in range(90000)]
val_expressions = [generate_expression() for _ in range(10000)]

# Function to encode expressions
def encode_expression(expression, max_length=20):
    tokens = expression.split()
    encoded = [TOKEN_TO_ID.get(token, 0) for token in tokens]
    return torch.tensor(encoded + [0] * (max_length - len(encoded)), dtype=torch.long)

# Dataset class
class ExpressionDataset(Dataset):
    def __init__(self, expressions, max_length=20):
        self.data = [encode_expression(expr, max_length) for expr in expressions]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# VAE Model
class PrefixVAE(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=64, hidden_dim=128, latent_dim=64, max_length=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.encoder(x)
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc_mu(h_n), self.fc_var(h_n)

    def reparameterize(self, mu, logvar):
        return mu  # Using deterministic latent space

    def decode(self, z, seq_length):
        h = self.decoder_input(z).unsqueeze(1).repeat(1, seq_length, 1)
        h, _ = self.decoder(h)
        output = self.output_layer(h)
        return output[:, :seq_length]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_x = self.decode(z, seq_length=x.shape[1])
        return decoded_x, mu, logvar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from difflib import SequenceMatcher

# **Set up device for GPU acceleration**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_valid_prefix(expression):
    """Checks if the given prefix expression is valid."""
    stack = []
    operators = {'+', '-', '*', '/', '^'}

    for token in reversed(expression):
        if token in operators:
            if len(stack) < 2:
                return False  # Not enough operands
            stack.pop()
            stack.pop()
            stack.append("expr")  # Replace two operands with one valid expression
        else:
            stack.append(token)

    return len(stack) == 1

def prefix_structure_loss(pred_tokens):
    """Computes a loss based on structurally valid prefix expressions."""
    total_invalid = 0
    batch_size = len(pred_tokens)

    for tokens in pred_tokens:
        expression = [ID_TO_TOKEN.get(t, '?') for t in tokens]
        if not is_valid_prefix(expression):
            total_invalid += 1

    return torch.tensor(total_invalid / batch_size, dtype=torch.float32, device=device)  # Move loss to GPU

def vae_loss(recon_x, x, mu, logvar):
    """Computes VAE loss with KL divergence and structural constraints."""
    recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), ignore_index=0)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    pred_tokens = recon_x.argmax(dim=-1).cpu().numpy().tolist()
    structure_loss = prefix_structure_loss(pred_tokens)

    return recon_loss + 0.01 * kl_loss + 0.1 * structure_loss  # Final loss

def train_vae(model, train_loader, val_loader, num_epochs=1000, lr=1e-3, patience=20, save_path="vae_model.pth"):
    """Trains VAE using GPU if available and tracks accuracy per epoch."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    early_stop_counter = 0

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        kl_weight = min(1.0, epoch / (num_epochs * 0.7))  # KL annealing

        for batch in train_loader:
            batch = batch.to(device)  # **Move batch to GPU**
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch)
            
            # Compute the full loss
            loss = vae_loss(recon_x, batch, mu, logvar)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # **Run Validation and Compute Accuracy**
        model.eval()
        val_loss = validate_vae(model, val_loader)
        accuracy = reconstruction_accuracy(model, val_loader)  # Compute accuracy
        model.train()  # Return to train mode

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved at epoch {epoch+1} with Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"❌ No improvement for {early_stop_counter} epochs.")

        if early_stop_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}!")
            break


def validate_vae(model, val_loader):
    """Validation step using GPU."""
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)  # **Move to GPU**
            recon_x, mu, logvar = model(batch)
            kl_loss = -0.0005 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
            recon_loss = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), batch.view(-1), ignore_index=0)
            loss = recon_loss + kl_loss
            total_loss += loss.item()

    return total_loss / len(val_loader)

def reconstruction_accuracy(model, val_loader):
    """Computes reconstruction accuracy using GPU."""
    total_accuracy = 0
    num_samples = len(val_loader.dataset)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            mu, logvar = model.encode(batch)
            z = model.reparameterize(mu, logvar)
            output = model.decode(z, seq_length=batch.shape[1])

            probs = F.softmax(output / 0.6, dim=-1)
            tokens = probs.argmax(dim=-1).cpu().numpy()

            for i in range(batch.shape[0]):
                original_expr = ' '.join([ID_TO_TOKEN.get(t.item(), '?') for t in batch[i]])
                reconstructed_expr = ' '.join([ID_TO_TOKEN.get(t, '?') for t in tokens[i]])

                matcher = SequenceMatcher(None, original_expr, reconstructed_expr)
                accuracy = matcher.ratio()
                total_accuracy += accuracy

    return total_accuracy / num_samples

# Training execution
train_loader = DataLoader(ExpressionDataset(train_expressions), batch_size=128, shuffle=True)
val_loader = DataLoader(ExpressionDataset(val_expressions), batch_size=128, shuffle=False)

# **Move model to GPU**
model = PrefixVAE().to(device)

# **Train using GPU**
train_vae(model, train_loader, val_loader, num_epochs=1000, patience=20)
