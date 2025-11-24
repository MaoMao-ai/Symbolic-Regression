import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sympy as sp

# Load the token mappings
TOKEN_TO_ID = {
    '+': 1, '-': 2, '*': 3, '/': 4, '^': 5,  
    'x': 6, 'y': 7, 'z': 8,  
    '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14, '6': 15, '7': 16, '8': 17, '9': 18,  
    '√': 19, 'log': 20, 'sin': 21, 'cos': 22,  
    'π': 23, 'e': 24  
}
ID_TO_TOKEN = {v: k for k, v in TOKEN_TO_ID.items()}

# Convert infix expression to prefix notation (Shunting Yard Algorithm)
def infix_to_prefix(expression):
    """
    Converts an infix mathematical expression to prefix notation.
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    stack, output = [], []
    tokens = expression.replace('(', ' ( ').replace(')', ' ) ').split()
    
    for token in reversed(tokens):
        if token.isnumeric() or token in {'x', 'y', 'z', 'π', 'e'}:
            output.append(token)
        elif token == ')':
            stack.append(token)
        elif token == '(':
            while stack and stack[-1] != ')':
                output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] != ')' and precedence.get(token, 0) <= precedence.get(stack[-1], 0):
                output.append(stack.pop())
            stack.append(token)

    while stack:
        output.append(stack.pop())

    return output[::-1]

# Function to encode the input expression
def encode_expression(prefix_seq, max_length=20):
    encoded = [TOKEN_TO_ID.get(token, 0) for token in prefix_seq]
    return torch.tensor(encoded + [0] * (max_length - len(encoded)), dtype=torch.long).unsqueeze(0)

# Define the PrefixVAE model
class PrefixVAE(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=64, hidden_dim=128, latent_dim=64, max_length=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Ensure `num_layers=2` and `bidirectional=True` match your training setup
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


# Load trained model
def load_model(model, model_path="vae_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Model successfully loaded from {model_path}")
    else:
        print(f"⚠️ Model file {model_path} not found!")

# Decode function using Gumbel-Softmax
def decode_expression(model, z, seq_length=20):
    """
    Decodes the latent vector into a prefix mathematical expression.
    Cleans the output by removing redundant tokens.
    """
    output = model.decode(z, seq_length)
    gumbel_probs = F.gumbel_softmax(output, tau=0.5, dim=-1)
    tokens = gumbel_probs.argmax(dim=-1).cpu().numpy().flatten().tolist()

    # Convert token IDs back to symbols
    expression = [ID_TO_TOKEN.get(t, '?') for t in tokens]

    # **Remove excessive duplicates (e.g., π π π π π)**
    cleaned_expr = []
    for i, token in enumerate(expression):
        if i == 0 or token != expression[i - 1]:  # Avoid duplicate adjacent tokens
            cleaned_expr.append(token)

    return ' '.join(cleaned_expr)



# Function to evaluate the original and generated function using SymPy
def evaluate_functions(original_expr, generated_expr):
    """
    Plots the original function and the generated function over a given range using SymPy.
    """
    x = sp.symbols('x')
    
    try:
        f_original = sp.sympify(original_expr.replace("^", "**"))
        f_generated = sp.sympify(generated_expr.replace("^", "**"))

        x_vals = np.linspace(-10, 10, 100)
        y_original = [f_original.subs(x, val) for val in x_vals]
        y_generated = [f_generated.subs(x, val) for val in x_vals]

        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_original, label="Original Function", linestyle='dashed', color='blue')
        plt.plot(x_vals, y_generated, label="Generated Function", linestyle='solid', color='red')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Original vs Generated Function")
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"⚠️ Error in function evaluation: {e}")

# Main function to run inference
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrefixVAE().to(device)
    load_model(model)

    # Get user input for the function
    user_input = input("Enter a mathematical function (e.g., 'x ^ 2 + 3 * x - 5'): ")
    
    # Convert to prefix notation
    prefix_seq = infix_to_prefix(user_input)
    encoded_input = encode_expression(prefix_seq).to(device)

    # Pass through the VAE
    with torch.no_grad():
        mu, logvar = model.encode(encoded_input)
        z = model.reparameterize(mu, logvar)
        generated_expr = decode_expression(model, z, seq_length=20)

    # Print and compare expressions
    print(f"\nOriginal Expression:  {user_input}")
    print(f"Generated Expression: {generated_expr}")

    # Plot both functions
    evaluate_functions(user_input, generated_expr)

if __name__ == "__main__":
    main()
