import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 定义符号回归中用到的数学符号（token）
tokens = ["x", "y", "+", "-", "*", "/", "sin", "cos", "exp", "log", "(", ")"]
token_to_idx = {token: i for i, token in enumerate(tokens)}
idx_to_token = {i: token for i, token in enumerate(tokens)}

# 训练数据：预定义数学表达式
training_data = [
    ["x", "*", "x", "+", "sin", "(", "x", ")"],  # x^2 + sin(x)
    ["exp", "(", "x", ")", "+", "log", "(", "y", ")"],  # e^x + log(y)
    ["x", "/", "y", "+", "cos", "(", "x", ")"],  # x/y + cos(x)
]

# 将表达式转换为索引序列
def encode_expression(expression):
    return [token_to_idx[token] for token in expression]

# 训练集转换
encoded_data = [encode_expression(expr) for expr in training_data]

# 设置 LSTM 训练参数
input_dim = len(tokens)
embedding_dim = 16
hidden_dim = 64
output_dim = len(tokens)
num_layers = 2

# LSTM 生成器模型
class ExpressionLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers):
        super(ExpressionLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

# 初始化模型
model = ExpressionLSTM(input_dim, embedding_dim, hidden_dim, output_dim, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 训练 LSTM 生成器
num_epochs = 500
for epoch in range(num_epochs):
    total_loss = 0
    for expr in encoded_data:
        input_seq = torch.tensor([expr[:-1]], dtype=torch.long)
        target_seq = torch.tensor([expr[1:]], dtype=torch.long)
        
        hidden = (torch.zeros(num_layers, 1, hidden_dim),
                  torch.zeros(num_layers, 1, hidden_dim))

        optimizer.zero_grad()
        output, _ = model(input_seq, hidden)
        loss = criterion(output.view(-1, output_dim), target_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

def token_sequence_to_expression(token_sequence):
    """将token序列转换为可读的数学表达式"""
    expr = " ".join(token_sequence)
    # 移除多余的空格
    expr = expr.replace("( ", "(").replace(" )", ")")
    return expr

# 生成新的数学表达式
def generate_expression(model, start_token="x", max_length=10):
    model.eval()
    input_seq = torch.tensor([[token_to_idx[start_token]]], dtype=torch.long)
    hidden = (torch.zeros(num_layers, 1, hidden_dim),
              torch.zeros(num_layers, 1, hidden_dim))
    
    generated = [start_token]
    for _ in range(max_length):
        output, hidden = model(input_seq, hidden)
        next_token_idx = torch.argmax(output[:, -1, :]).item()
        next_token = idx_to_token[next_token_idx]
        generated.append(next_token)
        input_seq = torch.tensor([[next_token_idx]], dtype=torch.long)
        if next_token == ")":
            break
    
    token_sequence = " ".join(generated)
    math_expression = token_sequence_to_expression(generated)
    print("Generated Token Sequence:", token_sequence)
    print("Mathematical Expression:", math_expression)
    return token_sequence

# 生成测试表达式
generated_expr = generate_expression(model)