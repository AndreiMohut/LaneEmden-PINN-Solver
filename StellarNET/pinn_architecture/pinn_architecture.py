import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu

class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_normal_(self.linear.weight)  # Glorot initialization for W
        nn.init.zeros_(self.linear.bias)            # Bias initialized to 0

    def forward(self, x):
        return self.linear(x)

class PirateNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PirateNetBlock, self).__init__()
        self.U = DenseLayer(input_dim, hidden_dim)  # U = σ(W1 * Φ(x) + b1)
        self.V = DenseLayer(input_dim, hidden_dim)  # V = σ(W2 * Φ(x) + b2)
        self.dense1 = DenseLayer(input_dim, hidden_dim)
        self.dense2 = DenseLayer(hidden_dim, hidden_dim)
        self.dense3 = DenseLayer(hidden_dim, input_dim)
        self.alpha = nn.Parameter(torch.zeros(1))  # Trainable parameter alpha

    def forward(self, phi_x):
        # First gate mechanism
        f1 = torch.relu(self.dense1(phi_x))  # f1 = σ(W1^l * phi_x + b1^l)
        U = torch.relu(self.U(phi_x))  # Learnable parameter U
        V = torch.relu(self.V(phi_x))  # Learnable parameter V
        z1 = f1 * U + (1 - f1) * V  # z1^l = f1 ⊙ U + (1 − f1) ⊙ V

        # Second gate mechanism
        f2 = torch.relu(self.dense2(z1))
        z2 = f2 * U + (1 - f2) * V  # z2^l = g1 ⊙ U + (1 − g1) ⊙ V

        # Third gate mechanism
        h = torch.relu(self.dense3(z2))  # h^l
        phi_x_next = self.alpha * h + (1 - self.alpha) * phi_x  # phi_x^(l+1) = α^l ⊙ h^l + (1 − α^l) ⊙ phi_x

        return phi_x_next

class PirateNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, output_dim=1):
        super(PirateNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # Adjustable output size
        self.blocks = nn.ModuleList([PirateNetBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])
    
    def forward(self, phi_x):
        phi_x = torch.tanh(self.input_layer(phi_x))
        for block in self.blocks:
            phi_x = block(phi_x)
        return self.output_layer(phi_x)

class ComplexDenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexDenseLayer, self).__init__()
        self.linear = ComplexLinear(input_dim, output_dim)

        # Apply Xavier Initialization manually to the weight matrices
        with torch.no_grad():
            nn.init.xavier_normal_(self.linear.fc_r.weight)  # Real part
            nn.init.xavier_normal_(self.linear.fc_i.weight)  # Imaginary part

            # Initialize biases to zero
            if self.linear.fc_r.bias is not None:
                nn.init.zeros_(self.linear.fc_r.bias)
            if self.linear.fc_i.bias is not None:
                nn.init.zeros_(self.linear.fc_i.bias)

    def forward(self, x):
        return self.linear(x)

class ComplexPirateNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ComplexPirateNetBlock, self).__init__()
        self.U = ComplexDenseLayer(input_dim, hidden_dim)
        self.V = ComplexDenseLayer(input_dim, hidden_dim)
        self.dense1 = ComplexDenseLayer(input_dim, hidden_dim)
        self.dense2 = ComplexDenseLayer(hidden_dim, hidden_dim)
        self.dense3 = ComplexDenseLayer(hidden_dim, input_dim)
        self.alpha = nn.Parameter(torch.zeros(1, dtype=torch.complex64))  # Trainable parameter alpha

    def forward(self, phi_x):
        f1 = complex_relu(self.dense1(phi_x))
        U = torch.sigmoid(self.U(phi_x))
        V = torch.sigmoid(self.V(phi_x))
        z1 = f1 * U + (1 - f1) * V

        f2 = complex_relu(self.dense2(z1))
        z2 = f2 * U + (1 - f2) * V

        h = complex_relu(self.dense3(z2))
        phi_x_next = self.alpha * h + (1 - self.alpha) * phi_x
        return phi_x_next

class ComplexPirateNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, output_dim=1):
        super(ComplexPirateNet, self).__init__()
        self.input_layer = ComplexDenseLayer(input_dim, hidden_dim)
        self.output_layer = ComplexDenseLayer(hidden_dim, output_dim)
        self.blocks = nn.ModuleList([ComplexPirateNetBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])
    
    def forward(self, phi_x):
        phi_x = complex_relu(self.input_layer(phi_x))
        for block in self.blocks:
            phi_x = block(phi_x)
        return self.output_layer(phi_x)

class StellarLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StellarLayer, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.dense1.weight)
        nn.init.xavier_normal_(self.dense2.weight)
        nn.init.zeros_(self.dense1.bias)
        nn.init.zeros_(self.dense2.bias)

    def forward(self, x):
        x = torch.sigmoid(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        return x

class StellarNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StellarNetBlock, self).__init__()
        self.U = StellarLayer(input_dim, hidden_dim)
        self.V = StellarLayer(input_dim, hidden_dim)
        
        # Multiple stellar dense layers
        self.stellar1 = StellarLayer(input_dim, hidden_dim)
        self.stellar2 = StellarLayer(hidden_dim, hidden_dim)
        self.dense = nn.Linear(hidden_dim, input_dim)
        
        # Learnable scaling parameter
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Stellar transformation
        nest_out = self.stellar1(x)
        nest_out = self.stellar2(nest_out)
        
        # Gate mechanism with softmax
        f = F.softmax(self.U(x), dim=-1)
        g = torch.maximum(self.V(nest_out), torch.zeros_like(nest_out))  # Max operation
        
        # Residual update
        z1 = f * g + (1 - f) * x
        
        # Final stellar skip connection
        h = self.stellar2(z1)
        x_next = self.beta * h + (1 - self.beta) * x
        return x_next

class StellarNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super(StellarNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.blocks = nn.ModuleList([StellarNetBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        
        # Initialize layers
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        # Input transformation
        x = torch.tanh(self.input_layer(x))
        
        # Stellar blocks
        for block in self.blocks:
            x = block(x)
        
        # Output layer
        return self.output_layer(x)