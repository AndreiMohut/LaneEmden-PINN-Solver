import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation, output_activation):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        
        self.activation = getattr(nn, activation)()
        self.output_activation = getattr(nn, output_activation)()

    def forward(self, t_n):
        x = self.activation(self.fc1(t_n))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output_activation(self.fc4(x))
        x = self.fc5(x)
        return x
