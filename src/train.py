import torch
import torch.optim as optim
from utils.differentiation_utils import lane_emden_residual
from utils.data_utils import generate_training_data
from config import TRAINING_CONFIG

def loss(model, t, n, t_boundary, n_boundary, theta_boundary, dtheta_dt_boundary):
    residual = lane_emden_residual(model, t, n)

    t_boundary.requires_grad_(True)
    n_boundary.requires_grad_(True)
    
    t_n_boundary = torch.cat((t_boundary, n_boundary), dim=1)
    
    boundary_loss = torch.mean((model(t_n_boundary) - theta_boundary) ** 2) + \
                    torch.mean((torch.autograd.grad(model(t_n_boundary), t_boundary, grad_outputs=torch.ones_like(t_boundary), create_graph=True)[0] - dtheta_dt_boundary) ** 2)
    return torch.mean(residual ** 2) + boundary_loss

def train_model(model, t_n, t_boundary, n_boundary, theta_boundary, dtheta_dt_boundary):
    optimizer = optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])

    for epoch in range(TRAINING_CONFIG['num_epochs']):
        optimizer.zero_grad()
        loss_value = loss(model, t_n[:, 0:1], t_n[:, 1:2], t_boundary, n_boundary, theta_boundary, dtheta_dt_boundary)
        loss_value.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss_value.item()}')

    return model
