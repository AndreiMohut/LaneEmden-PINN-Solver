import torch

def lane_emden_residual(model, t, n):
    t.requires_grad_(True)
    n.requires_grad_(True)
    
    t_n = torch.cat((t, n), dim=1)
    theta = model(t_n)
    
    dtheta_dt = torch.autograd.grad(theta, t, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    d2theta_dt2 = torch.autograd.grad(dtheta_dt, t, grad_outputs=torch.ones_like(dtheta_dt), create_graph=True)[0]
    residual = d2theta_dt2 + (2/t) * dtheta_dt + torch.pow(theta, n)
    return residual