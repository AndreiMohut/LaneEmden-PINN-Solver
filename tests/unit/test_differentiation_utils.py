import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import torch
from utils.differentiation_utils import lane_emden_residual
from model import PINN
from config import MODEL_CONFIG

def test_lane_emden_residual():
    model = PINN(
        input_size=MODEL_CONFIG['input_size'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        output_size=MODEL_CONFIG['output_size'],
        activation=MODEL_CONFIG['activation'],
        output_activation=MODEL_CONFIG['output_activation']
    )
    
    t_n = torch.tensor([[0.1, 1.0], [0.5, 2.0]], dtype=torch.float32)
    t = t_n[:, 0:1]
    n = t_n[:, 1:2]
    
    residual = lane_emden_residual(model, t, n)
    
    assert residual.shape == (2, 1), "Residual should have the same batch size as input"
    assert isinstance(residual, torch.Tensor), "Residual should be a torch.Tensor"
