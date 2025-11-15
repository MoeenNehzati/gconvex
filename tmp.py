import torch
from tools.dgps import generate_gaussian_pairs
n = 1000
μ_x = torch.tensor([1.,1.])
Σ_x = torch.tensor([[.7,0],[0,.7]])
μ_y = torch.tensor([0., 0.])
Σ_y = torch.tensor([[.7,0.5],[.5,.7]])
x, y, path = generate_gaussian_pairs(n, μ_x, Σ_x, μ_y, Σ_y)
