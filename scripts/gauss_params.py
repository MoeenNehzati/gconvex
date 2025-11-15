import torch
params = {
    "n": 1000,
    "μ_x": torch.tensor([0.0, 0.0]),
    "Σ_x": torch.tensor([[0.7, 0.5], [0.5, 0.7]]),
    "μ_y": torch.tensor([1.0, 1.0]),
    "Σ_y": torch.tensor([[0.4, -0.2], [-0.2, 0.4]]),
}
niters = 10000
model_size = 100
inner_steps = 1
lr = 5e-4