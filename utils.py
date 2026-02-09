import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device="cpu"):

    BATCH_SIZE, pc_dim, point_num = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1), device=device)
    interpolated = real * epsilon + fake * (1 - epsilon)
    
    norms = torch.norm(interpolated, dim=-1, keepdim=True)
    interpolated = interpolated / norms.clamp(min=1e-8) 

    interpolated.requires_grad_(True)

    critic_output = critic(interpolated)
    
    # gradients = ∂C(interpolated) / ∂(interpolated)
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=critic_output,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(BATCH_SIZE, -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty