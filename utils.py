import torch
import torch.nn as nn

def gradient_penalty(critic, real, fake, device="cpu"):

    BATCH_SIZE, pc_dim, point_num = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1), device=device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad_(True)

    critic_output = critic(interpolated_images)
    
    # gradients = ∂C(interpolated_images) / ∂(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=critic_output,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(BATCH_SIZE, -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty