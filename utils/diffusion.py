import torch
import torch.nn.functional as F

def forward_diffusion(x0, t, scheduler):
    t = t.to(scheduler.alphas_cumprod.device)
    alpha_bar = scheduler.alphas_cumprod[t].unsqueeze(1)
    noise = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
    return x_t, noise

def diffusion_loss(model, x_gene, t, cond, scheduler):
    x_t, noise = forward_diffusion(x_gene, t, scheduler)
    pred_noise = model(x_t, t, cond)
    # smooth L1 loss
    return F.smooth_l1_loss(pred_noise, noise)

@torch.no_grad()
def generate_samples(model, scheduler, cond, timesteps=1000, device="cuda"):
    """
    Deterministic or variance in reverse steps.
    model.add_variance_in_reverse controls if we add noise.
    """
    model.eval()
    batch_size = cond.size(0)
    # Start from x_T ~ N(0,I)
    x_t = torch.randn(batch_size, model.output_layer.out_features, device=device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        alpha_t = scheduler.alphas[t]
        pred_noise = model(x_t, t_tensor, cond)
        denom = torch.sqrt(alpha_t)
        x_t = (1.0 / denom) * (x_t - torch.sqrt(1.0 - alpha_t) * pred_noise)

        # If we want variance:
        if hasattr(model, "add_variance_in_reverse") and model.add_variance_in_reverse:
            # Beta_t = 1 - alpha_t
            beta_t = 1.0 - alpha_t
            sigma_t = torch.sqrt(beta_t) 
            if t > 0:
                x_t += sigma_t * torch.randn_like(x_t)

    return x_t

@torch.no_grad()
def reconstruct_samples(model, scheduler, x_gene, cond, timesteps=1000, device="cuda"):
    """
    1) Forward diffuse x_gene -> x_t at random t
    2) Reverse diffuse x_t -> x_0
    Possibly add variance if model.add_variance_in_reverse is True.
    """
    model.eval()
    batch_size = x_gene.size(0)
    t_rand = torch.full((batch_size,), fill_value=timesteps-500, device=device, dtype=torch.long)  

    alpha_bar_t = scheduler.alphas_cumprod[t_rand].unsqueeze(1)

    noise = torch.randn_like(x_gene)
    x_t = torch.sqrt(alpha_bar_t) * x_gene + torch.sqrt(1.0 - alpha_bar_t) * noise

    x_current = x_t.clone()
    for step in reversed(range(t_rand.max() + 1)):
        step_tensor = torch.full((batch_size,), step, device=device, dtype=torch.long)
        alpha_step = scheduler.alphas[step]
        pred_noise = model(x_current, step_tensor, cond)
        x_current = (1.0 / torch.sqrt(alpha_step)) * (
            x_current - torch.sqrt(1.0 - alpha_step) * pred_noise
        )
        # Optional variance
        if hasattr(model, "add_variance_in_reverse") and model.add_variance_in_reverse:
            beta_step = 1.0 - alpha_step
            sigma_step = torch.sqrt(beta_step)
            if step > 0:
                x_current += sigma_step * torch.randn_like(x_current)

    return x_t, x_current
