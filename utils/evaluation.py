import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def cosine_similarity_score(x, y):
    """
    Compute cosine similarity between two tensors.
    x: Original data (B, D)
    y: Reconstructed data (B, D)
    """
    return F.cosine_similarity(x, y)

def pearson_correlation_coefficient(x, y):
    """
    Compute Pearson correlation coefficient for each pair in a batch.
    x: Original data (B, D)
    y: Reconstructed data (B, D)
    Returns: A tensor of PCC values of shape (B,)
    """
    # Ensure x and y have the same shape
    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Center the data (subtract mean)
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)

    # Compute the numerator (covariance)
    numerator = (x_centered * y_centered).sum(dim=1)

    # Compute the denominator (product of standard deviations)
    x_std = torch.sqrt((x_centered ** 2).sum(dim=1))
    y_std = torch.sqrt((y_centered ** 2).sum(dim=1))
    denominator = x_std * y_std

    # Compute PCC for each pair
    pcc = numerator / (denominator + 1e-8)  # Add small epsilon to avoid division by zero
    return pcc

def evaluate_reconstruction(original, reconstructed):
    """
    Compute CSS and PCC for a batch of original and reconstructed data.
    original: Original data (B, D)
    reconstructed: Reconstructed data (B, D)
    """
    css = cosine_similarity_score(original, reconstructed)
    pcc = pearson_correlation_coefficient(original, reconstructed)
    return css, pcc

def evaluate_model(dataset, model, scheduler, batch_size=64, device="cuda", embeddings=None):
    """
    Evaluate a model on a dataset, computing reconstruction metrics
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_css = 0.0
    total_pcc = 0.0
    
    from utils.diffusion import diffusion_loss, reconstruct_samples
    
    with torch.no_grad():
        for x_gene, x_cond in loader:
            x_gene = x_gene.to(device)
            x_cond = x_cond.to(device)

            # Forward diffuse to get noised data
            t = torch.full((x_gene.size(0),), fill_value=scheduler.timesteps-1,
                           device=device, dtype=torch.long)
            # Reconstruct data using reverse diffusion
            _, x_reconstructed = reconstruct_samples(model, scheduler, x_gene, x_cond, timesteps=1000, device=device)

            # Compute loss (noise prediction)
            loss = diffusion_loss(model, x_gene, t, x_cond, scheduler)
            total_loss += loss.item() * x_gene.size(0)

            # Compute CSS and PCC for reconstruction
            css, pcc = evaluate_reconstruction(x_gene, x_reconstructed)
            total_css += css.sum().item()  
            total_pcc += pcc.sum().item()  

    # Compute average metrics
    avg_loss = total_loss / len(dataset)
    avg_css = total_css / len(dataset)
    avg_pcc = total_pcc / len(dataset)
    print(f"Eval Noise Prediction Loss: {avg_loss:.4f}, CSS: {avg_css:.4f}, PCC: {avg_pcc:.4f}")
    return avg_loss, avg_css, avg_pcc

def generate_entire_dataset(model, scheduler, test_dataset, timesteps=1000, device="cuda"):
    """
    Generate data for ALL cells in the test dataset (e.g. 5578 x 161),
    rather than a single batch. We'll accumulate them in a list and cat.
    """
    model.eval()
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    all_generated = []
    
    from utils.diffusion import generate_samples
    
    for x_gene_batch, x_cond_batch in loader:
        x_cond_batch = x_cond_batch.to(device)
        
        generated_batch = generate_samples(model, scheduler, x_cond_batch, timesteps, device)
        
        all_generated.append(generated_batch.cpu())

    all_generated_cat = torch.cat(all_generated, dim=0)  
    return all_generated_cat

def reconstruct_entire_dataset(model, scheduler, test_dataset, timesteps=1000, device="cuda", embeddings=None):
    """
    Reconstruct data for all cells in the test dataset,
    collecting the noised and final reconstruction for each batch.
    """
    model.eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)   
    all_noised = []
    all_reconstructed = []
    
    from utils.diffusion import reconstruct_samples
    
    for x_gene_batch, x_cond_batch in loader:
        x_gene_batch = x_gene_batch.to(device)
        x_cond_batch = x_cond_batch.to(device)

        x_noised_batch, x_reconstructed_batch = reconstruct_samples(
            model, scheduler, x_gene_batch, x_cond_batch, timesteps, device=device
        )
        all_noised.append(x_noised_batch.cpu())
        all_reconstructed.append(x_reconstructed_batch.cpu())

    all_noised_cat = torch.cat(all_noised, dim=0)
    all_reconstructed_cat = torch.cat(all_reconstructed, dim=0)
    return all_noised_cat, all_reconstructed_cat
