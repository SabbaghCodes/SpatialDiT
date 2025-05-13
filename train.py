import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CyclicLR

def train_model(
    dataset, 
    model, 
    optimizer, 
    scheduler_cyclic, 
    scheduler, 
    validation_split=0.1, 
    epochs=10, 
    batch_size=32, 
    patience=5, 
    device="cuda",
    wandb_log=False
):
    """
    Train a diffusion model with early stopping
    
    Args:
        dataset: PyTorch Dataset
        model: Model to train
        optimizer: PyTorch optimizer
        scheduler_cyclic: Learning rate scheduler
        scheduler: Diffusion noise scheduler
        validation_split: Fraction of data to use for validation
        epochs: Maximum number of epochs to train
        batch_size: Batch size for training
        patience: Early stopping patience
        device: Device to train on
        wandb_log: Whether to log to Weights & Biases
    """
    model.to(device)
    
    # Split dataset into training and validation sets
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0
    
    from utils.diffusion import diffusion_loss, reconstruct_samples
    from utils.evaluation import evaluate_reconstruction

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        for x_gene, x_cond in train_loader:
            x_gene = x_gene.to(device)
            x_cond = x_cond.to(device)
            
            t = torch.randint(0, scheduler.timesteps, (x_gene.size(0),), device=device)
            optimizer.zero_grad()
            loss = diffusion_loss(model, x_gene, t, x_cond, scheduler)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler_cyclic.step()
            total_loss += loss.item() * x_gene.size(0)
        
        avg_train_loss = total_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_css = 0.0
        val_pcc = 0.0
        with torch.no_grad():
            for x_gene, x_cond in val_loader:
                x_gene = x_gene.to(device)
                x_cond = x_cond.to(device)
                
                t = torch.randint(0, scheduler.timesteps, (x_gene.size(0),), device=device)
                loss = diffusion_loss(model, x_gene, t, x_cond, scheduler)
                val_loss += loss.item() * x_gene.size(0)

                # Compute CSS and PCC for validation
                _, x_reconstructed = reconstruct_samples(model, scheduler, x_gene, x_cond, timesteps=1000, device=device)
                css, pcc = evaluate_reconstruction(x_gene, x_reconstructed)
                val_css += css.sum().item()
                val_pcc += pcc.sum().item()
        
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_css = val_css / len(val_dataset)
        avg_val_pcc = val_pcc / len(val_dataset)

        # Log metrics
        print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation CSS: {avg_val_css:.4f}, "
              f"Validation PCC: {avg_val_pcc:.4f}")
        
        if wandb_log:
            import wandb
            wandb.log({
                "Training Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Validation CSS": avg_val_css,
                "Validation PCC": avg_val_pcc,
                "Epoch": epoch
            })
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))
    print("Training complete. Best model restored.")
    
    return model

def prepare_cond(cond, embeddings_to_select, perturbation_type=None, cond_type_to_perturb=None):
    """
    Prepare condition tensor with optional perturbations.
    
    Args:
        cond: Full condition tensor (B, D)
        embeddings_to_select: List of embeddings to include (e.g., ["st_embedding", "spatial_coordinates", "cell_type"])
        perturbation_type: "shuffle" or "noise"
        cond_type_to_perturb: "spatial_coordinates" or "cell_type" (required if perturbation_type is not None)
    """
    if perturbation_type is not None:
        assert cond_type_to_perturb in embeddings_to_select
        if perturbation_type == "shuffle":
            cond = shuffle_embeddings(cond, cond_type_to_perturb)
        else:
            raise ValueError("Unsupported perturbation type")

    # Prepare the condition tensor based on the selected embeddings
    prepared = []
    if "st_embedding" in embeddings_to_select:
        prepared.append(cond[:, :64])  # 64 for st_embeddings
    if "spatial_coordinates" in embeddings_to_select:
        prepared.append(cond[:, 64:66])  # 2 for centroid_x and centroid_y
    if "cell_type" in embeddings_to_select:
        prepared.append(cond[:, 66:])  # 16 for cell_type
    return torch.cat(prepared, dim=-1)

def shuffle_embeddings(cond, cond_type_to_perturb):
    """
    Shuffle embeddings of a specific type.
    
    Args:
        cond: Full condition tensor (B, D)
        cond_type_to_perturb: "spatial_coordinates" or "cell_type"
    """
    if cond_type_to_perturb == "spatial_coordinates":
        # Shuffle spatial coordinates (assumed to be in columns 64:66)
        cond[:, 64:66] = cond[:, 64:66][torch.randperm(cond.size(0))]
    elif cond_type_to_perturb == "cell_type":
        # Shuffle cell type labels (assumed to be in columns 66:])
        cond[:, 66:] = cond[:, 66:][torch.randperm(cond.size(0))]
    return cond

def add_gaussian_noise(cond, cond_type_to_perturb, noise_std=0.1):
    """
    Add Gaussian noise to embeddings of a specific type.
    
    Args:
        cond: Full condition tensor (B, D)
        cond_type_to_perturb: "spatial_coordinates" or "cell_type"
        noise_std: Standard deviation of the Gaussian noise
    """
    if cond_type_to_perturb == "spatial_coordinates":
        # Add noise to spatial coordinates (assumed to be in columns 64:66)
        noise = torch.randn_like(cond[:, 64:66]) * noise_std
        cond[:, 64:66] += noise
    elif cond_type_to_perturb == "cell_type":
        # Add noise to cell type logits (assumed to be in columns 66:])
        logits = torch.log(cond[:, 66:] + 1e-8)  # Convert to logits
        noise = torch.randn_like(logits) * noise_std
        perturbed_logits = logits + noise
        cond[:, 66:] = torch.nn.functional.softmax(perturbed_logits, dim=1)  # Convert back to probabilities
    return cond
