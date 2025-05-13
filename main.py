import os
import torch
import argparse
import numpy as np
import scanpy as sc
import wandb

from data.dataset import SpatialTranscriptomicsDataset
from models import ConditionalDiT, CosineNoiseScheduler
from train import train_model
from utils.evaluation import evaluate_model, generate_entire_dataset, reconstruct_entire_dataset
from torch.optim.lr_scheduler import CyclicLR

def parse_args():
    parser = argparse.ArgumentParser(description='Conditional Diffusion Transformer for Spatial Transcriptomics')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='merfish_all-slices.h5ad', help='Path to anndata file')
    parser.add_argument('--embeddings_path', type=str, default='merfish_novae_embeding.h5ad', help='Path to ST embeddings file')
    
    # Model parameters
    parser.add_argument('--block_type', type=str, default='adaLN', choices=['adaLN', 'cross_attention', 'in_context'], 
                        help='Type of attention block to use')
    parser.add_argument('--hidden_size', type=int, default=256, help='Dimension of hidden layers')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--add_variance', action='store_true', help='Whether to add variance in reverse diffusion')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    # Experiment parameters
    parser.add_argument('--leave_one_out', action='store_true', help='Whether to run leave-one-out cross validation')
    parser.add_argument('--test_slice', type=int, default=None, help='Specific slice to use as test set')
    parser.add_argument('--wandb_project', type=str, default=None, help='Weights & Biases project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    # ADDED: Example run mode
    parser.add_argument('--example', action='store_true', help='Run an example experiment with predefined parameters')
    
    return parser.parse_args()

def run_single_experiment(args, train_slices, test_slices):
    """
    Run a single experiment with specified train and test slices
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    train_slices : list
        List of slice indices to use for training
    test_slices : list
        List of slice indices to use for testing
        
    Returns:
    --------
    tuple
        (evaluation loss, CSS score, PCC score)
    """
    test_slice_str = '_'.join(map(str, test_slices))
    print(f"Running experiment with block_type={args.block_type}, hidden_size={args.hidden_size}, "
          f"num_layers={args.num_layers}, num_heads={args.num_heads}, lr={args.lr}, test_slice={test_slice_str}")
    
    # Set up wandb if requested
    if args.wandb_project:
        wandb_run_name = f"block_{args.block_type}_hidden_{args.hidden_size}_layers_{args.num_layers}_heads_{args.num_heads}_lr{args.lr}_test_slice={test_slice_str}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, reinit=True)
        wandb.config.update({
            "block_type": args.block_type,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "timesteps": args.timesteps,
            "train_slices": train_slices,
            "test_slice": test_slices
        })
        wandb_log = True
    else:
        wandb_log = False
    
    # Create datasets
    train_dataset = SpatialTranscriptomicsDataset(
        adata=adata_full,
        st_embeddings=st_embeddings,
        bregma_slices=train_slices,
        use_cell_class=True,
        one_hot_celltype=True,
        normalize_coords=True,
        device=args.device
    )

    test_dataset = SpatialTranscriptomicsDataset(
        adata=adata_full,
        st_embeddings=st_embeddings,
        bregma_slices=test_slices,
        use_cell_class=True,
        one_hot_celltype=True,
        normalize_coords=True,
        device=args.device
    )
    
    # Get dataset dimensions from a sample
    example_x, example_cond = train_dataset[0]
    cond_dim = example_cond.shape[0]
    gene_dim = example_x.shape[0]
    
    # Initialize model
    model = ConditionalDiT(
        input_dim=gene_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        block_type=args.block_type,
        cond_dim=cond_dim,
        add_variance_in_reverse=args.add_variance
    ).to(args.device)

    # Set up training components
    scheduler = CosineNoiseScheduler(timesteps=args.timesteps, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler_cyclic = CyclicLR(
        optimizer, 
        base_lr=args.lr/10., 
        max_lr=args.lr, 
        step_size_up=10, 
        step_size_down=10, 
        mode='triangular'
    )
    
    # Train model
    model = train_model(
        dataset=train_dataset,
        model=model,
        optimizer=optimizer,
        scheduler_cyclic=scheduler_cyclic,  
        scheduler=scheduler,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        wandb_log=wandb_log
    )

    # Evaluate model
    avg_eval_loss, avg_eval_css, avg_eval_pcc = evaluate_model(
        dataset=test_dataset,
        model=model,
        scheduler=scheduler,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if wandb_log:
        wandb.log({
            "Eval Loss": avg_eval_loss, 
            "Eval CSS": avg_eval_css, 
            "Eval PCC": avg_eval_pcc
        })
    
    # Save model and results
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_test_slice_{test_slice_str}.pt"))
    
    results = {
        'model_config': {
            'block_type': args.block_type,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
        },
        'train_slices': train_slices,
        'test_slices': test_slices,
        'metrics': {
            'loss': avg_eval_loss,
            'css': avg_eval_css,
            'pcc': avg_eval_pcc
        }
    }
    
    import json
    with open(os.path.join(args.save_dir, f"results_test_slice_{test_slice_str}.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Completed experiment: block_type={args.block_type}, hidden_size={args.hidden_size}, "
          f"num_layers={args.num_layers}, num_heads={args.num_heads}, lr={args.lr}, test_slice={test_slice_str}")
    
    if wandb_log:
        wandb.finish()
    
    return avg_eval_loss, avg_eval_css, avg_eval_pcc

# ADDED: Example experiment function
def run_example_experiment():
    """
    Run an example experiment with predefined parameters to demonstrate SpatialDiT model.
    
    This function demonstrates a complete experiment using the SpatialDiT model 
    on the Drosophila embryo dataset. It creates an optimized model configuration
    and trains on 15 slices while testing on slice 224.
    
    Example command:
    python main.py --example --data_path path/to/drosophila_data.h5ad --embeddings_path path/to/drosophila_embeddings.h5ad
    
    Note: This function modifies args in-place to set optimal hyperparameters.
    
    Returns:
    --------
    tuple
        (evaluation loss, CSS score, PCC score)
    """
    print("=" * 80)
    print("RUNNING EXAMPLE EXPERIMENT: 3D Reconstruction of Drosophila Embryo Data")
    print("=" * 80)
    
    # Set optimal hyperparameters for Drosophila dataset
    args.block_type = 'adaLN'  # Adaptive LayerNorm showed best performance
    args.hidden_size = 256     # Hidden dimension size
    args.num_layers = 6        # Number of transformer layers
    args.num_heads = 16        # Number of attention heads
    args.lr = 0.001            # Learning rate
    args.batch_size = 64       # Batch size
    args.epochs = 5            # Reduced epochs for example run (use 60 for full training)
    args.timesteps = 1000      # Number of diffusion timesteps
    
    print("\nModel Configuration:")
    print(f"  - Block Type: {args.block_type}")
    print(f"  - Hidden Size: {args.hidden_size}")
    print(f"  - Number of Layers: {args.num_layers}")
    print(f"  - Number of Heads: {args.num_heads}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Epochs: {args.epochs} (reduced for example)")
    print(f"  - Timesteps: {args.timesteps}")
    
    # For this example, we'll use a fixed test slice (224 for Drosophila)
    test_slice = 224
    
    # Get available slices and separate train/test
    slices = list(adata_full.obs["Bregma"].unique())
    if test_slice not in slices:
        print(f"Warning: Test slice {test_slice} not found. Using a random slice instead.")
        test_slice = np.random.choice(slices)
    
    train_slices = [s for s in slices if s != test_slice]
    test_slices = [test_slice]
    
    print(f"\nDataset Information:")
    print(f"  - Total number of slices: {len(slices)}")
    print(f"  - Training on {len(train_slices)} slices")
    print(f"  - Testing on slice {test_slice}")
    print(f"  - Total cells: {adata_full.n_obs}")
    print(f"  - Number of genes: {adata_full.n_vars}")
    
    print("\nRunning experiment...")
    results = run_single_experiment(args, train_slices, test_slices)
    
    print("\nExample Experiment Results:")
    print(f"  - Evaluation Loss: {results[0]:.6f}")
    print(f"  - Cell-type Spearman Score (CSS): {results[1]:.4f}")
    print(f"  - Pearson Correlation Coefficient (PCC): {results[2]:.4f}")
    print("\nResults saved to:", args.save_dir)
    print("=" * 80)
    
    return results

def main():
    """
    Main function to run SpatialDiT experiments.
    
    This function handles several experiment modes:
    1. Example mode: Runs a demo with predefined settings
    2. Leave-one-out cross-validation: Tests on each slice while training on others
    3. Single test slice: Tests on a specified slice
    4. Random split: Randomly selects a test slice
    
    Example commands:
    ----------------
    # Run the example experiment:
    python main.py --example --data_path drosophila_data.h5ad --embeddings_path drosophila_embeddings.h5ad
    
    # Run with specific test slice:
    python main.py --test_slice 224 --data_path drosophila_data.h5ad --embeddings_path drosophila_embeddings.h5ad
    
    # Run leave-one-out cross-validation:
    python main.py --leave_one_out --data_path merfish_all-slices.h5ad --embeddings_path merfish_novae_embeding.h5ad
    
    # Run with custom hyperparameters:
    python main.py --block_type adaLN --hidden_size 256 --num_layers 12 --num_heads 8 --lr 0.0001
    """
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Make sure the device is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead.")
        args.device = 'cpu'
    
    # Load data
    global adata_full, st_embeddings
    print(f"Loading data from {args.data_path}...")
    adata_full = sc.read_h5ad(args.data_path)
    print(f"Loading embeddings from {args.embeddings_path}...")
    st_embeddings = sc.read_h5ad(args.embeddings_path)
    
    # Round bregma values to integers
    adata_full.obs["Bregma"] = adata_full.obs["Bregma"].round().astype(int)
    
    # Get unique bregma slices
    slices = list(adata_full.obs["Bregma"].unique())
    print(f"Available bregma slices: {slices}")
    
    # ADDED: Example run mode
    if args.example:
        run_example_experiment()
        return
    
    if args.leave_one_out:
        # Run leave-one-out cross-validation
        all_results = []
        for i in range(len(slices)):
            train_slices = [slices[j] for j in range(len(slices)) if j != i]  # Use all but one slice for training
            test_slices = [slices[i]]  # Hold out one slice
            
            loss, css, pcc = run_single_experiment(args, train_slices, test_slices)
            all_results.append({
                'test_slice': test_slices[0],
                'loss': loss,
                'css': css,
                'pcc': pcc
            })
        
        # Summarize results
        print("\nLeave-one-out cross-validation results:")
        avg_loss = np.mean([r['loss'] for r in all_results])
        avg_css = np.mean([r['css'] for r in all_results])
        avg_pcc = np.mean([r['pcc'] for r in all_results])
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average CSS: {avg_css:.4f}")
        print(f"Average PCC: {avg_pcc:.4f}")
        
        # Save overall results
        import json
        with open(os.path.join(args.save_dir, "loocv_summary.json"), 'w') as f:
            json.dump({
                'model_config': {
                    'block_type': args.block_type,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'num_heads': args.num_heads,
                },
                'slice_results': all_results,
                'average_metrics': {
                    'loss': float(avg_loss),
                    'css': float(avg_css),
                    'pcc': float(avg_pcc)
                }
            }, f, indent=4)
    
    elif args.test_slice is not None:
        # Run single experiment with specific test slice
        if args.test_slice not in slices:
            raise ValueError(f"Test slice {args.test_slice} not found in data. Available slices: {slices}")
        
        train_slices = [s for s in slices if s != args.test_slice]
        test_slices = [args.test_slice]
        
        run_single_experiment(args, train_slices, test_slices)
    
    else:
        # Run with random train/test split
        test_idx = np.random.choice(len(slices))
        train_slices = [slices[i] for i in range(len(slices)) if i != test_idx]
        test_slices = [slices[test_idx]]
        
        run_single_experiment(args, train_slices, test_slices)

if __name__ == "__main__":
    main()