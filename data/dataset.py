import torch
import anndata
from torch.utils.data import Dataset

class SpatialTranscriptomicsDataset(Dataset):
    """
    Splits an AnnData object by specific Bregma slices (train/test).
    Builds a multi-channel condition vector:
        1) ST embeddings
        2) (x,y) coords (optionally normalized)
        3) cell type (one-hot or learned)
    """
    def __init__(
        self,
        adata: anndata.AnnData,
        st_embeddings: anndata.AnnData,
        bregma_slices: list,
        use_cell_class: bool = True,
        one_hot_celltype: bool = True,
        normalize_coords: bool = True,
        device: str = "cuda"
    ):
        """
        adata: full AnnData with .obs containing 'Bregma', 'Centroid_X', 'Centroid_Y'.
        st_embeddings: matching AnnData with latent embeddings
        bregma_slices: which slices to filter (train/test)
        use_cell_class: whether to use adata.obs['Cell_class'] or 'Neuron_cluster_ID'
        one_hot_celltype: if True, do one-hot; else keep as numeric
        normalize_coords: if True, scale (x,y) to [0,1] range
        """
        # Filter by slices
        mask = adata.obs["Bregma"].isin(bregma_slices)
        self.adata = adata[mask].copy()
        self.embeddings = st_embeddings[mask].copy()
        self.device = device

        # Centroid coords
        xvals = self.adata.obs["Centroid_X"].values
        yvals = self.adata.obs["Centroid_Y"].values
        if normalize_coords:
            # Scale (x,y) into [0,1]
            xvals = (xvals - xvals.min()) / (xvals.max() - xvals.min() + 1e-9)
            yvals = (yvals - yvals.min()) / (yvals.max() - yvals.min() + 1e-9)
        self.centroid_x = torch.tensor(xvals, dtype=torch.float32, device=device)
        self.centroid_y = torch.tensor(yvals, dtype=torch.float32, device=device)

        # Cell type: "Cell_class" or "Neuron_cluster_ID"
        if use_cell_class:
            raw_ct = self.adata.obs["Cell_class"].astype("category").values
        else:
            raw_ct = self.adata.obs["Neuron_cluster_ID"].astype("category").values
        self.celltype_categories = list(raw_ct.categories)
        ct_codes = raw_ct.codes  
        self.celltype_tensor = torch.tensor(ct_codes, dtype=torch.long, device=device)

        self.one_hot_celltype = one_hot_celltype

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, idx):
        # Genes (X)
        x_gene = self.adata.X[idx].toarray().squeeze()  
        x_gene_t = torch.tensor(x_gene, dtype=torch.float32, device=self.device)

        # ST embedding
        emb = self.embeddings.X[idx].squeeze()  
        emb_t = torch.tensor(emb, dtype=torch.float32, device=self.device)

        # Spatial coords
        x_coord = self.centroid_x[idx].unsqueeze(0) 
        y_coord = self.centroid_y[idx].unsqueeze(0) 

        # Cell type
        ct_label = self.celltype_tensor[idx].item()

        if self.one_hot_celltype:
            # One-hot
            cdim = len(self.celltype_categories)
            ct_onehot = torch.zeros(cdim, device=self.device)
            ct_onehot[ct_label] = 1.0
            cond = torch.cat([emb_t, x_coord, y_coord, ct_onehot], dim=0)
        else:
            ct_numeric = torch.tensor([ct_label], dtype=torch.float32, device=self.device)
            cond = torch.cat([emb_t, x_coord, y_coord, ct_numeric], dim=0)

        return x_gene_t, cond
