#!/usr/bin/env python3
# run_gnnexplainer.py  –  PyG 2.6.1, edge masks for every graph
# ------------------------------------------------------------
import shutil
import torch, numpy as np, random
from pathlib import Path
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import ModelConfig, ModelTaskLevel, ModelReturnType

# --- paths -----------------------------------
RUN_DIR  = Path("/DATA/hdhanu/GNN/Explainability/w5_g1000_m0")
BEST_PT  = RUN_DIR / "best_model.pt"
GRAPH_PT = RUN_DIR / "cleaned_scaled.pt"
MASK_DIR = RUN_DIR / "gnnexp_masks";  MASK_DIR.mkdir(exist_ok=True)

# --- load graphs & model ----------------------------------------
from src.Pipeline.burst_utils import GCN, FEAT_NAMES
graphs = torch.load(GRAPH_PT, weights_only=False)
print(f"Loaded {len(graphs)} graphs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GCN(len(FEAT_NAMES), hidden=64, num_classes=2).to(device)
model.load_state_dict(torch.load(BEST_PT, map_location=device))
model.eval()

# --- configure GNNExplainer -------------------------------------
gnn_algo = GNNExplainer(
    epochs         = 400,        # iterations per graph
    lr             = 0.02,
    edge_mask_type = "object",   # learn only edges
    node_mask_type = "object",       # skip node-feature masks (faster)
    coeffs         = dict(edge_size=3e-4, edge_ent=3e-4),
).to(device)

explainer = Explainer(
    model            = model,
    algorithm        = gnn_algo,
    explanation_type = "phenomenon",                # per-prediction
    edge_mask_type   = "object",
    node_mask_type   = "object",
    model_config     = ModelConfig(
        mode        = "multiclass_classification",  # two logits
        task_level  = ModelTaskLevel.graph,
        return_type = ModelReturnType.raw,          # raw logits
    ),
)

# --- helper to write one mask file ------------------------------
def save_mask(idx: int, edge_index: torch.Tensor, label, edge_mask_logits: torch.Tensor, node_mask_logits: torch.Tensor):
    np.savez(MASK_DIR / f"graph_{idx:05d}_y{label}.npz",
             edge_index=edge_index.cpu().numpy(),
             edge_mask=edge_mask_logits.sigmoid().cpu().numpy(),   # convert to prob
             node_mask=node_mask_logits.sigmoid().cpu().numpy())   # convert to prob

# --- run explainer (demo: first 100 graphs) ---------------------
          # set to len(graphs) for full run
random.seed(42)
subset_ids = range(0, len(graphs))
print(f"Running GNNExplainer on {len(subset_ids)} graphs")
loader = DataLoader([graphs[i] for i in subset_ids], batch_size=1)

for i, data in enumerate(loader):
    data = data.to(device)

    exp = explainer(
        x          = data.x,
        edge_index = data.edge_index,
        batch      = data.batch,
        target     = data.y.view(-1),          # scalar class id
    )

    save_mask(subset_ids[i], data.edge_index, data.y.view(-1), exp.edge_mask, exp.node_mask)
    print(f"graph {subset_ids[i]:05d} done")

print(f"Saved {len(subset_ids)} explanations → {MASK_DIR}")

