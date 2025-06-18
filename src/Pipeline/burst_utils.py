# burst_utils.py
import multiprocessing as mp, torch, numpy as np
from pathlib import Path
from joblib import Parallel, delayed

# ---- bring in all helpers you already wrote -------------------
from src.Pipeline.Full_run import (
    chunker, process_chunk, clean_and_scale_data,
    GCN, run_epoch, FEAT_NAMES, H5_PATH
)

ENTRIES_PT = Path("/DATA/hdhanu/GNN/Subgraphs/last_quarter_subgraphs.pt")

def _ensure_spawn():
    """
    Call this once per process; safe if start method already set.
    """
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def build_data_list(window: int,
                    gap: int,
                    mask_shift: int,
                    *,
                    entries_path: Path | str = ENTRIES_PT,
                    chunk_size: int = 100,
                    n_jobs: int = 20,
                    scale: bool = True):
    """
    Generate a list[torch_geometric.data.Data] for ONE (window, gap, shift).
    Mirrors the logic used in your main sweep script.

    Args
    ----
    window, gap, mask_shift : ints    : same meaning as in sweep
    entries_path            : .pt     : file with burst 'entries'
    chunk_size              : int     : how many burst-entries per worker batch
    n_jobs                  : int     : number of joblib workers
    scale                   : bool    : whether to z-score node features

    Returns
    -------
    data_list : list[Data]
    """
    _ensure_spawn()

    entries = torch.load(entries_path, weights_only=False)
    chunks  = list(chunker(entries, chunk_size))
    max_w   = min(n_jobs, mp.cpu_count() - 2)

    results = Parallel(n_jobs=max_w, backend="loky", verbose=0)(
        delayed(process_chunk)(c, window, gap, mask_shift) for c in chunks
    )
    data_list = [g for ch in results for g in ch]

    if scale:
        # write to a temporary file then reload (clean_and_scale_data operates in-place)
        tmp = Path(".tmp_scaled.pt")
        clean_and_scale_data(data_list, tmp)
        tmp.unlink(missing_ok=True)

    return data_list


# ---- tidy re-exports so user can "from burst_utils import X" ----
__all__ = [
    "build_data_list",
    "GCN",
    "run_epoch",
    "FEAT_NAMES",
]
