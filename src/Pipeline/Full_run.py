#!/usr/bin/env python3
# burst_subgraph_pipeline.py
# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import os, time, logging, multiprocessing as mp, yaml, subprocess
from itertools import product
from pathlib import Path
from datetime import datetime

import h5py, torch, numpy as np, pandas as pd
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import entropy
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from clean_and_scale import clean_and_scale_data

# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------
H5_PATH = "/DATA/hdhanu/GNN/Burst_Data/tR_1.0--fE_0.98_10000.h5"
_worker_h5, _spike_cache = None, {}

def bin_to_tick(burst_bin, bin_width_ms=10, tick_ms=0.1):
    return int(burst_bin * (bin_width_ms / tick_ms))

def get_spike_features(spikes, window_ms: int):
    """Return feature dict for one neuron & one time-window."""
    isis   = np.diff(spikes)
    count  = len(spikes)
    mean_i = np.mean(isis) if count     else window_ms
    ent_i  = entropy(isis) if count > 1 else 0.0
    rate   = count / (window_ms * 1e-3)           # Hz
    last_l = window_ms - (spikes[-1] if count else 0)
    return dict(mean_isi=mean_i, entropy_isi=ent_i,
                count=count, last_lag=last_l, rate=rate)

FEAT_NAMES = ['mean_isi', 'entropy_isi', 'count', 'last_lag', 'rate']

def chunker(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def get_spikes_for_neuron(n: int):
    global _worker_h5, _spike_cache
    if _worker_h5 is None:
        _worker_h5 = h5py.File(H5_PATH, 'r')
    if n not in _spike_cache:
        ds = f"Neuron_{n}"
        _spike_cache[n] = (np.sort(_worker_h5[ds][()]).astype(float)
                           if ds in _worker_h5 else
                           np.empty(0, dtype=float))
    return _spike_cache[n]

def process_chunk(entries_chunk, window, gap, mask_shift):
    """Executed in parallel by joblib."""
    global _worker_h5, _spike_cache
    if _worker_h5 is None:
        _worker_h5 = h5py.File(H5_PATH, 'r')
        _spike_cache = {}

    out   = []
    w_ms  = window * 10
    pad_to = lambda arr: np.pad(arr, (window - len(arr), 0), 'constant') \
                         if len(arr) < window else arr[-window:]

    for entry in entries_chunk:
        burst_tick = bin_to_tick(entry['globalBin'])
        G          = entry['subgraph']
        nodes      = list(G.nodes())
        ei         = from_networkx(G).edge_index
        bid        = entry.get('burst_id', f"auto_{entry['globalBin']}")

        pre_feats  = np.zeros((len(nodes), len(FEAT_NAMES)), dtype=np.float32)
        non_feats  = np.zeros_like(pre_feats)

        for idx, n in enumerate(nodes):
            spikes = get_spikes_for_neuron(n)
            i0     = np.searchsorted(spikes, burst_tick)

            pre = spikes[max(0, i0-window-mask_shift): i0-mask_shift]
            non = spikes[max(0, i0-gap-window-mask_shift): i0-gap-mask_shift]

            pre, non = pad_to(pre), pad_to(non)

            pre_feats[idx] = [get_spike_features(pre,  w_ms)[f] for f in FEAT_NAMES]
            non_feats[idx] = [get_spike_features(non, w_ms)[f] for f in FEAT_NAMES]

        for feats, label in ((pre_feats, 1), (non_feats, 0)):
            g = Data(x=torch.from_numpy(feats),
                     edge_index=ei,
                     y=torch.tensor([label]))
            g.burst_id = bid
            out.append(g)
    return out

# ------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden, normalize=True)
        self.conv2 = GCNConv(hidden,      hidden, normalize=True)
        self.conv3 = GCNConv(hidden,      hidden, normalize=True)
        self.lin   = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch, *, return_embed=False, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        embed = global_mean_pool(x, batch)
        if return_embed:
            return embed
        out = F.dropout(embed, p=0.5, training=self.training)
        return self.lin(out)

# ------------------------------------------------------------
# 3. Epoch-runner
# ------------------------------------------------------------
def run_epoch(model, loader, crit, device, *,
              train=False, opt=None, compute_prf=False):
    model.train() if train else model.eval()
    tot_loss, corr = 0.0, 0
    all_y, all_pred = [], []

    for data in loader:
        data = data.to(device)
        if train: opt.zero_grad()

        out  = model(data.x, data.edge_index, data.batch)
        loss = crit(out, data.y.view(-1))
        if train:
            loss.backward(); opt.step()

        tot_loss += loss.item() * data.num_graphs
        preds     = out.argmax(1)
        corr     += (preds == data.y.view(-1)).sum().item()

        if compute_prf:
            all_y.extend(data.y.view(-1).cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    N     = len(loader.dataset)
    l_avg = tot_loss / N
    acc   = corr / N

    if compute_prf:
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_y, all_pred, average='binary', zero_division=0)
        return l_avg, acc, prec, rec, f1
    return l_avg, acc, None, None, None

def extract_embeddings(model, loader, device):
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z    = model(data.x, data.edge_index, data.batch, return_embed=True)
            embs.append(z.cpu().numpy())
            labs.append(data.y.view(-1).cpu().numpy())
    return np.vstack(embs), np.hstack(labs)

# ------------------------------------------------------------
# 4. Main sweep
# ------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    logging.basicConfig(format="%(asctime)s %(levelname)s | %(message)s",
                        level=logging.INFO)
    entries = torch.load('/DATA/hdhanu/GNN/Subgraphs/last_quarter_subgraphs.pt',
                         weights_only=False)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    windows, gaps, mask_shifts = [5], [1000], [0] # [10, 20, 50] [10, 50, 100,] [2, 5]
    logging.info(f"Dataset size = {len(entries)} bursts")

    chunk_size  = 100
    all_chunks  = list(chunker(entries, chunk_size))
    max_workers = min(20, mp.cpu_count() - 2)

    for window, gap, mask_shift in product(windows, gaps, mask_shifts):
        run_base = Path(f"w{window}_g{gap}_m{mask_shift}")
        if run_base.exists():
            logging.info(f"Skip {run_base} (done)")
            continue
        run_base.mkdir(parents=True)
        logging.info(f"=== {run_base} ===")

        yaml.safe_dump(dict(window=window, gap=gap, mask_shift=mask_shift,
                            ts=datetime.now().isoformat(),
                            git_sha=subprocess.getoutput("git rev-parse --short HEAD")),
                       open(run_base / "config.yaml", "w"))

        t0 = time.time()
        results   = Parallel(n_jobs=max_workers, backend='loky', verbose=5)(
            delayed(process_chunk)(c, window, gap, mask_shift) for c in all_chunks)
        data_list = [g for chunk in results for g in chunk]
        logging.info(f"Built {len(data_list)} graphs in {time.time()-t0:.1f}s")

        clean_and_scale_data(data_list, run_base / 'cleaned_scaled.pt')

        groups   = np.array([d.burst_id for d in data_list])
        gss      = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tr_val_i, te_i = next(gss.split(np.zeros(len(data_list)), groups=groups))

        gss_val  = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
        tr_i, va_i = next(gss_val.split(np.zeros(len(tr_val_i)),
                                        groups=groups[tr_val_i]))

        tr, va, te = ([data_list[i] for i in idx] for idx in (tr_i, va_i, te_i))
        logging.info("Group overlap OK")

        pos_cnt = sum(d.y.item() for d in tr)
        neg_cnt = len(tr) - pos_cnt
        skew    = max(pos_cnt, neg_cnt) / max(1, min(pos_cnt, neg_cnt))
        if skew > 1.5:
            weights = torch.tensor([len(tr)/neg_cnt, len(tr)/pos_cnt],
                                   dtype=torch.float, device=device)
            crit = torch.nn.CrossEntropyLoss(weight=weights)
            logging.info(f"Weighted loss {weights.tolist()}")
        else:
            crit = torch.nn.CrossEntropyLoss()

        dl_k = dict(batch_size=32, pin_memory=True, num_workers=0)
        tr_ld, va_ld, te_ld = (DataLoader(split, shuffle=shuf, **dl_k)
                               for split, shuf in ((tr, True), (va, False), (te, False)))

        in_ch = tr[0].x.size(1)
        model = GCN(in_channels=in_ch, hidden=64, num_classes=2).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=4, verbose=False)

        history, best_val, patience = [], float('inf'), 0
        for epoch in range(1, 101):
            tr_loss, tr_acc, *_ = run_epoch(model, tr_ld, crit, device,
                                            train=True, opt=opt)
            va_loss, va_acc, va_p, va_r, va_f1 = run_epoch(
                model, va_ld, crit, device, compute_prf=True)
            sched.step(va_loss)

            history.append((epoch, tr_loss, tr_acc,
                            va_loss, va_acc, va_p, va_r, va_f1))
            logging.info(f"{run_base} | Ep {epoch:03d} | "
                         f"Tr {tr_loss:.4f}/{tr_acc:.3f} | "
                         f"Va {va_loss:.4f}/{va_acc:.3f} "
                         f"P={va_p:.3f} R={va_r:.3f} F1={va_f1:.3f}")

            if va_loss < best_val:
                best_val, patience = va_loss, 0
                torch.save(model.state_dict(), run_base / 'best_model.pt')
            else:
                patience += 1
                if patience >= 10:
                    logging.info("Early stopping")
                    break

        model.load_state_dict(torch.load(run_base / 'best_model.pt'))
        te_loss, te_acc, te_p, te_r, te_f1 = run_epoch(
            model, te_ld, crit, device, compute_prf=True)
        logging.info(f"{run_base} | TEST "
                     f"loss={te_loss:.4f} acc={te_acc:.3f} "
                     f"P={te_p:.3f} R={te_r:.3f} F1={te_f1:.3f}")

        df = pd.DataFrame(history, columns=['epoch','train_loss','train_acc',
                                            'val_loss','val_acc',
                                            'val_prec','val_rec','val_f1'])
        df.loc['test'] = ['', '', '', te_loss, te_acc, te_p, te_r, te_f1]
        df.to_csv(run_base / 'metrics.csv', index=False)

        embs, labs = extract_embeddings(model, te_ld, device)
        np.save(run_base / 'embeddings.npy', embs)
        np.save(run_base / 'labels.npy',      labs)

        (run_base / "COMPLETE").touch()
        torch.cuda.empty_cache()
        logging.info(f"Finished {run_base}")

    torch.cuda.empty_cache()
    logging.info("All sweeps complete ✨")
