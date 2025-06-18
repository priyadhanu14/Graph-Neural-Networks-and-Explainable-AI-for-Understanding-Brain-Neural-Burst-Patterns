import logging
import h5py
import numpy as np
import networkx as nx
import pickle
from joblib import Parallel, delayed
from scipy.stats import entropy
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

def bin_to_tick(burst_bin, bin_width_ms=10, tick_ms=0.1):
    """
    Convert burst start bin number to tick.
    
    Parameters:
    - burst_bin: int, the burst start bin index.
    - bin_width_ms: float, duration of each bin in milliseconds (default 10ms).
    - tick_ms: float, duration of one tick in milliseconds (default 0.1ms).
    
    Returns:
    - tick: int, corresponding tick number.
    """
    ticks_per_bin = bin_width_ms / tick_ms
    return int(burst_bin * ticks_per_bin)



def get_pre_burst_spikes_and_features(h5_path, node_ids, burst_tick, k=20):
    """
    For each node in node_ids, extract last k spikes before burst_tick and compute features.
    Returns a dict: node_id -> {'spikes': np.array[k], **features_dict}
    """
    result = {}
    with h5py.File(h5_path, 'r') as h5:
        for n in node_ids:
            ds = f"Neuron_{n}"
            if ds not in h5:
                arr = np.zeros(k, dtype=float)
                feats = get_spike_features(arr)
            else:
                spikes = np.sort(h5[ds][()]).astype(float)
                idx0 = np.searchsorted(spikes, burst_tick)
                last_k = spikes[max(0, idx0-k):idx0]
                if len(last_k) < k:
                    last_k = np.pad(last_k, (k-len(last_k), 0), 'constant')
                arr = last_k
                feats = get_spike_features(arr)
            entry = {'spikes': arr}
            entry.update(feats)
            result[n] = entry
    return result

def get_non_burst_spikes_and_features(h5_path, node_ids, burst_tick, k=20, gap = 100):
    """
    Extract the last k spike times before burst_tick for each node.
    Returns a dict: node_id -> 1D np.array of length k.
    """
    result = {}
    with h5py.File(h5_path, 'r') as h5:
        for n in node_ids:
            ds = f"Neuron_{n}"
            if ds not in h5:
                arr = np.zeros(k, dtype=float)
                feats = get_spike_features(arr)
            else:
                spikes = np.sort(h5[ds][()]).astype(float)
                idx0 = np.searchsorted(spikes, burst_tick)
                last_k = spikes[max(0, idx0 - k - gap):idx0 - gap]
                if len(last_k) < k:
                    last_k = np.pad(last_k, (k-len(last_k), 0), 'constant')
                arr = last_k
                feats = get_spike_features(arr)
            entry = {'spikes': arr}
            entry.update(feats)
            result[n] = entry
    return result

def get_spike_features(spikes):
    """
    Convert spike times to features.
    Here we simply return the spike times as features.
    """
    # Example feature extraction: normalize and scale
    isis = np.diff(spikes)
    m, s = np.nanmean(isis), np.nanstd(isis)

    return {
       'mean_isi': m,
       'entropy_isi': entropy(isis)       
    }

# assume these helper functions are defined elsewhere in your script:
# - bin_to_tick(burst_bin, bin_width_ms, tick_ms)
# - get_spikes_and_features(h5_path, node_ids, burst_tick, k)
# - get_non_burst_spikes_and_features(h5_path, node_ids, burst_tick, k, gap)
# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S"
)

h5_path    = "/DATA/hdhanu/GNN/Burst_Data/tR_1.0--fE_0.98_10000.h5"
pkl_path   = "/DATA/hdhanu/GNN/Graphs_and_Features/Sub_graph/subgraphs_final.pkl"
entries    = pickle.load(open(pkl_path, 'rb'))
num_bursts = len(entries)
logging.info(f"Loaded {num_bursts} bursts to process")

# The names of summary features must match keys returned by get_spike_features
feat_names = ['mean_isi', 'entropy_isi']

# Process a single burst entry
def process_entry(idx, entry):
    originbin  = entry['burstoriginbin']
    burst_tick = bin_to_tick(originbin, bin_width_ms=10, tick_ms=0.1)
    G          = entry['subgraph']
    nodes      = list(G.nodes())
    edge_index = from_networkx(G).edge_index

    pre_dict = get_pre_burst_spikes_and_features(h5_path, nodes, burst_tick, k=20)
    non_dict = get_non_burst_spikes_and_features(h5_path, nodes, burst_tick, k=20, gap=100)

    # Build feature matrices in node order
    pre_feats = [[pre_dict[n][f] for f in feat_names] for n in nodes]
    non_feats = [[non_dict[n][f] for f in feat_names] for n in nodes]

    pre_data = Data(
        x=torch.tensor(pre_feats, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([1])
    )
    non_data = Data(
        x=torch.tensor(non_feats, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor([0])
    )

    # Log progress
    logging.info(f"Processed {idx+1}/{num_bursts} bursts")
    return pre_data, non_data

if __name__ == '__main__':
    logging.info("Starting parallel processing of bursts...")
    # Parallelize over all entries with their index
    results = Parallel(n_jobs= 20 , verbose=10)(
        delayed(process_entry)(i, entry) for i, entry in enumerate(entries)
    )
    logging.info("Parallel processing complete")

    # Unpack results
    data_list = []
    for pre_data, non_data in results:
        data_list.extend([pre_data, non_data])

    total_graphs = len(data_list)
    logging.info(f"Created {total_graphs} graph samples ({num_bursts} pre-burst + {num_bursts} non-burst)")

    # Save and create DataLoader
    torch.save(data_list, '/DATA/hdhanu/GNN/Manual_check/processed_data.pt')
    loader = DataLoader(data_list, batch_size=32, shuffle=True)

    # Sanity check first batch
    batch = next(iter(loader))
    logging.info(f"Sample batch → x: {batch.x.shape}, edge_index: {batch.edge_index.shape}, y: {batch.y.shape}")


