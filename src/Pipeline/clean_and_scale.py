import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def clean_and_scale_data(
        data_list,
    scaled_pt: str
):
    """
    1) Loads a list of PyG Data objects from `input_pt`
    2) Drops any burst‐pairs (pre+non) containing NaNs
    3) Saves the cleaned list to `clean_pt`
    4) Fits a StandardScaler on all node‐features
       and transforms every Data.x in place
    5) Saves the scaled list to `scaled_pt`
    6) Dumps the fitted scaler to `scaler_pkl`
    """

    # --- (2) drop NaN pairs ---
    clean_list = []
    dropped    = 0
    for i in range(0, len(data_list), 2):
        pre, non = data_list[i], data_list[i+1]
        if torch.isnan(pre.x).any() or torch.isnan(non.x).any():
            dropped += 1
            continue
        clean_list.extend([pre, non])
    print(f"Dropped {dropped} bursts ({dropped*2} graphs) with NaNs")

    # --- (3) stack all node features to fit scaler ---
    F = np.vstack([ d.x.numpy() for d in clean_list ])  # shape [total_nodes, n_feats]

    scaler = StandardScaler().fit(F)

    # --- (4) transform each graph’s x in place ---
    for data in clean_list:
        X = data.x.numpy()
        Xs = scaler.transform(X)
        data.x = torch.tensor(Xs, dtype=torch.float)

    # --- save scaled data ---
    torch.save(clean_list, scaled_pt)
    print(f"Scaled data saved to {scaled_pt}")

    return clean_list, scaler
