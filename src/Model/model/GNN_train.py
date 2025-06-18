import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import h5py
import numpy as np
import networkx as nx
import pickle
from joblib import Parallel, delayed
from scipy.stats import entropy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.manifold import TSNE

# --------------------------------------------------------
# 1) Insert your DataLoader definitions here:
#    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)
# --------------------------------------------------------
data_list = torch.load('/DATA/hdhanu/GNN/Sweep_and_tune/w20_g100_m0/cleaned_scaled.pt',weights_only=False)
print(f"Loaded {len(data_list)} graphs")

train_list, test_list = train_test_split(
    data_list, test_size=0.2, random_state=42, shuffle=True
)
print(f"{len(train_list)} train graphs, {len(test_list)} test graphs")

train_loader = DataLoader(train_list, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_list,  batch_size=32, shuffle=False)
print(f"Train loader: {len(train_loader)} batches, Test loader: {len(test_loader)} batches")

# 2) Model definition
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels,   hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin   = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, return_embed=False):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        embed = global_mean_pool(x, batch)
        if return_embed:
            return embed
        out = F.dropout(embed, p=0.5, training=self.training)
        return self.lin(out)

# Instantiate model
sample = next(iter(train_loader))
in_ch = sample.x.size(1)
num_classes = int(sample.y.max().item()) + 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=in_ch, hidden_channels=64, num_classes=num_classes).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=0.01)
crit  = torch.nn.CrossEntropyLoss()

# Containers for metrics
history = {
    'epoch': [],
    'train_loss': [], 'train_acc': [],
    'test_loss': [],  'test_acc': []
}

def train_epoch():
    model.train()
    total_loss = 0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = crit(out, data.y.view(-1))
        loss.backward()
        opt.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y.view(-1)).sum().item()
    avg_loss = total_loss / len(train_loader.dataset)
    acc = correct / len(train_loader.dataset)
    return avg_loss, acc

def eval_epoch(loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            total_loss += crit(out, data.y.view(-1)).item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

# 5) Training loop with logging
epochs = 100
for epoch in range(1, epochs+1):
    tr_loss, tr_acc = train_epoch()
    te_loss, te_acc = eval_epoch(test_loader)

    history['epoch'].append(epoch)
    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['test_loss'].append(te_loss)
    history['test_acc'].append(te_acc)

    print(f"Epoch {epoch:03d} | "
          f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
          f"Test Loss: {te_loss:.4f}, Test Acc: {te_acc:.4f}")

# 6) Save metrics to CSV
df_hist = pd.DataFrame(history)
df_hist.to_csv('training_metrics_1/4.csv', index=False)
print("Saved training_metrics_1/4.csv")

# 7) Extract embeddings
def extract_embeddings(loader):
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z = model(data.x, data.edge_index, data.batch, return_embed=True)
            embs.append(z.cpu().numpy())
            labs.append(data.y.view(-1).cpu().numpy())
    return np.vstack(embs), np.hstack(labs)

# 8) t-SNE & save plot
tsne = TSNE(n_components=2, random_state=42)

# Test tsne plot 
embeddings, labels = extract_embeddings(test_loader)
emb2d = tsne.fit_transform(embeddings)
plt.figure(figsize=(8,6))
for cls in np.unique(labels):
    idx = labels == cls
    plt.scatter(emb2d[idx,0], emb2d[idx,1], alpha=0.6, label=f"Class {cls}")
plt.legend()
plt.title("t-SNE of Graph Embeddings")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.tight_layout()
plt.savefig('tsne_embeddings_test.png')
plt.show()
print("Saved tSNE plot to tsne_embeddings_test.png")

# Train tsne plot
embeddings_train, labels_train = extract_embeddings(train_loader)
emb2d_train = tsne.fit_transform(embeddings_train)
plt.figure(figsize=(8,6))
for cls in np.unique(labels_train):
    idx = labels_train == cls
    plt.scatter(emb2d_train[idx,0], emb2d_train[idx,1], alpha=0.6, label=f"Class {cls}")
plt.legend()
plt.title("t-SNE of Graph Embeddings")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.tight_layout()
plt.savefig('tsne_embeddings_train.png')
plt.show()
print("Saved tSNE plot to tsne_embeddings_train.png")