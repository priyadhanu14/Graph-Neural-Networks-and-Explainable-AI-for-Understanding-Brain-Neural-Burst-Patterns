# Analyzing Brain Neural Graphs with Graph Neural Networks  
*A thesis project by Hari Priya Dhanasekaran – University of Washington Bothell*

---

## ✨ Overview
Understanding burst dynamics in large-scale spiking‐neural networks  
is hard with classical spike-train statistics alone.  
This thesis builds an **end-to-end ML pipeline** that

1. **Extracts** 10 k × 10 k-node connectivity graphs + spike trains from the
   *Graphitti* simulator (≈ 4 TB raw HDF5).
2. **Generates** sub-graphs around burst origins and engineers biologically
   motivated features (mean ISI, entropy, spike count, last-spike lag, …).
3. **Trains** a 3-layer **GCN** to classify *pre-burst* vs *non-burst*
   sub-graphs – reaching **F1 ≈ 0.99** on 5 378 unseen graphs.
4. **Explains** predictions with **PGExplainer / GNNExplainer** and
   visualises node & edge saliency on true *(x,y)* neuron positions.
5. **Validates** with an automated ablation suite
   (label-shuffle, edge randomisation, degree-preserving shuffles, LOSO CV).

> **Goal:** reveal repeated local motifs that consistently precede network
> bursts and may act as triggers.

---
@mastersthesis{dhanasekaran2025gnn,
  author      = {Hari Priya Dhanasekaran},
  title       = {Analyzing Brain Neural Graphs with Graph Neural Networks},
  school      = {University of Washington Bothell},
  year        = {2025},
  url         = {https://github.com/priyadhanu14/GNN-Spiking-Neural-Networks}
}

