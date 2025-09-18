# 🧠 Analyzing Brain Neural Graphs with Graph Neural Networks  
*A Master’s Thesis by Hari Priya Dhanasekaran – University of Washington Bothell (2025)*

---

## 📌 Overview
Spontaneous **bursting activity** is a hallmark of cortical networks—but the *mechanisms that trigger bursts* remain elusive.  
This project builds an **end-to-end machine learning pipeline** that integrates spiking activity and synaptic connectivity using **Graph Neural Networks (GNNs)** to uncover hidden patterns preceding burst initiation.

---

## 🚀 Pipeline

1. **Simulate & Extract Data**  
   - 10,000-neuron networks grown in the **Graphitti simulator**  
   - Output: ≈ 4 TB HDF5 spike trains + synaptic weights

2. **Subgraph Construction**  
   - 2-hop neighborhoods centered on burst origins  
   - Node features: mean ISI, ISI entropy, firing rate, last-spike lag

3. **Train a GCN Model**  
   - 3-layer Graph Convolutional Network (PyTorch Geometric)  
   - Binary classification: *pre-burst* vs *non-burst* subgraphs  
   - Achieved **F1 ≈ 0.996**
     
     <img width="1079" height="411" alt="image" src="https://github.com/user-attachments/assets/011694c2-f281-411d-9012-e8b2d192ef61" />

4. **Explainability**  
   - Applied **GNNExplainer** and **PGExplainer**  
   - Visualized salient neurons/edges in true spatial coordinates

5. **Validation**  
   - Automated ablation suite (label shuffle, node shuffle, no edges, MLP baseline)  
   - Confirmed both **topology + spike features** are critical

---

## 🔬 Key Findings
- **Two robust motifs** emerged in pre-burst graphs:  
  - 🌀 *Local Hub* – densely connected salient neurons near the burst origin  
  - 🔗 *Remote Ring* – ring-like structures of salient nodes ~2 hops away  
- Burst initiation is **distributed**, not localized—challenging origin-centric models.  
- Ablation shows:  
  - Features alone → ~95% accuracy  
  - Features + topology → **99.6% accuracy**

---

## 📊 Results Snapshot
<img width="558" height="537" alt="image" src="https://github.com/user-attachments/assets/43bf2eb3-9cff-4d1b-af69-4171e9b59946" />

<img width="477" height="477" alt="image" src="https://github.com/user-attachments/assets/a49d6b4d-6f50-4236-8fe8-3a7927a78be2" />
  
Example: Saliency maps of pre-burst motifs (Local Hub vs Remote Ring)

---

## 🛠️ Tech Stack
- **Simulation**: Graphitti (GPU-enabled LIF model)  
- **Frameworks**: PyTorch, PyTorch Geometric, NetworkX  
- **Explainability**: GNNExplainer, PGExplainer  
- **Visualization**: Matplotlib, t-SNE  

---

## 📖 Citation
```bibtex
@mastersthesis{dhanasekaran2025gnn,
  author      = {Hari Priya Dhanasekaran},
  title       = {Analyzing Brain Neural Graphs with Graph Neural Networks},
  school      = {University of Washington Bothell},
  year        = {2025},
  url         = {https://github.com/priyadhanu14/GNN-Spiking-Neural-Networks}
}
