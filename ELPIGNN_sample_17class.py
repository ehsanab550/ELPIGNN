"""
@Ehsanab

"""

# elpignn_multiclass_electrides.py
"""
ELPIGNN: Angle-aware GNN for Multi-Class Electride Classification (17 categories) 
"""

import os
import glob
import math
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import scatter

# Pymatgen for POSCAR parsing + PBC neighbor search
from pymatgen.core import Structure

# Metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, 
                            precision_recall_fscore_support, confusion_matrix, 
                            classification_report, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_for_nan(tensor, name=""):
    """Check for NaN values in tensor and report"""
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

# -----------------------------
# Plotting Functions
# -----------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='turbo', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Accuracy'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', 
                   color='green', linewidth=2)
    axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision, Recall, F1 (if available)
    if 'val_precision' in history and 'val_recall' in history and 'val_f1' in history:
        axes[1, 0].plot(history['val_precision'], label='Precision', linewidth=2)
        axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2)
        axes[1, 0].plot(history['val_f1'], label='F1-Score', linewidth=2)
        axes[1, 0].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate', color='purple', linewidth=2)
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(labels, class_names, save_path):
    """Plot and save class distribution"""
    plt.figure(figsize=(14, 8))
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create bar plot
    bars = plt.bar(range(len(unique)), counts, color=plt.cm.turbo(np.linspace(0, 1, len(unique))))
    
    plt.title('Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(count)}', ha='center', va='bottom', rotation=90, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_metrics(metrics, class_names, save_path):
    """Plot and save performance metrics per class"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    
    # Precision
    axes[0].bar(range(len(class_names)), metrics['precision'], 
                color=plt.cm.turbo(np.linspace(0, 1, len(class_names))))
    axes[0].set_title('Precision per Class', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim(0, 1.05)
    
    # Add values on top of bars
    for i, v in enumerate(metrics['precision']):
        axes[0].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', rotation=90, fontsize=9)
    
    # Recall
    axes[1].bar(range(len(class_names)), metrics['recall'], 
                color=plt.cm.turbo(np.linspace(0, 1, len(class_names))))
    axes[1].set_title('Recall per Class', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1.05)
    
    # Add values on top of bars
    for i, v in enumerate(metrics['recall']):
        axes[1].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', rotation=90, fontsize=9)
    
    # F1-Score
    axes[2].bar(range(len(class_names)), metrics['f1'], 
                color=plt.cm.turbo(np.linspace(0, 1, len(class_names))))
    axes[2].set_title('F1-Score per Class', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xlabel('Class')
    axes[2].set_xticks(range(len(class_names)))
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim(0, 1.05)
    
    # Add values on top of bars
    for i, v in enumerate(metrics['f1']):
        axes[2].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', rotation=90, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_score, class_names, save_path):
    """Plot and save ROC curves for each class"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize the output for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 10))
    colors = plt.cm.turbo(np.linspace(0, 1, len(class_names)))
    
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='{0} (AUC = {1:0.2f})'.format(class_names[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves for All Classes', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 10})
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_metrics(metrics, class_names, save_path):
    """Plot combined precision, recall, and F1 in a single plot"""
    plt.figure(figsize=(16, 10))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.bar(x - width, metrics['precision'], width, label='Precision', alpha=0.8)
    plt.bar(x, metrics['recall'], width, label='Recall', alpha=0.8)
    plt.bar(x + width, metrics['f1'], width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Performance Metrics by Class', fontsize=16, fontweight='bold')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (p, r, f) in enumerate(zip(metrics['precision'], metrics['recall'], metrics['f1'])):
        plt.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
        plt.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
        plt.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'combined_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# Data ingestion & graph building
# -----------------------------
@dataclass
class BuildConfig:
    cutoff: float = 5.0           # Å neighbor radius
    max_neighbors: int = 64       # limit per atom for stability
    add_self_loops: bool = False  # no self loops for distance angle logic

class ElementTable:
    """Maps element symbol → feature vector from ptable CSV."""
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        # Find symbol column
        symbol_col = None
        for c in ["symbol", "Symbol", "element", "Element", "el", "EL"]:
            if c in df.columns:
                symbol_col = c
                break
        if symbol_col is None:
            raise ValueError("Element CSV must contain a symbol column.")
        
        self.symbols = df[symbol_col].astype(str).str.strip().tolist()
        # Take all numeric columns except the symbol column
        num_df = df.drop(columns=[symbol_col])
        num_df = num_df.select_dtypes(include=[np.number])
        
        # Replace infinite values and NaN with column means
        num_df = num_df.replace([np.inf, -np.inf], np.nan)
        num_df = num_df.fillna(num_df.mean())
        
        # Normalize features using StandardScaler
        self.scaler = StandardScaler()
        self.feats = self.scaler.fit_transform(num_df.values.astype(np.float32))
        
        self.feature_names = list(num_df.columns)
        self.map: Dict[str, np.ndarray] = {s: self.feats[i] for i, s in enumerate(self.symbols)}
        self.dim = self.feats.shape[1]
        print(f"Element table loaded with {self.dim} features")

    def __call__(self, symbol: str) -> np.ndarray:
        if symbol not in self.map:
            # Return zero vector for unknown elements
            print(f"Warning: Element '{symbol}' not found in element table. Using zeros.")
            return np.zeros(self.dim, dtype=np.float32)
        return self.map[symbol]

class ElectrideDataset(Dataset):
    def __init__(
        self,
        ptable_csv: str,
        feature_csv: str,
        poscar_glob: str,
        build_cfg: BuildConfig,
    ):
        super().__init__()
        self.elem_table = ElementTable(ptable_csv)
        self.df = pd.read_csv(feature_csv)
        self.build_cfg = build_cfg

        # Clean the dataframe
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Identify columns
        assert 'third_dir' in self.df.columns, "'third_dir' column not found in features CSV"
        assert 'second_dir' in self.df.columns, "'second_dir' column not found in features CSV"
        self.id_col = 'third_dir'
        self.label_col = 'second_dir'

        # Map second_dir to class labels
        self.df['second_dir'] = self.df['second_dir'].str.replace('No_', 'NO_').str.replace('No_SrN', 'NO_SrN')
        
        # Get all unique class names
        self.class_names = sorted(self.df[self.label_col].unique())
        self.num_classes = len(self.class_names)
        print(f"Found {self.num_classes} classes: {self.class_names}")
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df[self.label_col])
        
        # Determine global feature columns
        exclude = {self.id_col, self.label_col, 'material_name', 'indicator', 'label_encoded'}
        self.global_cols = [c for c in self.df.columns if c not in exclude and pd.api.types.is_numeric_dtype(self.df[c])]
        self.global_dim = len(self.global_cols)
        print(f"Found {self.global_dim} global features")

        # Clean global features
        for col in self.global_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        # Map third_dir → POSCAR path
        self.poscar_paths = glob.glob(poscar_glob, recursive=True)
        if len(self.poscar_paths) == 0:
            raise FileNotFoundError(f"No POSCAR files found for pattern: {poscar_glob}")
        
        self.id_to_path = {}
        for p in self.poscar_paths:
            # Extract ID from path (assuming structure: .../ID/POSCAR)
            path_parts = p.split(os.sep)
            if len(path_parts) >= 2:
                parent_dir = path_parts[-2]  # Directory containing POSCAR
                self.id_to_path[parent_dir] = p
        
        # Filter out entries without POSCAR files
        self.valid_indices = []
        self.missing_ids = []
        for idx in range(len(self.df)):
            _id = str(self.df.iloc[idx][self.id_col]).strip()
            if _id in self.id_to_path:
                self.valid_indices.append(idx)
            else:
                self.missing_ids.append(_id)
        
        print(f"Found {len(self.valid_indices)} valid entries with POSCAR files")
        if self.missing_ids:
            print(f"Missing POSCAR for {len(self.missing_ids)} entries")

    def __len__(self):
        return len(self.valid_indices)

    def _structure_to_graph(self, struct: Structure) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build graph from structure with edges and angle triplets."""
        cfg = self.build_cfg
        N = len(struct.sites)

        # Node features
        x_list = []
        for site in struct.sites:
            symbol = site.specie.symbol
            x_list.append(self.elem_table(symbol))
        x = torch.tensor(np.vstack(x_list), dtype=torch.float32)

        # Positions (cartesian)
        pos = torch.tensor(struct.cart_coords, dtype=torch.float32)

        # Neighbor search with PBC
        edges = []  # directed (i->j)
        dists = []
        neighbor_info = [[] for _ in range(N)]  # store neighbor vectors per center j
        
        for j in range(N):
            site_j = struct.sites[j]
            try:
                neighs = struct.get_neighbors(site_j, cfg.cutoff)
                # Sort by distance then cap to max_neighbors
                neighs = sorted(neighs, key=lambda n: n.nn_distance)[:cfg.max_neighbors]
                for nb in neighs:
                    i = nb.index  # neighbor atom index i
                    if i == j:
                        continue
                    # Vector from j to i
                    vec_ji = nb.coords - site_j.coords
                    d = float(np.linalg.norm(vec_ji))
                    if d <= 1e-8:
                        continue
                    edges.append((i, j))
                    dists.append(d)
                    neighbor_info[j].append((i, vec_ji, d))
            except Exception as e:
                print(f"Error getting neighbors for atom {j}: {e}")
                continue
        
        if len(edges) == 0:
            # Fallback: create a minimal graph with self-loops if no edges found
            print("Warning: No edges found, creating fallback graph")
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edges.append((i, j))
                        dists.append(1.0)  # placeholder distance
                        if len(edges) >= cfg.max_neighbors * N:
                            break
                if len(edges) >= cfg.max_neighbors * N:
                    break

        # Build edge index and features
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(dists, dtype=torch.float32).unsqueeze(1)  # [E,1]

        # Add small epsilon to prevent division by zero
        edge_attr = edge_attr + 1e-8

        # Map (i->j) to edge id
        e2id = {(i, j): idx for idx, (i, j) in enumerate(edges)}

        # Build angle triplets for center j: i–j–k
        triplets_list = []
        for j in range(N):
            neighs = neighbor_info[j]
            L = len(neighs)
            if L < 2:
                continue
            for a in range(L):
                i, v_ij, d_ij = neighs[a]
                for b in range(a+1, L):  # Avoid duplicate pairs
                    k, v_kj, d_kj = neighs[b]
                    # Calculate angle at j between i and k
                    dot = float(np.dot(v_ij, v_kj))
                    norm = float(np.linalg.norm(v_ij) * np.linalg.norm(v_kj) + 1e-12)
                    cos_th = max(min(dot / norm, 1.0), -1.0)
                    sin_th = float(math.sqrt(max(0.0, 1.0 - cos_th * cos_th)))
                    
                    # Add both orderings for symmetry
                    if (i, j) in e2id and (k, j) in e2id:
                        e_ij = e2id[(i, j)]
                        e_kj = e2id[(k, j)]
                        triplets_list.append([e_ij, e_kj, cos_th, sin_th, d_ij, d_kj])
        
        if len(triplets_list) == 0:
            triplets = torch.zeros((0, 6), dtype=torch.float32)
        else:
            triplets = torch.tensor(triplets_list, dtype=torch.float32)
        
        return x, edge_index, edge_attr, pos, triplets

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        
        # Get the third_dir (ID) for this row
        third_dir = str(row[self.id_col]).strip()
        
        # Get the POSCAR path
        poscar_path = self.id_to_path.get(third_dir)
        if poscar_path is None:
            raise ValueError(f"POSCAR path not found for ID: {third_dir}")
        
        # Parse POSCAR
        try:
            struct = Structure.from_file(poscar_path)
        except Exception as e:
            raise ValueError(f"Error parsing POSCAR file {poscar_path}: {e}")
        
        # Build graph from structure
        x, edge_index, edge_attr, pos, triplets = self._structure_to_graph(struct)

        # Global features
        try:
            g_values = row[self.global_cols].values.astype(float)
            g = torch.tensor(g_values, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing global features for {third_dir}: {e}")
            g = torch.zeros(self.global_dim, dtype=torch.float32)

        # Label
        y = torch.tensor(row['label_encoded'], dtype=torch.long)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            triplets=triplets,
            global_feat=g,
            y=y,
            id=third_dir,
            original_label=row[self.label_col]
        )
        return data

# -----------------------------
# Batch Collator
# -----------------------------
class BatchCollator:
    """Custom collator to handle batching of graph data with angle triplets"""
    
    def __init__(self, scaler=None):
        self.scaler = scaler
    
    def __call__(self, batch):
        # Initialize lists
        xs, edge_indices, edge_attrs, ys, global_feats, triplets_list = [], [], [], [], [], []
        batch_assignments = []
        node_offset = 0
        edge_offset = 0
        
        for i, data in enumerate(batch):
            # Append data
            xs.append(data.x)
            edge_indices.append(data.edge_index + node_offset)
            edge_attrs.append(data.edge_attr)
            ys.append(data.y)
            global_feats.append(data.global_feat)
            
            # Adjust triplet indices for batching
            if data.triplets.numel() > 0:
                adjusted_triplets = data.triplets.clone()
                adjusted_triplets[:, 0] += edge_offset  # e_ij
                adjusted_triplets[:, 1] += edge_offset  # e_kj
                triplets_list.append(adjusted_triplets)
            
            # Track assignments
            num_nodes = data.x.size(0)
            batch_assignments.append(torch.full((num_nodes,), i, dtype=torch.long))
            
            # Update offsets
            node_offset += num_nodes
            edge_offset += data.edge_index.size(1)
        
        # Concatenate all data
        x = torch.cat(xs, dim=0)
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        batch_assignments = torch.cat(batch_assignments, dim=0)
        
        # Handle triplets
        if triplets_list:
            triplets = torch.cat(triplets_list, dim=0)
        else:
            triplets = torch.zeros((0, 6), dtype=torch.float32)
        
        # Process global features
        global_feat = torch.stack(global_feats, dim=0)
        if self.scaler is not None:
            global_feat_numpy = global_feat.numpy()
            global_feat_std = self.scaler.transform(global_feat_numpy)
            global_feat = torch.tensor(global_feat_std, dtype=torch.float32)
        
        # Create the batched Data object
        batched_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            triplets=triplets,
            global_feat=global_feat,
            batch=batch_assignments
        )
        
        return batched_data

# -----------------------------
# Model Components
# -----------------------------
class AngleBlock(nn.Module):
    """Angle-aware edge update block."""
    def __init__(self, edge_dim: int, hidden: int):
        super().__init__()
        self.mlp_phi = nn.Sequential(
            nn.Linear(2*edge_dim + 4, hidden), 
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), 
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.mlp_psi = nn.Sequential(
            nn.Linear(hidden, edge_dim),
        )
        self.norm = nn.LayerNorm(edge_dim)

    def forward(self, edge_feat: torch.Tensor, triplets: torch.Tensor) -> torch.Tensor:
        if triplets.numel() == 0:
            return edge_feat
        
        e_ij = triplets[:, 0].long()
        e_kj = triplets[:, 1].long()
        cos = triplets[:, 2:3]
        sin = triplets[:, 3:4]
        d_ij = triplets[:, 4:5]
        d_kj = triplets[:, 5:6]
        
        f_ij = edge_feat[e_ij]
        f_kj = edge_feat[e_kj]
        
        # Check for NaN in inputs
        if check_for_nan(f_ij, "f_ij") or check_for_nan(f_kj, "f_kj"):
            return edge_feat
            
        m = self.mlp_phi(torch.cat([f_ij, f_kj, cos, sin, d_ij, d_kj], dim=1))
        agg = scatter(m, e_ij, dim=0, dim_size=edge_feat.size(0), reduce='mean')
        out = edge_feat + self.mlp_psi(agg)
        return self.norm(out)

class EdgeInit(nn.Module):
    def __init__(self, in_edge: int, edge_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_edge, edge_dim), 
            nn.SiLU(),
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, edge_dim), 
            nn.SiLU(),
            nn.LayerNorm(edge_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_attr)

class NodeUpdate(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__(aggr='mean', flow='source_to_target')
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden), 
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), 
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.node_upd = nn.Sequential(
            nn.Linear(node_dim + hidden, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_feat):
        m = self.propagate(edge_index, x=x, edge_feat=edge_feat)
        out = torch.cat([x, m], dim=1)
        out = x + self.node_upd(out)
        return self.norm(out)

    def message(self, x_j, edge_feat):
        return self.edge_mlp(torch.cat([x_j, edge_feat], dim=1))

class ELPIGNN(nn.Module):
    def __init__(self,
                 node_dim,
                 global_dim,
                 num_classes,
                 edge_in=1,
                 edge_dim=64,
                 node_hidden=128,
                 n_layers=3,
                 mlp_hidden=64,
                 dropout=0.2):
        super().__init__()

        # Save config
        self.hidden_dim = node_hidden
        self.global_dim = global_dim
        self.num_classes = num_classes

        # Node embedding
        self.node_emb = nn.Sequential(
            nn.Linear(node_dim, node_hidden),
            nn.LayerNorm(node_hidden),
            nn.SiLU(),
            nn.Dropout(dropout/2),
        )

        # Edge embedding
        self.edge_init = EdgeInit(edge_in, edge_dim)

        # Angle-aware blocks
        self.angle_blocks = nn.ModuleList([
            AngleBlock(edge_dim, edge_dim) for _ in range(n_layers)
        ])
        
        # Node update blocks
        self.node_blocks = nn.ModuleList([
            NodeUpdate(node_hidden, edge_dim, node_hidden) for _ in range(n_layers)
        ])
        
        # Classifier MLP
        cls_in_dim = node_hidden + global_dim
        self.cls = nn.Sequential(
            nn.Linear(cls_in_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.LayerNorm(mlp_hidden//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, trip = data.x, data.edge_index, data.edge_attr, data.triplets
        g = data.global_feat

        # Check for NaN in inputs
        if check_for_nan(x, "input x") or check_for_nan(edge_attr, "input edge_attr") or check_for_nan(g, "input g"):
            # Return zeros if NaN detected
            return torch.zeros(data.batch.max().item() + 1, self.num_classes, device=x.device)

        # Initialize embeddings
        e = self.edge_init(edge_attr)
        x = self.node_emb(x)

        # Apply angle-aware message passing
        for angle_block, node_block in zip(self.angle_blocks, self.node_blocks):
            e = angle_block(e, trip)
            x = node_block(x, edge_index, e)

        # Graph pooling
        h = global_mean_pool(x, data.batch)

        # Ensure global features have the right shape
        if g.dim() == 1:
            g = g.unsqueeze(0)
        
        # Concatenate with global features
        hg = torch.cat([h, g], dim=1)
        logits = self.cls(hg)

        return logits

# -----------------------------
# Training & evaluation
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 30
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_fold(model, train_loader, val_loader, cfg: TrainConfig, fold_id: int, results_dir: str):
    device = cfg.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1
    best_state = None
    no_improve = 0
    best_epoch = 0
    
    # Track training history
    train_losses = []
    val_accuracies = []
    val_losses = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # Check for NaN in input
            if check_for_nan(batch.x, "batch.x") or check_for_nan(batch.edge_attr, "batch.edge_attr") or \
               check_for_nan(batch.global_feat, "batch.global_feat"):
                print(f"Skipping batch {batch_idx} due to NaN in input")
                continue
                
            logits = model(batch)
            
            # Check for NaN in output
            if check_for_nan(logits, "logits"):
                print(f"Skipping batch {batch_idx} due to NaN in output")
                continue
                
            loss = criterion(logits, batch.y)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"Skipping batch {batch_idx} due to NaN loss")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            tr_losses.append(loss.item())

        # Evaluation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        # Calculate additional metrics
        avg_train_loss = np.mean(tr_losses) if tr_losses else float('nan')
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        # Calculate precision, recall, f1
        val_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # Track history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
            best_epoch = epoch
        else:
            no_improve += 1
        
        print(f"Fold {fold_id} | Epoch {epoch:03d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_acc={val_accuracy:.4f} | best_acc={best_accuracy:.4f}@{best_epoch}")
        
        if no_improve >= cfg.patience:
            print("Early stopping.")
            break

    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'val_precision': val_precisions,
        'val_recall': val_recalls,
        'val_f1': val_f1s
    }
    
    history_path = os.path.join(results_dir, f"fold_{fold_id}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_accuracy, history

def evaluate(model, loader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            
            _, predicted = torch.max(logits.data, 1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(probs.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec, rec, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate ROC AUC for each class (one-vs-rest)
    roc_auc = {}
    if len(class_names) > 2:
        for i, class_name in enumerate(class_names):
            try:
                roc_auc[class_name] = roc_auc_score((y_true == i).astype(int), y_score[:, i])
            except ValueError:
                roc_auc[class_name] = float('nan')
    else:
        roc_auc["macro"] = roc_auc_score(y_true, y_score[:, 1])
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'precision_per_class': prec,
        'recall_per_class': rec,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score
    }

# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    import types
    from datetime import datetime
    
    # Check if we're running in a Jupyter notebook
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except:
        in_notebook = False
    
    if in_notebook:
        # Use default values for Jupyter notebook
        args = types.SimpleNamespace()
        args.ptable = "/data/ptable_final_cleaned2.csv"
        args.features = "/data/PI_electride_featursfinal2.csv"
        args.poscar_glob = "/ele_PO&OSZ/*/separatePOSCAR/*/*/POSCAR"
        args.cutoff = 5.0
        args.max_neighbors = 64
        args.epochs = 200
        args.batch_size = 8
        args.lr = 1e-4
        args.weight_decay = 1e-5
        args.patience = 30
        args.n_layers = 3
        args.edge_dim = 64
        args.node_hidden = 128
        args.mlp_hidden = 64
        args.dropout = 0.2
        args.save_dir = './checkpoints'
        args.results_dir = './plot_eleEX3'
        args.seed = 42
    else:
        # Parse command-line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--ptable', type=str, required=True)
        parser.add_argument('--features', type=str, required=True)
        parser.add_argument('--poscar_glob', type=str, required=True)
        parser.add_argument('--cutoff', type=float, default=5.0)
        parser.add_argument('--max_neighbors', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--patience', type=int, default=30)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--edge_dim', type=int, default=64)
        parser.add_argument('--node_hidden', type=int, default=128)
        parser.add_argument('--mlp_hidden', type=int, default=64)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--save_dir', type=str, default='./checkpoints')
        parser.add_argument('--results_dir', type=str, default='./plot_eleEX3')
        parser.add_argument('--seed', type=int, default=42)
        args = parser.parse_args()

    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create dataset
    build_cfg = BuildConfig(cutoff=args.cutoff, max_neighbors=args.max_neighbors)
    dataset = ElectrideDataset(args.ptable, args.features, args.poscar_glob, build_cfg)

    # Prepare global feature standardization
    all_global_feats = []
    for i in range(len(dataset)):
        data = dataset[i]
        all_global_feats.append(data.global_feat.numpy())
    
    all_global_feats = np.vstack(all_global_feats)
    scaler = StandardScaler()
    scaler.fit(all_global_feats)

    # Build stratified folds
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_score = []
    all_histories = []
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n========== Fold {fold} ==========")
        
        # Create data loaders
        train_subset = torch.utils.data.Subset(dataset, tr_idx.tolist())
        val_subset = torch.utils.data.Subset(dataset, va_idx.tolist())

        train_collator = BatchCollator(scaler=scaler)
        val_collator = BatchCollator(scaler=scaler)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collator)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=val_collator)

        # Model
        model = ELPIGNN(
            node_dim=dataset.elem_table.dim,
            global_dim=dataset.global_dim,
            num_classes=dataset.num_classes,
            edge_in=1,
            edge_dim=args.edge_dim,
            node_hidden=args.node_hidden,
            n_layers=args.n_layers,
            mlp_hidden=args.mlp_hidden,
            dropout=args.dropout,
        )
        
        tcfg = TrainConfig(
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            lr=args.lr, 
            weight_decay=args.weight_decay, 
            patience=args.patience
        )
        
        model, best_acc, history = train_one_fold(model, train_loader, val_loader, tcfg, fold, args.results_dir)
        all_histories.append(history)

        # Save model
        ckpt = os.path.join(args.save_dir, f"fold_{fold}.pt")
        torch.save({
            'state_dict': model.state_dict(),
            'config': vars(args),
            'fold': fold,
            'best_val_accuracy': best_acc,
            'class_names': dataset.class_names,
        }, ckpt)
        print(f"Saved best model to {ckpt}")

        # Evaluation
        metrics = evaluate(model, val_loader, tcfg.device, dataset.class_names)
        print(f"Fold {fold} accuracy: {metrics['accuracy']:.4f}")
        print(f"Fold {fold} weighted F1: {metrics['f1_weighted']:.4f}")
        
        # Store results for overall evaluation
        fold_metrics.append(metrics)
        all_y_true.extend(metrics['y_true'])
        all_y_pred.extend(metrics['y_pred'])
        all_y_score.extend(metrics['y_score'])
        
        # Save fold results
        fold_results = {
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'precision_per_class': metrics['precision_per_class'].tolist(),
            'recall_per_class': metrics['recall_per_class'].tolist(),
            'f1_per_class': metrics['f1_per_class'].tolist(),
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
        
        results_path = os.path.join(args.results_dir, f"fold_{fold}_results.json")
        with open(results_path, 'w') as f:
            json.dump(fold_results, f)

    # Overall evaluation
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)
    
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
    overall_cm = confusion_matrix(all_y_true, all_y_pred)
    
    # Calculate ROC AUC for each class
    overall_roc_auc = {}
    for i, class_name in enumerate(dataset.class_names):
        try:
            overall_roc_auc[class_name] = roc_auc_score((all_y_true == i).astype(int), all_y_score[:, i])
        except ValueError:
            overall_roc_auc[class_name] = float('nan')
    
    # Calculate precision, recall, f1 per class
    prec_per_class = precision_score(all_y_true, all_y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(all_y_true, all_y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(all_y_true, all_y_pred, average=None, zero_division=0)
    
    # Save overall results
    overall_results = {
        'accuracy': overall_accuracy,
        'f1_weighted': overall_f1,
        'confusion_matrix': overall_cm.tolist(),
        'roc_auc': overall_roc_auc,
        'class_names': dataset.class_names,
        'precision_per_class': prec_per_class.tolist(),
        'recall_per_class': rec_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'classification_report': classification_report(all_y_true, all_y_pred, target_names=dataset.class_names, output_dict=True)
    }
    
    overall_path = os.path.join(args.results_dir, "overall_results.json")
    with open(overall_path, 'w') as f:
        json.dump(overall_results, f)
    
    # Print overall metrics
    print("\n===== Overall Results =====")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Weighted F1: {overall_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=dataset.class_names))
    
    print("\nROC AUC per class:")
    for class_name, auc in overall_roc_auc.items():
        print(f"{class_name}: {auc:.4f}")

    # -----------------------------
    # Plotting Section
    # -----------------------------
    print("\nGenerating plots...")
    
    # Create a subdirectory for plots
    plot_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plot_confusion_matrix(all_y_true, all_y_pred, dataset.class_names, plot_dir)
    
    # 2. Class Distribution
    plot_class_distribution([dataset[i].y.item() for i in range(len(dataset))], dataset.class_names, plot_dir)
    
    # 3. Performance Metrics
    metrics_dict = {
        'precision': prec_per_class,
        'recall': rec_per_class,
        'f1': f1_per_class
    }
    plot_performance_metrics(metrics_dict, dataset.class_names, plot_dir)
    
    # 4. Combined Metrics
    plot_combined_metrics(metrics_dict, dataset.class_names, plot_dir)
    
    # 5. ROC Curves
    plot_roc_curves(all_y_true, all_y_score, dataset.class_names, plot_dir)
    
    # 6. Training History for each fold
    for fold, history in enumerate(all_histories, start=1):
        plot_training_history(history, os.path.join(plot_dir, f"fold_{fold}_training_history.png"))
    
    # 7. Average Training History across folds
    if all_histories:
        avg_history = {}
        for key in all_histories[0].keys():
            # Find the minimum length among all folds for this metric
            min_len = min(len(h[key]) for h in all_histories)
            # Truncate all histories to the minimum length and average
            truncated = [h[key][:min_len] for h in all_histories]
            avg_history[key] = np.mean(truncated, axis=0).tolist()
        
        plot_training_history(avg_history, os.path.join(plot_dir, "average_training_history.png"))
    
    print(f"All plots saved to {plot_dir}")

if __name__ == '__main__':

    main()
