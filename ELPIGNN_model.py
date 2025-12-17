"""
@author: Ehsanab
"""

"""
ELPIGNN Model:  Graph Neural Network with Angle-aware Message Passing
Handles POSCAR to graph conversion with element table embeddings and angle triplets
"""

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import scatter

# Pymatgen for POSCAR parsing + PBC neighbor search
from pymatgen.core import Structure

# -----------------------------
# Configuration & Utils
# -----------------------------

@dataclass
class BuildConfig:
    cutoff: float = 5.0           # Å neighbor radius
    max_neighbors: int = 64       # limit per atom for stability
    add_self_loops: bool = False  # no self loops for distance angle logic

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
# Element Table & Graph Construction
# -----------------------------

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
        from sklearn.preprocessing import StandardScaler
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

class GraphBuilder:
    """Handles conversion from POSCAR files to graph data with angle triplets"""
    
    def __init__(self, ptable_csv: str, build_cfg: BuildConfig):
        self.elem_table = ElementTable(ptable_csv)
        self.build_cfg = build_cfg
    
    def structure_to_graph(self, struct: Structure) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

# -----------------------------
# ELPIGNN Model Components
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
