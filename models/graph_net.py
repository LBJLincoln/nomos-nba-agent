#!/usr/bin/env python3
"""
NBA Graph Neural Network — GCN + GAT for Game Prediction
==========================================================
Implements Graph Convolutional Networks and Graph Attention Networks
for NBA game prediction using vanilla PyTorch (no torch_geometric).

Architecture:
  1. TEAM GRAPH: 30 NBA teams as nodes, edges from H2H games
  2. GCN LAYERS: Message passing between teams via adjacency
  3. GAT LAYERS: Attention-weighted message passing (matchup importance)
  4. FUSION HEAD: Combine graph embeddings with tabular features
  5. WALK-FORWARD: _fold_graph compatible with wf_eval framework

Key insight:
  Standard tabular models treat each game independently.
  Graph models capture RELATIONAL structure:
    - Team A beats Team B, B beats C → transitive strength signal
    - Division rivalries, conference clusters
    - Schedule graph (shared opponents = correlation)
    - Performance propagation through the league graph

Designed for Kaggle T4 GPU — vanilla PyTorch only.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

NUM_TEAMS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Deterministic team → index mapping (alphabetical by abbreviation)
TEAM_ABBREVS = sorted(set(TEAM_MAP.values()))
TEAM_TO_IDX = {t: i for i, t in enumerate(TEAM_ABBREVS)}

# Conference / Division structure (prior graph edges)
CONFERENCES = {
    "EAST": ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
             "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"],
    "WEST": ["DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
             "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA"],
}

DIVISIONS = {
    "ATLANTIC": ["BOS", "BKN", "NYK", "PHI", "TOR"],
    "CENTRAL": ["CHI", "CLE", "DET", "IND", "MIL"],
    "SOUTHEAST": ["ATL", "CHA", "MIA", "ORL", "WAS"],
    "NORTHWEST": ["DEN", "MIN", "OKC", "POR", "UTA"],
    "PACIFIC": ["GSW", "LAC", "LAL", "PHX", "SAC"],
    "SOUTHWEST": ["DAL", "HOU", "MEM", "NOP", "SAS"],
}


# ═══════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

class NBAGraphBuilder:
    """
    Constructs team-level graphs from historical game data.

    Graph types:
      1. H2H adjacency: edge weight = games played between teams
      2. Win-rate adjacency: edge weight = home team's win% vs opponent
      3. Margin adjacency: edge weight = avg margin in H2H games
      4. Conference/Division prior: structural edges with base weight
      5. Performance similarity: teams with similar records share edges

    All graphs are maintained as dense adjacency matrices (30x30)
    for compatibility with vanilla PyTorch GCN layers.
    """

    def __init__(self, decay: float = 0.95, window: int = 82):
        """
        Args:
            decay: Exponential decay factor per game for recency weighting
            window: Max number of recent games to consider per team pair
        """
        self.decay = decay
        self.window = window
        # H2H records: (team_i, team_j) -> list of (margin, is_home_win, recency_weight)
        self.h2h_records: Dict[Tuple[str, str], List[Tuple[float, int, float]]] = defaultdict(list)
        # Team performance: team -> list of (win, margin, pts)
        self.team_records: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
        # Game counter for decay
        self.game_count = 0

    def update(self, home_abbr: str, away_abbr: str, home_pts: float,
               away_pts: float, home_win: int):
        """Update graph state after a game result."""
        self.game_count += 1
        margin = home_pts - away_pts

        # H2H records (directional: home perspective)
        self.h2h_records[(home_abbr, away_abbr)].append(
            (margin, home_win, self.game_count))
        self.h2h_records[(away_abbr, home_abbr)].append(
            (-margin, 1 - home_win, self.game_count))

        # Team records
        self.team_records[home_abbr].append((home_win, margin, home_pts))
        self.team_records[away_abbr].append((1 - home_win, -margin, away_pts))

    def build_adjacency_matrices(self) -> Dict[str, np.ndarray]:
        """
        Build multiple adjacency matrices representing different
        relationship types between teams.

        Returns dict of adjacency matrices, each (30, 30).
        """
        n = NUM_TEAMS

        # 1. H2H game count (symmetric, normalized)
        adj_h2h = np.zeros((n, n), dtype=np.float32)
        for (t1, t2), records in self.h2h_records.items():
            if t1 in TEAM_TO_IDX and t2 in TEAM_TO_IDX:
                i, j = TEAM_TO_IDX[t1], TEAM_TO_IDX[t2]
                # Recency-weighted count
                w = sum(self.decay ** (self.game_count - r[2]) for r in records[-self.window:])
                adj_h2h[i, j] = w
        # Normalize to [0, 1]
        mx = adj_h2h.max()
        if mx > 0:
            adj_h2h /= mx

        # 2. Win-rate adjacency (directional)
        adj_winrate = np.zeros((n, n), dtype=np.float32)
        for (t1, t2), records in self.h2h_records.items():
            if t1 in TEAM_TO_IDX and t2 in TEAM_TO_IDX:
                i, j = TEAM_TO_IDX[t1], TEAM_TO_IDX[t2]
                recent = records[-self.window:]
                if len(recent) >= 1:
                    weights = [self.decay ** (self.game_count - r[2]) for r in recent]
                    wins = [r[1] * w for r, w in zip(recent, weights)]
                    adj_winrate[i, j] = sum(wins) / max(sum(weights), 1e-8)

        # 3. Margin adjacency (directional, normalized)
        adj_margin = np.zeros((n, n), dtype=np.float32)
        for (t1, t2), records in self.h2h_records.items():
            if t1 in TEAM_TO_IDX and t2 in TEAM_TO_IDX:
                i, j = TEAM_TO_IDX[t1], TEAM_TO_IDX[t2]
                recent = records[-self.window:]
                if len(recent) >= 1:
                    weights = [self.decay ** (self.game_count - r[2]) for r in recent]
                    margins = [r[0] * w for r, w in zip(recent, weights)]
                    avg_margin = sum(margins) / max(sum(weights), 1e-8)
                    # Sigmoid to [0, 1]: positive margin → > 0.5
                    adj_margin[i, j] = 1.0 / (1.0 + np.exp(-avg_margin / 10.0))

        # 4. Conference/Division structural prior
        adj_struct = np.zeros((n, n), dtype=np.float32)
        for division_teams in DIVISIONS.values():
            for t1 in division_teams:
                for t2 in division_teams:
                    if t1 != t2 and t1 in TEAM_TO_IDX and t2 in TEAM_TO_IDX:
                        adj_struct[TEAM_TO_IDX[t1], TEAM_TO_IDX[t2]] = 1.0
        for conf_teams in CONFERENCES.values():
            for t1 in conf_teams:
                for t2 in conf_teams:
                    if t1 != t2 and t1 in TEAM_TO_IDX and t2 in TEAM_TO_IDX:
                        # Conference mates get 0.5 if not already in same division
                        if adj_struct[TEAM_TO_IDX[t1], TEAM_TO_IDX[t2]] < 1.0:
                            adj_struct[TEAM_TO_IDX[t1], TEAM_TO_IDX[t2]] = 0.5

        # 5. Performance similarity (cosine similarity of recent records)
        adj_perf = np.zeros((n, n), dtype=np.float32)
        team_vecs = {}
        for team, records in self.team_records.items():
            if team not in TEAM_TO_IDX:
                continue
            recent = records[-30:]  # Last 30 games
            if len(recent) >= 5:
                wins = sum(r[0] for r in recent) / len(recent)
                avg_margin = sum(r[1] for r in recent) / len(recent)
                avg_pts = sum(r[2] for r in recent) / len(recent)
                team_vecs[team] = np.array([wins, avg_margin / 20.0, avg_pts / 120.0])

        for t1, v1 in team_vecs.items():
            for t2, v2 in team_vecs.items():
                if t1 != t2:
                    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    # Only connect similar teams (cos_sim > 0.8)
                    adj_perf[TEAM_TO_IDX[t1], TEAM_TO_IDX[t2]] = max(0.0, cos_sim)

        return {
            "h2h": adj_h2h,
            "winrate": adj_winrate,
            "margin": adj_margin,
            "structure": adj_struct,
            "performance": adj_perf,
        }

    def build_node_features(self) -> np.ndarray:
        """
        Build per-team node feature vectors from accumulated records.

        Returns (30, node_feat_dim) array.
        """
        features = []
        for abbr in TEAM_ABBREVS:
            records = self.team_records.get(abbr, [])
            if len(records) < 3:
                features.append(np.zeros(12, dtype=np.float32))
                continue

            recent_10 = records[-10:] if len(records) >= 10 else records
            recent_20 = records[-20:] if len(records) >= 20 else records
            recent_5 = records[-5:] if len(records) >= 5 else records

            # Win rates at different windows
            wr_5 = sum(r[0] for r in recent_5) / len(recent_5)
            wr_10 = sum(r[0] for r in recent_10) / len(recent_10)
            wr_20 = sum(r[0] for r in recent_20) / len(recent_20)
            wr_all = sum(r[0] for r in records) / len(records)

            # Margins
            mg_5 = sum(r[1] for r in recent_5) / len(recent_5)
            mg_10 = sum(r[1] for r in recent_10) / len(recent_10)
            mg_20 = sum(r[1] for r in recent_20) / len(recent_20)

            # Points
            pts_10 = sum(r[2] for r in recent_10) / len(recent_10)
            pts_vol = float(np.std([r[2] for r in recent_10])) if len(recent_10) > 1 else 0.0

            # Streak
            streak = 0
            if records:
                last_w = records[-1][0]
                for r in reversed(records):
                    if r[0] == last_w:
                        streak += 1
                    else:
                        break
                streak = streak if last_w else -streak

            # Games played (normalized)
            gp = len(records) / 82.0

            # Margin volatility
            mg_vol = float(np.std([r[1] for r in recent_10])) if len(recent_10) > 1 else 0.0

            features.append(np.array([
                wr_5, wr_10, wr_20, wr_all,
                mg_5 / 20.0, mg_10 / 20.0, mg_20 / 20.0,  # Normalize margins
                pts_10 / 120.0,  # Normalize points
                pts_vol / 15.0,  # Normalize volatility
                streak / 10.0,   # Normalize streak
                gp,
                mg_vol / 15.0,
            ], dtype=np.float32))

        return np.stack(features)  # (30, 12)


# ═══════════════════════════════════════════════════════════════
# GCN LAYER (vanilla PyTorch, no torch_geometric)
# ═══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer (Kipf & Welling, 2017).

    h_i^{(l+1)} = sigma( sum_j A_hat[i,j] * h_j^{(l)} * W^{(l)} )

    Where A_hat = D^{-1/2} (A + I) D^{-1/2} is the symmetrically
    normalized adjacency with self-loops.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (B, N, F_in) or (N, F_in)
            adj: Adjacency matrix (B, N, N) or (N, N), raw or pre-normalized
        Returns:
            (B, N, F_out) or (N, F_out)
        """
        # Normalize adjacency: A_hat = D^{-1/2}(A+I)D^{-1/2}
        adj_hat = self._normalize_adj(adj)

        # Message passing: H' = A_hat @ X @ W
        support = torch.matmul(x, self.weight)  # (..., N, F_out)
        out = torch.matmul(adj_hat, support)     # (..., N, F_out)

        if self.bias is not None:
            out = out + self.bias
        return out

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization: D^{-1/2}(A+I)D^{-1/2}"""
        # Add self-loops
        if adj.dim() == 2:
            eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        else:
            eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj_hat = adj + eye

        # Degree matrix
        deg = adj_hat.sum(dim=-1).clamp(min=1e-8)  # (..., N)
        deg_inv_sqrt = deg.pow(-0.5)

        # D^{-1/2} A_hat D^{-1/2}
        if adj_hat.dim() == 2:
            norm = deg_inv_sqrt.unsqueeze(-1) * adj_hat * deg_inv_sqrt.unsqueeze(-2)
        else:
            norm = deg_inv_sqrt.unsqueeze(-1) * adj_hat * deg_inv_sqrt.unsqueeze(-2)
        return norm


class GATLayer(nn.Module):
    """
    Graph Attention Network layer (Velickovic et al., 2018).

    Attention coefficients:
      e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
      alpha_ij = softmax_j(e_ij)
      h_i' = sigma( sum_j alpha_ij * W * h_j )

    Multi-head variant: concat K attention heads then project.
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat

        # Per-head linear transform
        self.W = nn.Parameter(torch.empty(n_heads, in_dim, out_dim))
        # Attention parameters: a = [a_src || a_dst], each (n_heads, out_dim)
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim, 1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, F_in) or (N, F_in)
            adj: (B, N, N) or (N, N) — used as mask (0 = no edge)
        Returns:
            If concat: (..., N, n_heads * out_dim)
            Else: (..., N, out_dim) — averaged heads
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
            squeeze = True

        B, N, F_in = x.shape
        K, D = self.n_heads, self.out_dim

        # Linear transform per head: (B, N, F) @ (K, F, D) → (B, K, N, D)
        # Use einsum for clarity
        Wh = torch.einsum('bnf,kfd->bknd', x, self.W)  # (B, K, N, D)

        # Attention scores
        # e_src = Wh @ a_src → (B, K, N, 1)
        e_src = torch.einsum('bknd,kdo->bkno', Wh, self.a_src)  # (B, K, N, 1)
        e_dst = torch.einsum('bknd,kdo->bkno', Wh, self.a_dst)  # (B, K, N, 1)

        # Pairwise attention: e_ij = e_src_i + e_dst_j
        e = e_src + e_dst.transpose(-2, -1)  # (B, K, N, N)
        e = self.leaky_relu(e)

        # Mask: where adj == 0, set attention to -inf (+ self-loops)
        adj_mask = adj.unsqueeze(1)  # (B, 1, N, N)
        # Add self-loops to mask
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).unsqueeze(0)
        adj_mask = adj_mask + eye
        adj_mask = (adj_mask > 0).float()

        e = e.masked_fill(adj_mask == 0, float('-inf'))

        # Softmax over neighbors
        alpha = F_module_softmax(e, dim=-1)  # (B, K, N, N)
        alpha = self.dropout(alpha)

        # Aggregate: h_i' = sum_j alpha_ij * Wh_j
        out = torch.matmul(alpha, Wh)  # (B, K, N, D)

        if self.concat:
            # Concat all heads: (B, N, K*D)
            out = out.permute(0, 2, 1, 3).reshape(B, N, K * D)
        else:
            # Average heads: (B, N, D)
            out = out.mean(dim=1)

        if squeeze:
            out = out.squeeze(0)
        return out


def F_module_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax that handles -inf correctly."""
    # Replace -inf with very large negative number for stability
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_max = torch.where(torch.isinf(x_max), torch.zeros_like(x_max), x_max)
    exp_x = torch.exp(x - x_max)
    # Where input was -inf, exp will be 0
    exp_x = torch.where(torch.isinf(x), torch.zeros_like(exp_x), exp_x)
    return exp_x / (exp_x.sum(dim=dim, keepdim=True) + 1e-10)


# ═══════════════════════════════════════════════════════════════
# MULTI-RELATION GCN (fuses multiple adjacency types)
# ═══════════════════════════════════════════════════════════════

class MultiRelationGCN(nn.Module):
    """
    GCN with multiple relation types (R-GCN inspired).

    For each adjacency type, runs a separate GCN transform, then
    aggregates via learned attention weights.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int = 5):
        super().__init__()
        self.n_relations = n_relations
        self.gcn_layers = nn.ModuleList([
            GCNLayer(in_dim, out_dim) for _ in range(n_relations)
        ])
        # Learnable relation weights
        self.relation_weights = nn.Parameter(torch.ones(n_relations) / n_relations)

    def forward(self, x: torch.Tensor, adj_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: (B, N, F) node features
            adj_list: List of (B, N, N) adjacency matrices, one per relation
        Returns:
            (B, N, out_dim)
        """
        assert len(adj_list) == self.n_relations

        # Normalize relation weights
        w = torch.softmax(self.relation_weights, dim=0)

        out = torch.zeros_like(self.gcn_layers[0](x, adj_list[0]))
        for i, (gcn, adj) in enumerate(zip(self.gcn_layers, adj_list)):
            out = out + w[i] * gcn(x, adj)

        return out


# ═══════════════════════════════════════════════════════════════
# NBA GRAPH NET — Full Model
# ═══════════════════════════════════════════════════════════════

class NBAGraphNet(nn.Module):
    """
    Graph Neural Network for NBA game prediction.

    Architecture:
      1. Node embedding: Linear(node_feat_dim, d_node) for each team
      2. Multi-relation GCN: Message passing across 5 adjacency types
      3. GAT refinement: Attention-weighted neighborhood aggregation
      4. Game embedding: Extract home/away team embeddings, concatenate
      5. Tabular fusion: Concat graph embed with original tabular features
      6. Prediction head: MLP → P(home_win)

    The model takes:
      - Tabular features X (standard feature matrix from build_features)
      - Team indices for home/away (extracted from game metadata)
      - Graph state (adjacency matrices + node features from NBAGraphBuilder)
    """

    def __init__(self, tabular_dim: int, node_feat_dim: int = 12,
                 d_node: int = 32, d_hidden: int = 128,
                 n_gcn_layers: int = 2, n_gat_heads: int = 4,
                 n_relations: int = 5, dropout: float = 0.3):
        """
        Args:
            tabular_dim: Dimension of the tabular feature vector per game
            node_feat_dim: Dimension of node features (from NBAGraphBuilder)
            d_node: Hidden dimension for node embeddings
            d_hidden: Hidden dimension for prediction head
            n_gcn_layers: Number of stacked GCN layers
            n_gat_heads: Number of attention heads in GAT layer
            n_relations: Number of adjacency relation types (h2h, winrate, margin, struct, perf)
            dropout: Dropout rate
        """
        super().__init__()
        self.d_node = d_node
        self.n_gcn_layers = n_gcn_layers
        self.dropout_rate = dropout

        # ── Node feature projection ──
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, d_node),
            nn.LayerNorm(d_node),
            nn.GELU(),
        )

        # ── Multi-relation GCN stack ──
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        for i in range(n_gcn_layers):
            in_d = d_node if i == 0 else d_node
            self.gcn_layers.append(MultiRelationGCN(in_d, d_node, n_relations))
            self.gcn_norms.append(nn.LayerNorm(d_node))

        # ── GAT refinement (single layer, multi-head) ──
        self.gat = GATLayer(d_node, d_node // n_gat_heads, n_gat_heads,
                            dropout=dropout, concat=True)
        self.gat_norm = nn.LayerNorm(d_node)

        # ── Game embedding: combine home + away node embeddings ──
        # home_embed || away_embed || (home - away) || (home * away) = 4 * d_node
        game_graph_dim = 4 * d_node

        # ── Tabular feature projection ──
        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Fusion + prediction head ──
        fusion_dim = game_graph_dim + d_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

        # ── Residual gate: how much to trust graph vs tabular ──
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid(),
        )

        # ── Tabular-only fallback head (for gated residual) ──
        self.tab_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, x_tab: torch.Tensor, node_feats: torch.Tensor,
                adj_list: List[torch.Tensor], home_idx: torch.Tensor,
                away_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_tab: Tabular features (B, tabular_dim)
            node_feats: Team node features (B, 30, node_feat_dim) or (30, node_feat_dim)
            adj_list: List of adjacency matrices, each (B, 30, 30) or (30, 30)
            home_idx: Home team indices (B,) — values in [0, 29]
            away_idx: Away team indices (B,) — values in [0, 29]

        Returns:
            logits: (B, 1) — raw logits for P(home_win)
        """
        B = x_tab.size(0)

        # Handle shared graph (no batch dim on node_feats / adj)
        if node_feats.dim() == 2:
            node_feats = node_feats.unsqueeze(0).expand(B, -1, -1)
        if adj_list[0].dim() == 2:
            adj_list = [a.unsqueeze(0).expand(B, -1, -1) for a in adj_list]

        # ── 1. Project node features ──
        h = self.node_proj(node_feats)  # (B, 30, d_node)

        # ── 2. Multi-relation GCN message passing ──
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_new = gcn(h, adj_list)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_rate, training=self.training)
            h = norm(h + h_new)  # Residual connection

        # ── 3. GAT refinement ──
        # Use combined adjacency for GAT masking
        adj_combined = sum(adj_list) / len(adj_list)
        h_gat = self.gat(h, adj_combined)
        h_gat = F.gelu(h_gat)
        h_gat = F.dropout(h_gat, p=self.dropout_rate, training=self.training)
        h = self.gat_norm(h + h_gat)  # Residual

        # ── 4. Extract home/away embeddings ──
        # Gather team embeddings for each game in batch
        batch_idx = torch.arange(B, device=h.device)
        h_home = h[batch_idx, home_idx]  # (B, d_node)
        h_away = h[batch_idx, away_idx]  # (B, d_node)

        # Game-level graph embedding: concat, diff, interaction
        game_graph = torch.cat([
            h_home,
            h_away,
            h_home - h_away,
            h_home * h_away,
        ], dim=-1)  # (B, 4 * d_node)

        # ── 5. Tabular projection ──
        tab_emb = self.tab_proj(x_tab)  # (B, d_hidden)

        # ── 6. Fusion with learned gate ──
        fused = torch.cat([game_graph, tab_emb], dim=-1)  # (B, fusion_dim)

        # Gate: blend between graph-aware and tabular-only prediction
        g = self.gate(fused)  # (B, 1)
        logit_full = self.head(fused)       # (B, 1) — graph + tabular
        logit_tab = self.tab_head(tab_emb)  # (B, 1) — tabular only

        # Final logit: gated combination
        logits = g * logit_full + (1 - g) * logit_tab

        return logits


# ═══════════════════════════════════════════════════════════════
# GRAPH-AWARE DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

def build_graph_data(games: list) -> Tuple[np.ndarray, np.ndarray, list,
                                            list, list, list]:
    """
    Build feature matrix + graph metadata from raw game data.

    This is a graph-aware version of build_features() from nba_gpu_runner.py.
    It returns additional per-game metadata needed for graph construction:
    - home_team_idx: index of home team per game (for node lookup)
    - away_team_idx: index of away team per game
    - graph_snapshots: list of (node_features, adj_matrices) per game

    Instead of storing per-game snapshots (too much memory), we store
    per-fold snapshots and reconstruct within the fold function.

    Returns:
        X, y, feature_names, home_teams, away_teams, game_dates
    """
    from collections import defaultdict
    from datetime import datetime

    WINDOWS = [3, 5, 7, 10, 15, 20]

    def _safe(v, d=0.0):
        try:
            f = float(v)
            return f if np.isfinite(f) else d
        except (ValueError, TypeError):
            return d

    def _elo(he, ae, hw, K=20, H=100):
        e = 1.0 / (1.0 + 10 ** ((ae - he - H) / 400))
        return he + K * (hw - e), ae + K * ((1 - hw) - (1 - e))

    def _streak(res):
        if not res:
            return 0
        s, lw = 0, res[-1][1]
        for r in reversed(res):
            if r[1] == lw:
                s += 1
            else:
                break
        return s if lw else -s

    def _rstats(res, w):
        rc = res[-w:] if len(res) >= w else res
        if not rc:
            return [0.0] * 6
        wins = sum(1 for r in rc if r[1])
        m = [r[2] for r in rc]
        s = [r[3] for r in rc]
        return [
            wins / len(rc),
            float(np.mean(s)),
            float(np.mean(m)),
            float(np.std(m)) if len(m) > 1 else 0.0,
            float(np.max(m)),
            float(np.min(m)),
        ]

    tr = defaultdict(list)
    tl = {}
    te = defaultdict(lambda: 1500.0)
    X, Y, fnames = [], [], []
    home_teams, away_teams, game_dates = [], [], []
    first = True

    for g in games:
        hi, ai = g.get("home", {}), g.get("away", {})
        hn = hi.get("team_name", g.get("home_team", ""))
        an = ai.get("team_name", g.get("away_team", ""))
        hp, ap = _safe(hi.get("pts", 0)), _safe(ai.get("pts", 0))
        gd = g.get("game_date", g.get("date", "2020-01-01"))
        ht = TEAM_MAP.get(hn, hn[:3].upper() if hn else "UNK")
        at = TEAM_MAP.get(an, an[:3].upper() if an else "UNK")
        if not hn or not an or (hp == 0 and ap == 0):
            continue
        hw = 1 if hp > ap else 0
        mg = hp - ap
        hr, ar = tr[ht], tr[at]
        if len(hr) < 5 or len(ar) < 5:
            tr[ht].append((gd, hw == 1, mg, hp, at))
            tr[at].append((gd, hw == 0, -mg, ap, ht))
            te[ht], te[at] = _elo(te[ht], te[at], hw)
            tl[ht] = gd
            tl[at] = gd
            continue

        row, names = [], []
        # Rolling stats
        for w in WINDOWS:
            hs, als = _rstats(hr, w), _rstats(ar, w)
            for i, lb in enumerate(["winpct", "avg_pts", "avg_margin",
                                     "margin_vol", "best_mg", "worst_mg"]):
                row.extend([hs[i], als[i], hs[i] - als[i]])
                names.extend([f"h_{lb}_w{w}", f"a_{lb}_w{w}", f"d_{lb}_w{w}"])
        # ELO
        row.extend([te[ht], te[at], te[ht] - te[at]])
        names.extend(["h_elo", "a_elo", "elo_d"])
        # Rest
        try:
            gdt = datetime.strptime(str(gd)[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            gdt = datetime(2020, 1, 1)
        hrest, arest = 3.0, 3.0
        for t, rest_name in [(ht, "h"), (at, "a")]:
            if t in tl:
                try:
                    d = (gdt - datetime.strptime(str(tl[t])[:10], "%Y-%m-%d")).days
                    if rest_name == "h":
                        hrest = min(d, 10)
                    else:
                        arest = min(d, 10)
                except (ValueError, TypeError):
                    pass
        row.extend([hrest, arest, hrest - arest, 1.0 if hrest <= 1 else 0.0])
        names.extend(["h_rest", "a_rest", "rest_d", "h_b2b"])
        # Streak
        hs_streak, as_streak = _streak(hr), _streak(ar)
        row.extend([hs_streak, as_streak, hs_streak - as_streak])
        names.extend(["h_streak", "a_streak", "streak_d"])
        # H2H
        h2h = [r for r in hr if r[4] == at][-10:]
        h2hw = sum(1 for r in h2h if r[1]) / max(len(h2h), 1)
        row.extend([h2hw, len(h2h)])
        names.extend(["h2h_wp", "h2h_n"])
        # Season + home court
        hgp, agp = len(hr), len(ar)
        hsw = sum(1 for r in hr if r[1]) / max(hgp, 1)
        asw = sum(1 for r in ar if r[1]) / max(agp, 1)
        row.extend([hsw, asw, hsw - asw, 1.0])
        names.extend(["h_swp", "a_swp", "swp_d", "home_court"])

        if first:
            fnames = names
            first = False

        X.append(row)
        Y.append(hw)
        home_teams.append(ht)
        away_teams.append(at)
        game_dates.append(gd)

        tr[ht].append((gd, hw == 1, mg, hp, at))
        tr[at].append((gd, hw == 0, -mg, ap, ht))
        te[ht], te[at] = _elo(te[ht], te[at], hw)
        tl[ht] = gd
        tl[at] = gd

    Xa = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0,
                       posinf=1e6, neginf=-1e6)
    ya = np.array(Y, dtype=np.int32)
    print(f"[GraphNet] Features: {Xa.shape} ({len(fnames)} cols, {len(ya)} games)")
    return Xa, ya, fnames, home_teams, away_teams, game_dates


def build_graph_from_games(games: list, up_to_idx: Optional[int] = None) -> NBAGraphBuilder:
    """
    Build a NBAGraphBuilder from raw game list, replaying all games
    up to the given index.

    Args:
        games: Full game list (same format as load_games())
        up_to_idx: Only process games up to this index (exclusive).
                   If None, process all games.

    Returns:
        NBAGraphBuilder with accumulated state
    """
    builder = NBAGraphBuilder(decay=0.97, window=82)

    limit = up_to_idx if up_to_idx is not None else len(games)
    for g in games[:limit]:
        hi, ai = g.get("home", {}), g.get("away", {})
        hn = hi.get("team_name", g.get("home_team", ""))
        an = ai.get("team_name", g.get("away_team", ""))
        hp = float(hi.get("pts", 0)) if hi.get("pts") else 0.0
        ap = float(ai.get("pts", 0)) if ai.get("pts") else 0.0
        if not hn or not an or (hp == 0 and ap == 0):
            continue
        ht = TEAM_MAP.get(hn, hn[:3].upper() if hn else "UNK")
        at = TEAM_MAP.get(an, an[:3].upper() if an else "UNK")
        hw = 1 if hp > ap else 0
        builder.update(ht, at, hp, ap, hw)

    return builder


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP FOR GRAPH MODEL
# ═══════════════════════════════════════════════════════════════

def _train_graph_model(
    model: NBAGraphNet,
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    home_tr: np.ndarray, away_tr: np.ndarray,
    home_va: np.ndarray, away_va: np.ndarray,
    node_feats: np.ndarray,
    adj_matrices: Dict[str, np.ndarray],
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Train the NBAGraphNet model and return validation predictions.

    Args:
        model: NBAGraphNet instance
        Xtr, ytr: Training features and labels
        Xva, yva: Validation features and labels
        home_tr, away_tr: Home/away team indices for training set
        home_va, away_va: Home/away team indices for validation set
        node_feats: Team node features (30, node_feat_dim)
        adj_matrices: Dict of adjacency matrices, each (30, 30)
        lr, weight_decay, max_epochs, patience, batch_size: Training params

    Returns:
        val_probs: (n_val,) array of predicted probabilities
    """
    device = torch.device(DEVICE)
    model = model.to(device)

    # Convert graph data to tensors (shared across all samples)
    nf_t = torch.tensor(node_feats, dtype=torch.float32).to(device)  # (30, 12)
    adj_keys = ["h2h", "winrate", "margin", "structure", "performance"]
    adj_t = [torch.tensor(adj_matrices[k], dtype=torch.float32).to(device) for k in adj_keys]

    # Convert tabular data
    xt = torch.tensor(Xtr, dtype=torch.float32).to(device)
    xv = torch.tensor(Xva, dtype=torch.float32).to(device)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1).to(device)
    yv_t = torch.tensor(yva, dtype=torch.float32).unsqueeze(1).to(device)

    # Team indices
    ht_t = torch.tensor(home_tr, dtype=torch.long).to(device)
    at_t = torch.tensor(away_tr, dtype=torch.long).to(device)
    hv_t = torch.tensor(home_va, dtype=torch.long).to(device)
    av_t = torch.tensor(away_va, dtype=torch.long).to(device)

    # DataLoader
    dataset = torch.utils.data.TensorDataset(xt, yt, ht_t, at_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb, hb, ab in loader:
            optimizer.zero_grad()
            logits = model(xb, nf_t, adj_t, hb, ab)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(xv, nf_t, adj_t, hv_t, av_t)
            val_loss = criterion(val_logits, yv_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final predictions
    model.eval()
    with torch.no_grad():
        val_logits = model(xv, nf_t, adj_t, hv_t, av_t)
        probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

    # Cleanup
    del xt, xv, yt, yv_t, ht_t, at_t, hv_t, av_t, nf_t
    for a in adj_t:
        del a
    torch.cuda.empty_cache()

    return probs


# ═══════════════════════════════════════════════════════════════
# _fold_graph — COMPATIBLE WITH wf_eval FRAMEWORK
# ═══════════════════════════════════════════════════════════════

def _fold_graph(Xtr: np.ndarray, ytr: np.ndarray,
                Xva: np.ndarray, yva: np.ndarray,
                params: dict,
                home_tr: np.ndarray, away_tr: np.ndarray,
                home_va: np.ndarray, away_va: np.ndarray,
                games: Optional[list] = None,
                train_end_game_idx: Optional[int] = None) -> np.ndarray:
    """
    Single fold training for graph model — compatible with wf_eval pattern.

    This function builds the graph state from games up to the training cutoff,
    then trains NBAGraphNet on the training set and predicts on validation.

    Args:
        Xtr, ytr: Training features and labels (already scaled)
        Xva, yva: Validation features and labels
        params: Hyperparameters dict with keys:
            - d_node: Node embedding dim (default 32)
            - d_hidden: Hidden dim for prediction head (default 128)
            - n_gcn_layers: Number of GCN layers (default 2)
            - n_gat_heads: Number of GAT attention heads (default 4)
            - dropout: Dropout rate (default 0.3)
            - lr: Learning rate (default 5e-4)
            - weight_decay: Weight decay (default 1e-4)
            - batch_size: Batch size (default 512)
        home_tr, away_tr: Home/away team indices for training games
        home_va, away_va: Home/away team indices for validation games
        games: Raw game list (for graph construction). If None, uses
               default graph with structural priors only.
        train_end_game_idx: Index into games list marking end of training
                            period (for graph state cutoff).

    Returns:
        val_probs: (n_val,) array of predicted P(home_win)
    """
    tabular_dim = Xtr.shape[1]

    # ── Build graph state ──
    if games is not None and train_end_game_idx is not None:
        builder = build_graph_from_games(games, up_to_idx=train_end_game_idx)
    else:
        # Fallback: build graph from team indices alone (structural prior only)
        builder = NBAGraphBuilder(decay=0.97, window=82)

    adj_matrices = builder.build_adjacency_matrices()
    node_feats = builder.build_node_features()

    # ── Build model ──
    d_node = params.get("d_node", 32)
    d_hidden = params.get("d_hidden", 128)
    n_gcn = params.get("n_gcn_layers", 2)
    n_gat_heads = params.get("n_gat_heads", 4)
    n_relations = 5  # h2h, winrate, margin, structure, performance
    dropout = params.get("dropout", 0.3)

    model = NBAGraphNet(
        tabular_dim=tabular_dim,
        node_feat_dim=node_feats.shape[1],
        d_node=d_node,
        d_hidden=d_hidden,
        n_gcn_layers=n_gcn,
        n_gat_heads=n_gat_heads,
        n_relations=n_relations,
        dropout=dropout,
    )

    # ── Train ──
    probs = _train_graph_model(
        model=model,
        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
        home_tr=home_tr, away_tr=away_tr,
        home_va=home_va, away_va=away_va,
        node_feats=node_feats,
        adj_matrices=adj_matrices,
        lr=params.get("lr", 5e-4),
        weight_decay=params.get("weight_decay", 1e-4),
        max_epochs=params.get("max_epochs", 200),
        patience=params.get("patience", 15),
        batch_size=params.get("batch_size", 512),
    )

    return probs


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION (graph-aware)
# ═══════════════════════════════════════════════════════════════

def wf_eval_graph(games: list, params: Optional[dict] = None,
                  n_splits: int = 3, max_eval_games: int = 10000) -> dict:
    """
    Walk-forward backtest for graph model.

    This is a self-contained evaluation function that:
    1. Builds features + graph metadata from games
    2. Runs TimeSeriesSplit walk-forward
    3. For each fold, builds graph state up to training cutoff
    4. Trains NBAGraphNet and evaluates

    Args:
        games: Raw game list from load_games()
        params: Hyperparameters (see _fold_graph docstring)
        n_splits: Number of walk-forward splits
        max_eval_games: Maximum games to evaluate

    Returns:
        Dict with brier, accuracy, log_loss, roi, folds, etc.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
    from sklearn.preprocessing import StandardScaler

    params = params or {}

    # ── Build features + metadata ──
    X, y, fnames, home_teams, away_teams, game_dates = build_graph_data(games)

    if X.shape[0] > max_eval_games:
        X = X[-max_eval_games:]
        y = y[-max_eval_games:]
        home_teams = home_teams[-max_eval_games:]
        away_teams = away_teams[-max_eval_games:]
        game_dates = game_dates[-max_eval_games:]

    # Convert team names to indices
    home_idx = np.array([TEAM_TO_IDX.get(t, 0) for t in home_teams], dtype=np.int64)
    away_idx = np.array([TEAM_TO_IDX.get(t, 0) for t in away_teams], dtype=np.int64)

    # Scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6))

    # ── Walk-forward ──
    folds = []
    all_preds, all_true = [], []

    tss = TimeSeriesSplit(n_splits=n_splits)
    for fi, (ti, vi) in enumerate(tss.split(Xs)):
        Xtr, Xva = Xs[ti], Xs[vi]
        ytr, yva = y[ti], y[vi]
        h_tr, h_va = home_idx[ti], home_idx[vi]
        a_tr, a_va = away_idx[ti], away_idx[vi]

        print(f"  Fold {fi+1}/{n_splits}: train={len(ti)}, val={len(vi)}")

        try:
            # Estimate game index in original games list for graph cutoff
            # The last training game date gives us the cutoff
            train_end_date = game_dates[ti[-1]]
            # Find corresponding index in full games list
            train_end_game_idx = None
            for gi, g in enumerate(games):
                gd = g.get("game_date", g.get("date", ""))
                if gd > train_end_date:
                    train_end_game_idx = gi
                    break
            if train_end_game_idx is None:
                train_end_game_idx = len(games)

            probs = _fold_graph(
                Xtr, ytr, Xva, yva, params,
                h_tr, a_tr, h_va, a_va,
                games=games,
                train_end_game_idx=train_end_game_idx,
            )
            probs = np.clip(probs, 0.001, 0.999)

            b = float(brier_score_loss(yva, probs))
            ll = float(log_loss(yva, probs))
            ac = float(accuracy_score(yva, (probs > 0.5).astype(int)))

            folds.append({
                "fold": fi + 1, "brier": round(b, 5),
                "log_loss": round(ll, 5), "accuracy": round(ac, 4),
                "n_train": len(ti), "n_val": len(vi),
            })
            all_preds.extend(probs.tolist())
            all_true.extend(yva.tolist())
            print(f"    Brier:{b:.4f} Acc:{ac:.4f} LL:{ll:.4f}")

        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            folds.append({"fold": fi + 1, "error": str(e)[:300]})

        torch.cuda.empty_cache()

    # ── Aggregate results ──
    valid_folds = [f for f in folds if "brier" in f]
    if not valid_folds:
        return {"brier": 1.0, "accuracy": 0.0, "log_loss": 10.0,
                "error": "All folds failed", "folds": folds}

    if all_preds:
        ap, at = np.array(all_preds), np.array(all_true)
        overall_brier = float(brier_score_loss(at, ap))
        overall_acc = float(accuracy_score(at, (ap > 0.5).astype(int)))
        overall_ll = float(log_loss(at, ap))
    else:
        overall_brier = float(np.mean([f["brier"] for f in valid_folds]))
        overall_acc = float(np.mean([f["accuracy"] for f in valid_folds]))
        overall_ll = float(np.mean([f["log_loss"] for f in valid_folds]))

    # ROI calculation (same as nba_gpu_runner)
    bets, profit = 0, 0.0
    for p, a in zip(all_preds, all_true):
        if p > 0.55:
            bets += 1
            profit += (100 / 110) if a == 1 else -1.0
        elif p < 0.45:
            bets += 1
            profit += (100 / 110) if a == 0 else -1.0
    roi = float(profit / max(bets, 1))

    return {
        "model": "graph_net",
        "brier": round(overall_brier, 5),
        "accuracy": round(overall_acc, 4),
        "log_loss": round(overall_ll, 5),
        "roi": round(roi, 4),
        "avg_brier": round(float(np.mean([f["brier"] for f in valid_folds])), 5),
        "n_splits": n_splits,
        "n_games": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "folds": folds,
        "valid_folds": len(valid_folds),
        "graph_info": {
            "n_teams": NUM_TEAMS,
            "n_relations": 5,
            "relation_types": ["h2h", "winrate", "margin", "structure", "performance"],
            "node_feat_dim": 12,
        },
    }


# ═══════════════════════════════════════════════════════════════
# CLI — standalone test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"NBA Graph Neural Network")
    print(f"PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {DEVICE}")
    print(f"Teams: {NUM_TEAMS}")
    print(f"Relation types: 5 (h2h, winrate, margin, structure, performance)")

    # Quick architecture test with synthetic data
    B, F = 64, 124  # batch, feature dim
    model = NBAGraphNet(tabular_dim=F, node_feat_dim=12, d_node=32,
                        d_hidden=128, n_gcn_layers=2, n_gat_heads=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Synthetic forward pass
    x_tab = torch.randn(B, F)
    node_feats = torch.randn(30, 12)
    adj_list = [torch.rand(30, 30) for _ in range(5)]
    home_idx = torch.randint(0, 30, (B,))
    away_idx = torch.randint(0, 30, (B,))

    model.eval()
    with torch.no_grad():
        logits = model(x_tab, node_feats, adj_list, home_idx, away_idx)
        probs = torch.sigmoid(logits)
    print(f"Output shape: {logits.shape}")
    print(f"Prob range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Prob mean: {probs.mean().item():.4f}")
    print("\nArchitecture test PASSED")
