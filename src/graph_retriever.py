"""Graph-enhanced retriever with contrastive embeddings and synonym/polysemy awareness."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = self.W(h)
        N = Wh.size(0)
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)
        e = self.leaky_relu(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        return attention @ Wh


class MultiHeadGAT(nn.Module):
    """Multi-head graph attention network."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int,
                 num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_features
            heads = nn.ModuleList([
                GraphAttentionLayer(in_dim, hidden_features // num_heads, dropout)
                for _ in range(num_heads)
            ])
            self.layers.append(heads)
            self.norms.append(nn.LayerNorm(hidden_features))
        self.proj_out = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for heads, norm in zip(self.layers, self.norms):
            h = torch.cat([head(x, adj) for head in heads], dim=-1)
            h = norm(h)
            h = F.elu(h)
            h = self.dropout(h)
            x = x + h if x.shape == h.shape else h
        return self.proj_out(x)


class SynonymGraph:
    """Synonym/polysemy graph with node embeddings."""

    def __init__(self):
        self.nodes: Dict[str, int] = {}
        self.node_texts: List[str] = []
        self.edges: List[Tuple[int, int, str]] = []
        self.adj_matrix: Optional[torch.Tensor] = None

    def add_node(self, text: str) -> int:
        if text not in self.nodes:
            idx = len(self.nodes)
            self.nodes[text] = idx
            self.node_texts.append(text)
        return self.nodes[text]

    def add_edge(self, src: str, dst: str, edge_type: str = "synonym"):
        src_idx = self.add_node(src)
        dst_idx = self.add_node(dst)
        self.edges.append((src_idx, dst_idx, edge_type))

    def build_adjacency(self) -> torch.Tensor:
        n = len(self.nodes)
        adj = torch.zeros(n, n)
        for src, dst, _ in self.edges:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0
        adj += torch.eye(n)
        self.adj_matrix = adj
        return adj

    def get_laplacian(self) -> torch.Tensor:
        """Compute normalized graph Laplacian for regularization."""
        adj = self.adj_matrix if self.adj_matrix is not None else self.build_adjacency()
        degree = adj.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree.clamp(min=1e-8)))
        I = torch.eye(adj.shape[0], device=adj.device)
        return I - D_inv_sqrt @ adj @ D_inv_sqrt


class GraphContrastiveRetriever(nn.Module):
    """Graph-aware contrastive retriever built on top of a sentence encoder."""

    def __init__(self, encoder_dim: int = 1024, gnn_hidden: int = 1024,
                 gnn_layers: int = 3, gnn_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, encoder_dim),
        )
        self.doc_proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, encoder_dim),
        )
        self.gnn = MultiHeadGAT(
            in_features=encoder_dim,
            hidden_features=gnn_hidden,
            out_features=encoder_dim,
            num_heads=gnn_heads,
            num_layers=gnn_layers,
            dropout=dropout,
        )
        self.graph_gate = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.Sigmoid(),
        )

    def encode_queries(self, query_embeds: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_proj(query_embeds), p=2, dim=-1)

    def encode_docs(self, doc_embeds: torch.Tensor, graph_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        proj = self.doc_proj(doc_embeds)
        if graph_embeds is not None:
            gate = self.graph_gate(torch.cat([proj, graph_embeds], dim=-1))
            proj = gate * proj + (1 - gate) * graph_embeds
        return F.normalize(proj, p=2, dim=-1)

    def compute_graph_embeddings(self, node_embeds: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.gnn(node_embeds, adj)

    def contrastive_loss(self, query_embeds: torch.Tensor, pos_doc_embeds: torch.Tensor,
                         neg_doc_embeds: Optional[torch.Tensor] = None,
                         temperature: float = 0.05) -> torch.Tensor:
        """InfoNCE contrastive loss with in-batch negatives."""
        q = self.encode_queries(query_embeds)
        p = self.encode_docs(pos_doc_embeds)
        pos_sim = (q * p).sum(dim=-1) / temperature

        if neg_doc_embeds is not None:
            n = self.encode_docs(neg_doc_embeds)
            neg_sim = (q.unsqueeze(1) @ n.T.unsqueeze(0)).squeeze(1) / temperature
        else:
            neg_sim = (q @ p.T) / temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def graph_laplacian_loss(self, embeddings: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """Graph Laplacian regularization: tr(E^T L E) encourages smooth embeddings over graph."""
        return torch.trace(embeddings.T @ laplacian @ embeddings) / embeddings.shape[0]

    def forward(self, query_embeds: torch.Tensor, pos_doc_embeds: torch.Tensor,
                neg_doc_embeds: Optional[torch.Tensor] = None,
                node_embeds: Optional[torch.Tensor] = None,
                adj: Optional[torch.Tensor] = None,
                laplacian: Optional[torch.Tensor] = None,
                temperature: float = 0.05,
                laplacian_weight: float = 0.1) -> Dict[str, torch.Tensor]:
        graph_embeds = None
        if node_embeds is not None and adj is not None:
            graph_embeds = self.compute_graph_embeddings(node_embeds, adj)

        info_nce = self.contrastive_loss(query_embeds, pos_doc_embeds, neg_doc_embeds, temperature)

        lap_loss = torch.tensor(0.0, device=query_embeds.device)
        if graph_embeds is not None and laplacian is not None:
            lap_loss = self.graph_laplacian_loss(graph_embeds, laplacian)

        total = info_nce + laplacian_weight * lap_loss
        return {"loss": total, "info_nce": info_nce, "laplacian_loss": lap_loss}

    def retrieve(self, query_embeds: torch.Tensor, doc_embeds: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """Retrieve top-k documents for given queries. Returns indices."""
        q = self.encode_queries(query_embeds)
        d = self.encode_docs(doc_embeds)
        scores = q @ d.T
        _, indices = scores.topk(top_k, dim=-1)
        return indices
