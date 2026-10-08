"""FaceTopoFormer: the complete inference model, with fixed dataset presets.

Paper modules: FRDO is implemented in infer.py; FDSMixer combines diffusion
states and ordered face paths; BGSR refines face scores across shared edges.
Parameter names are retained for compatibility with the original checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelPreset:
    num_classes: int
    path_count: int
    path_length: int
    diffusion_steps: int
    window: int
    chunk_size: int


PRESETS = {
    "sum": ModelPreset(6, 4, 6, 4, 256, 49152),
    "sum-parts": ModelPreset(12, 4, 6, 4, 256, 49152),
    "gw-meshseg": ModelPreset(2, 8, 10, 6, 320, 32768),
}


def _segment_sum(values, index, count):
    return values.new_zeros((count,) + values.shape[1:]).index_add_(0, index, values)


def _segment_softmax(scores, index, count):
    expanded = index[:, None].expand_as(scores)
    maximum = scores.new_full((count, scores.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, expanded, scores, reduce="amax", include_self=True)
    numerator = (scores - maximum[index]).exp()
    return numerator / _segment_sum(numerator, index, count)[index]


class FeedForward(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(4 * dim, dim), nn.Dropout(0.0),
        )

    def forward(self, x):
        return self.net(x)


class TextureEncoder(nn.Module):
    """Seven texture samples -> attention-weighted feature and feature spread."""

    def __init__(self, dim=96):
        super().__init__()
        self.sample_mlp = nn.Sequential(nn.Linear(3, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim))
        self.score = nn.Linear(dim, 1)
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))

    def forward(self, texture):
        features = self.sample_mlp(texture)
        attended = (self.score(features).softmax(1) * features).sum(1)
        spread = (((features - attended[:, None]) ** 2).mean(1) + 1e-6).sqrt()
        return self.out(torch.cat((attended, spread), 1))


class TopologyAttention(nn.Module):
    """Edge-conditioned attention; edge_index[0] is the receiving face."""

    def __init__(self, dim=96, heads=6):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.edge_k, self.edge_v = nn.Linear(7, dim), nn.Linear(7, dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_norm, self.ffn = nn.LayerNorm(dim), FeedForward(dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel():
            src, dst = edge_index
            z = self.norm(x)
            q = self.q(z[src]).view(-1, self.heads, self.head_dim)
            k = (self.k(z[dst]) + self.edge_k(edge_attr)).view_as(q)
            v = (self.v(z[dst]) + self.edge_v(edge_attr)).view_as(q)
            scores = (q * k).sum(-1) * self.head_dim ** -0.5
            weights = _segment_softmax(scores, src, len(x))
            context = _segment_sum(weights.unsqueeze(-1) * v, src, len(x)).reshape_as(x)
            x = x + self.proj(context)
        return x + self.ffn(self.ffn_norm(x))


class _ContextAttention(nn.Module):
    def __init__(self, dim=96, heads=6):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.ffn_norm, self.ffn = nn.LayerNorm(dim), FeedForward(dim)

    def _attend(self, x, z, tokens):
        count, dim = x.shape
        q = self.q(z).view(count, self.heads, self.head_dim)
        k = self.k(tokens).view(count, -1, self.heads, self.head_dim)
        v = self.v(tokens).view_as(k)
        scores = (q[:, None] * k).sum(-1).permute(0, 2, 1)
        weights = (scores * self.head_dim ** -0.5).softmax(-1)
        context = (weights.unsqueeze(-1) * v.permute(0, 2, 1, 3)).sum(2).reshape_as(x)
        x = x + self.proj(context) * self.gamma
        return x + self.ffn(self.ffn_norm(x))


class DiffusionStateAttention(_ContextAttention):
    """Finite dual-graph diffusion states selected by face-wise cross-attention."""

    def __init__(self, steps, dim=96, heads=6):
        super().__init__(dim, heads)
        self.steps = steps
        self.step_embedding = nn.Parameter(torch.zeros(1, steps + 1, dim))
        nn.init.trunc_normal_(self.step_embedding, std=0.02)

    def forward(self, x, edge_index):
        z = self.norm(x)
        states, current = [z], z
        if edge_index.numel():
            src, dst = edge_index
            degree = _segment_sum(torch.ones_like(src, dtype=z.dtype), src, len(x)).clamp_min(1)[:, None]
            for _ in range(self.steps):
                neighbor = _segment_sum(current[dst], src, len(x)) / degree
                current = 0.5 * current + 0.5 * neighbor
                states.append(current)
        else:
            states.extend([z] * self.steps)
        return self._attend(x, z, torch.stack(states, 1) + self.step_embedding)


class OrderedPathAttention(_ContextAttention):
    """Cross-attention over ordered face paths with learned path/step embeddings."""

    def __init__(self, path_count, path_length, dim=96, heads=6):
        super().__init__(dim, heads)
        self.step_embedding = nn.Parameter(torch.zeros(1, path_count, path_length, dim))
        nn.init.trunc_normal_(self.step_embedding, std=0.02)

    def forward(self, x, path_index):
        z = self.norm(x)
        tokens = z[path_index] + self.step_embedding
        return self._attend(x, z, tokens.flatten(1, 2))


class SerializedWindowAttention(nn.Module):
    """Window attention on Morton-ordered faces, with alternating half shifts."""

    def __init__(self, window, shift, dim=96, heads=6):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm, self.ffn = nn.LayerNorm(dim), FeedForward(dim)

    def forward(self, x):
        count, dim = x.shape
        z = self.norm(x)
        padding = (-count) % self.window
        z = torch.cat((z, z.new_zeros(padding, dim)), 0)
        z = torch.roll(z, -self.shift, dims=0).view(-1, self.window, dim)
        # Keep the original unmasked zero-padding convention for checkpoint parity.
        attended, _ = self.attention(z, z, z, need_weights=False)
        attended = torch.roll(attended.reshape(-1, dim), self.shift, dims=0)[:count]
        x = x + attended
        return x + self.ffn(self.ffn_norm(x))


class _LocalWindowBlock(nn.Module):
    def __init__(self, preset):
        super().__init__()
        self.topology = TopologyAttention()
        self.serialized = SerializedWindowAttention(preset.window, preset.window // 2)

    def forward(self, x, edge_index, edge_attr, path_index):
        return self.serialized(self.topology(x, edge_index, edge_attr))


class _PathWindowBlock(_LocalWindowBlock):
    """Additional path context in the fixed GW-MeshSeg block schedule."""

    def __init__(self, preset):
        super().__init__(preset)
        self.path = OrderedPathAttention(preset.path_count, preset.path_length)

    def forward(self, x, edge_index, edge_attr, path_index):
        x = self.topology(x, edge_index, edge_attr)
        return self.serialized(self.path(x, path_index))


class FDSMixer(nn.Module):
    """Shared topology -> parallel diffusion/path contexts -> face-wise mixing."""

    def __init__(self, preset):
        super().__init__()
        self.topology = TopologyAttention()
        self.diffusion = DiffusionStateAttention(preset.diffusion_steps)
        self.path = OrderedPathAttention(preset.path_count, preset.path_length)
        self.relation_gate = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 2))
        self.serialized = SerializedWindowAttention(preset.window, 0)
        nn.init.zeros_(self.relation_gate[-1].weight)
        nn.init.zeros_(self.relation_gate[-1].bias)

    def forward(self, x, edge_index, edge_attr, path_index):
        base = self.topology(x, edge_index, edge_attr)
        diffusion = self.diffusion(base, edge_index)
        path = self.path(base, path_index)
        gate = self.relation_gate(base).softmax(1)
        mixed = base + gate[:, :1] * (diffusion - base) + gate[:, 1:] * (path - base)
        return self.serialized(mixed)


class BGSR(nn.Module):
    """Boundary-gated semantic refinement with two residual score updates."""

    def __init__(self, num_classes, dim=96):
        super().__init__()
        self.update = nn.Sequential(nn.Linear(dim + 3 * num_classes, dim // 2), nn.GELU(), nn.Linear(dim // 2, num_classes))
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, features, logits, edge_index, boundary_logits):
        if not edge_index.numel():
            return logits
        src, dst = edge_index
        gate = torch.sigmoid(-boundary_logits).unsqueeze(1)
        degree = _segment_sum(gate, src, len(logits)).clamp_min(1e-5)
        for _ in range(2):
            neighbor = _segment_sum(gate * logits[dst], src, len(logits)) / degree
            residual = self.update(torch.cat((features, logits, neighbor, logits - neighbor), 1))
            logits = logits + self.gamma * residual
        return logits


class FaceTopoFormer(nn.Module):
    """Fixed full model for SUM, SUM-Parts, or GW-MeshSeg; no ablation switches.

    Inputs: geom [N,10/38], texture [N,7,3], edge_index [2,E],
    edge_attr [E,7], path_index [N,K,L]. Faces must already be Morton-ordered.
    Logits retain this input face order. Use eval() and float32 for inference.
    """

    def __init__(self, dataset="sum"):
        super().__init__()
        if dataset not in PRESETS:
            raise ValueError(f"Unknown dataset {dataset!r}; choose from {tuple(PRESETS)}")
        self.preset = preset = PRESETS[dataset]
        self.dataset = dataset
        heritage = dataset == "gw-meshseg"
        self.geom_stem = nn.Sequential(nn.Linear(38 if heritage else 10, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 96))
        self.texture_stem = TextureEncoder()
        self.fusion = nn.Sequential(nn.Linear(192, 96), nn.LayerNorm(96), nn.GELU())
        secondary = _PathWindowBlock if heritage else _LocalWindowBlock
        self.blocks = nn.ModuleList([FDSMixer(preset) if i % 2 == 0 else secondary(preset) for i in range(6)])
        self.context = nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 96))
        self.final_norm = nn.LayerNorm(96)
        self.classifier = nn.Linear(96, preset.num_classes)
        self.coarse_classifier = nn.Linear(96, 6) if dataset == "sum-parts" else None
        self.boundary = nn.Sequential(nn.Linear(103, 48), nn.GELU(), nn.Linear(48, 1))
        self.semantic_refiner = BGSR(preset.num_classes)
        self.tile_presence = nn.Linear(28, 1) if heritage else None
        self.tile_presence_gamma = nn.Parameter(torch.tensor(0.1)) if heritage else None

    def forward(self, geom, texture, edge_index, edge_attr, path_index):
        x = self.fusion(torch.cat((self.geom_stem(geom), self.texture_stem(texture)), 1))
        for block in self.blocks:
            x = block(x, edge_index, edge_attr, path_index)
        pooled = torch.cat((x.mean(0), x.amax(0)), 0)
        x = self.final_norm(x + self.context(pooled).unsqueeze(0))
        src, dst = edge_index
        boundary = self.boundary(torch.cat(((x[src] - x[dst]).abs(), edge_attr), 1)).squeeze(1)
        logits = self.semantic_refiner(x, self.classifier(x), edge_index, boundary)
        presence = None
        if self.tile_presence is not None:
            presence = self.tile_presence(geom[0, -28:]).reshape(1)
            bias = self.tile_presence_gamma * presence.clamp_max(0.0)
            logits = logits + torch.cat((-0.5 * bias, 0.5 * bias)).unsqueeze(0)
        return {
            "logits": logits,
            "boundary_logits": boundary,
            "coarse_logits": self.coarse_classifier(x) if self.coarse_classifier is not None else None,
            "tile_presence_logits": presence,
            "features": x,
        }
