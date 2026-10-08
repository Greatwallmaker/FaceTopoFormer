"""Minimal textured-triangle-mesh inference, preserving original face indices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from model import FaceTopoFormer, PRESETS


def morton_order(points):
    """FRDO: stable 10-bit Morton ordering of face centroids."""
    span = np.maximum(points.max(0) - points.min(0), 1e-6)
    q = np.floor(np.clip((points - points.min(0)) / span, 0, 1) * 1023).astype(np.uint64)
    code = np.zeros(len(points), dtype=np.uint64)
    for bit in range(10):
        for axis in range(3):
            code |= ((q[:, axis] >> bit) & 1) << (3 * bit + axis)
    return np.argsort(code, kind="stable")


def face_neighbors(faces):
    """Shared-edge adjacency, using the trained model's 12-slot convention."""
    pairs = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), 1).reshape(-1, 2)
    pairs.sort(axis=1)
    owners = np.repeat(np.arange(len(faces)), 3)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs, owners = pairs[order], owners[order]
    starts = np.r_[0, np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(pairs)]
    counts = ends - starts
    a, b = owners[starts[counts == 2]], owners[starts[counts == 2] + 1]
    src, dst = [a, b], [b, a]
    extra_src, extra_dst = [], []
    for begin, end in zip(starts[counts > 2], ends[counts > 2]):
        group = np.unique(owners[begin:end])
        for first in group:
            for second in group:
                if first != second:
                    extra_src.append(first)
                    extra_dst.append(second)
    if extra_src:
        src.append(np.asarray(extra_src))
        dst.append(np.asarray(extra_dst))
    src, dst = np.concatenate(src), np.concatenate(dst)
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    _, first, degree = np.unique(src, return_index=True, return_counts=True)
    rank = np.arange(len(src)) - np.repeat(first, degree)
    table = np.full((len(faces), 12), -1, dtype=np.int64)
    keep = rank < 12
    table[src[keep], rank[keep]] = dst[keep]
    return table


def load_mesh(path, texture_path=None):
    """Read triangular PLY + per-face UVs. Labels and face colors are not inputs."""
    mesh = PlyData.read(str(path))
    vertices = np.column_stack([mesh["vertex"][axis] for axis in ("x", "y", "z")]).astype(np.float32)
    face_data = mesh["face"].data
    if not len(face_data):
        raise ValueError("The mesh has no faces.")
    faces = np.stack(face_data["vertex_indices"]).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Input faces must already be triangles; inference does not remesh them.")
    if faces.min() < 0 or faces.max() >= len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Invalid vertex coordinates or face indices.")
    if "texcoord" not in face_data.dtype.names:
        raise ValueError("PLY faces must contain texcoord: three (u,v) pairs per triangle.")
    uv = np.stack(face_data["texcoord"]).astype(np.float32)
    if uv.shape != (len(faces), 6) or not np.isfinite(uv).all():
        raise ValueError("Each face must have six finite UV coordinates.")
    if texture_path is None:
        candidates = [path.parent / c.split(maxsplit=1)[1].strip()
                      for c in mesh.comments if c.lower().startswith("texturefile ")]
        stem = path.stem
        for suffix in ("_groundtruth_L1", "_L2", "_L1"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        candidates.append(path.with_name(stem + "_0.jpg"))
        texture_path = next((p for p in candidates if p.is_file()), None)
    if texture_path is None or not texture_path.is_file():
        raise FileNotFoundError("Texture image not found; supply --texture path/to/texture.jpg.")
    with Image.open(texture_path) as image:
        tex = np.asarray(image.convert("RGB"))
    bary = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1/3, 1/3, 1/3]], dtype=np.float32)
    sample_uv = np.einsum("sc,fcd->fsd", bary, uv.reshape(-1, 3, 2), optimize=True)
    height, width = tex.shape[:2]
    tx = np.clip(np.rint(sample_uv[..., 0] * (width - 1)), 0, width - 1).astype(np.int64)
    ty = np.clip(np.rint((1 - sample_uv[..., 1]) * (height - 1)), 0, height - 1).astype(np.int64)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    normals, area = cross / norm, 0.5 * norm[:, 0]
    centroids = triangles.mean(1)
    lengths = np.stack([np.linalg.norm(triangles[:, b] - triangles[:, a], axis=1)
                        for a, b in ((0, 1), (1, 2), (2, 0))], 1)
    origin = np.asarray([np.median(centroids[:, 0]), np.median(centroids[:, 1]), centroids[:, 2].min()], dtype=np.float32)
    geom = np.concatenate(((centroids - origin) / 50, normals, np.log1p(area[:, None]), np.log1p(lengths)), 1)
    return {name: torch.from_numpy(value) for name, value in {
        "geom": geom, "texture": tex[ty, tx].copy(), "centroids": centroids,
        "normals": normals, "area": area, "neighbors": face_neighbors(faces),
        "order": morton_order(centroids),
    }.items()}


def tile_descriptor(tile):
    """The 28-dimensional whole-tile descriptor for GW-MeshSeg."""
    geom = tile["geom"]
    texture = tile["texture"].float().reshape(-1, 3) / 255
    return torch.cat((geom.mean(0), geom.std(0, unbiased=False),
                      texture.mean(0), texture.std(0, unbiased=False),
                      torch.stack((torch.tensor(float(len(geom))).log1p() / 15,
                                   tile["area"].sum().log1p() / 15))))


def ordered_paths(neighbors, path_count, path_length):
    """Deterministic shared-edge walks; revisits retain their step positions."""
    count, degree = neighbors.shape
    current = torch.arange(count, device=neighbors.device)[:, None].expand(count, path_count).clone()
    previous = current.clone()
    paths = torch.empty((count, path_count, path_length), dtype=torch.long, device=neighbors.device)
    paths[:, :, 0] = current
    direction = torch.arange(path_count, device=neighbors.device)[None, :]
    for step in range(1, path_length):
        candidates = neighbors[current]
        slot = (direction * 5 + step * 3) % degree
        primary = candidates.gather(2, slot.expand(count, path_count).unsqueeze(-1)).squeeze(-1)
        alternate_slot = ((slot + step + 1) % degree).expand(count, path_count).unsqueeze(-1)
        alternate = candidates.gather(2, alternate_slot).squeeze(-1)
        candidate = torch.where((primary < 0) | (primary == previous), alternate, primary)
        previous, current = current, torch.where(candidate < 0, current, candidate)
        paths[:, :, step] = current
    return paths


def make_chunk(tile, indices, preset, device, descriptor=None):
    """FRDO: gather faces, remap adjacency, and construct ordered face paths."""
    mapping = torch.full((len(tile["geom"]),), -1, dtype=torch.long)
    mapping[indices] = torch.arange(len(indices))
    neighbors = tile["neighbors"][indices]
    mapped = mapping[neighbors.clamp_min(0)]
    valid = (neighbors >= 0) & (mapped >= 0)
    local_neighbors = torch.where(valid, mapped, torch.full_like(mapped, -1))
    src = torch.arange(len(indices))[:, None].expand_as(neighbors)[valid]
    dst = mapped[valid]
    centroids, normals, area = (tile[k][indices] for k in ("centroids", "normals", "area"))
    delta = (centroids[dst] - centroids[src]) / 50
    edge_attr = torch.cat((delta, delta.norm(dim=1, keepdim=True),
                           (normals[src] * normals[dst]).sum(1, keepdim=True).clamp(-1, 1),
                           torch.cross(normals[src], normals[dst], dim=1).norm(dim=1, keepdim=True),
                           ((area[dst] + 1e-8) / (area[src] + 1e-8)).log()[:, None]), 1)
    geom = tile["geom"][indices]
    if descriptor is not None:
        geom = torch.cat((geom, descriptor[None].expand(len(indices), -1)), 1)
    return (geom.to(device), (tile["texture"][indices].float() / 255).to(device),
            torch.stack((src, dst)).to(device), edge_attr.to(device),
            ordered_paths(local_neighbors.to(device), preset.path_count, preset.path_length))


def load_model(checkpoint, dataset, device):
    model = FaceTopoFormer(dataset)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, tile, device, chunk_size):
    order = tile["order"].long()
    descriptor = tile_descriptor(tile) if model.dataset == "gw-meshseg" else None
    probability = np.empty((len(order), model.preset.num_classes), dtype=np.float32)
    for begin in range(0, len(order), chunk_size):
        indices = order[begin:begin + chunk_size]
        inputs = make_chunk(tile, indices, model.preset, device, descriptor)
        scores = model(*inputs)["logits"]
        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite model output; check the mesh and checkpoint.")
        probability[indices.numpy()] = scores.softmax(1).cpu().numpy()
        print(f"Predicted {min(begin + chunk_size, len(order))}/{len(order)} faces", flush=True)
    return probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Triangular PLY in metres, with per-face UVs")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(PRESETS), required=True)
    parser.add_argument("--output", type=Path, default=Path("prediction.npz"))
    parser.add_argument("--texture", type=Path, help="Override the PLY TextureFile image")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, help="Default: the dataset's evaluation chunk size")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        parser.error("Output already exists; select a new filename.")
    chunk_size = args.chunk_size if args.chunk_size is not None else PRESETS[args.dataset].chunk_size
    if chunk_size < 1:
        parser.error("--chunk-size must be positive")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, args.dataset, device)
    tile = load_mesh(args.mesh, args.texture)
    probability = predict(model, tile, device, chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, prediction=probability.argmax(1).astype(np.int16),
                        probability=probability.astype(np.float16),
                        face_index=np.arange(len(probability), dtype=np.int64))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
