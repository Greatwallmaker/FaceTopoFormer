# FaceTopoFormer

## Requirements

Python 3.9+, PyTorch 2.0+, NumPy, Pillow, and plyfile.
Install a PyTorch build appropriate for your CPU/CUDA environment, then run:

```bash
pip install -r requirements.txt
```

## Inference

Run from this directory with a compatible trained checkpoint:

```bash
python infer.py --mesh scene.ply --checkpoint model.pth --dataset sum --output prediction.npz
```

Choose `sum`, `sum-parts`, or `gw-meshseg` to match the checkpoint. The input is
a triangular PLY mesh in metres, with `vertex_indices` and six `texcoord` values
per face. Its texture image is read from the PLY `TextureFile` comment; use
`--texture texture.jpg` to supply it explicitly. Ground-truth labels are not
required. The minimal loader supports one texture image per mesh.

Add `--device cpu` for CPU inference. Dataset-specific chunk sizes are used by
default; `--chunk-size 8192` reduces memory use but changes the context available
to each face. The checkpoint may be a model state dictionary or the original
checkpoint containing a `model` state dictionary; it must match the full model.

The output stores `prediction` (zero-based class IDs), `probability` (class
probabilities), and `face_index`, all in the original PLY face order. For SUM
and SUM-Parts, add 1 to the predicted class IDs when using their original label
convention. GW-MeshSeg uses 0 for background and 1 for heritage. Pretrained
checkpoints are not included in this initial release; inference requires one.

## Code references

- [PyTorch](https://github.com/pytorch/pytorch): neural-network and attention operators.
- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3): reference for serialization-based spatial context.
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer): reference for shifted-window attention.
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter): grouped reductions used in the original implementation; this minimal version uses native PyTorch reductions.

The PTv3 and Swin links acknowledge design references; their model source files
are not bundled here.

The remaining code and resources will be released progressively after the paper is accepted.
