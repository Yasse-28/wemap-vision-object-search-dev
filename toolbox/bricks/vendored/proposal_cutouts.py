"""Memory-bounded `create_proposal_cutouts`.

Vendored from `third_party/object_search/prepare/proposal_cutouts.py` — see
`PROVENANCE.md`. **Only the orchestration function is copied**; every geometric
helper (`erp_pixel_centers_to_spherical`, `build_base_grid`,
`gnomonic_projection_batch`, `spherical_to_grid`, `build_padding_mask`) and
`ProposalCutout` itself are imported from the mirror, so the projection maths stays
single-sourced and a backend change to it is inherited here for free.

## Why this copy exists

The mirrored function OOMs on an 8 GB card. Measured on a 5760x2880 ERP with
MetaCLIP2 worldwide-huge + YOLO-World + GroundingDINO resident (4.15 GiB of
weights, 6.80 GiB visible after detection, so ~0.85 GiB free):

    img_batch = img.repeat(grid_chunk.shape[0], 1, 1, 1)   # 10 x 199 MB = 1.85 GiB

Two problems, and the second is the one that actually kills it:

1. `BATCH = 10` is a local literal with no way to lower it. Its own comment —
   "BATCH=10 => ~2GB of memory for 5.6k RGB ERP images" — states the budget, which
   simply does not fit here.
2. **The loop transiently holds two copies.** `img_batch = img.repeat(...)`
   evaluates the new 1.85 GiB tensor *before* rebinding the name, so the previous
   iteration's tensor is still alive. The real requirement is ~3.7 GiB, not 1.85.

Measured peak allocation, same image, same 67 proposals:

| BATCH | as mirrored | with `del img_batch` |
|---|---|---|
| 10 | **OOM** | 6.88 GiB |
| 4 | 6.51 GiB | 5.77 GiB |
| 2 | 5.77 GiB | 5.40 GiB |
| 1 | 5.40 GiB | 5.21 GiB |

`peak(BATCH=n, as mirrored) == peak(BATCH=2n, with del)` throughout — the
signature of the double allocation.

## The `del` is necessary but not sufficient, and the default stays 10 anyway

On a *single* image the `del` is enough. Across a run it is not: repeatedly
allocating and freeing 1.85 GiB blocks fragments the caching allocator, and a later
image fails with 4.99 GiB allocated but **2.14 GiB reserved-but-unallocated** — the
free memory exists, not as one contiguous block. Measured on 8 GB, `--limit 8`:

| Config | Result |
|---|---|
| `batch=10`, no env | OOM on a later image (fragmentation) |
| `batch=10` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 8 images, 807 rows |
| `batch=4`, no env | 8 images, 807 rows |
| `batch=2`, no env | 8 images, 807 rows |

All three succeeding runs produced identical `metadata.parquet` and the same
`embeddings.npy` SHA-256, which is the end-to-end evidence for the output claim
below.

The default is left at 10 so behaviour matches production out of the box; an 8 GB
card needs either the env var or a lower `--cutout-batch`.

## The two deltas

1. `del img_batch` at the end of each iteration.
2. `batch` is a keyword parameter, defaulting to `DEFAULT_CUTOUT_BATCH = 10` —
   production's literal, unchanged.

Both are **memory-only**. Output is bitwise identical to the mirror's, which
`toolbox/tests/test_proposal_cutouts.py` asserts on CPU.

The double allocation wastes peak memory on *every* GPU, not just small ones, so
this belongs in `wemap-vision-backend`; until it lands there, this override is what
lets the pipeline run locally. See `../../../docs/adr/0002-align-on-backend-pipeline.md`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from prepare.proposal_cutouts import (
    ProposalCutout,
    build_base_grid,
    build_padding_mask,
    erp_pixel_centers_to_spherical,
    gnomonic_projection_batch,
    spherical_to_grid,
)

# Production's literal, deliberately unchanged: with the `del` in place it fits.
DEFAULT_CUTOUT_BATCH = 10


def create_proposal_cutouts(
    img_np: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    device: str,
    out_size: int,
    *,
    batch: int = DEFAULT_CUTOUT_BATCH,
) -> list[ProposalCutout]:
    """Gnomonic cutout per proposal. Mirror-identical output, bounded peak memory.

    Args:
        img_np: ERP image, `(H, W, 3)` uint8.
        bboxes: Proposal boxes as `(x1, y1, x2, y2)` in ERP pixels.
        device: Torch device for the projection and sampling.
        out_size: Square cutout side, i.e. the embedder's input resolution.
        batch: Proposals sampled per `grid_sample` call. Peak memory is
            `batch x H x W x 3 x 4` bytes for the replicated ERP, so lower it on a
            small card. Defaults to production's value.

    Returns:
        One `ProposalCutout` per box, in input order.

    Raises:
        ValueError: `batch` is not positive.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}.")
    if not bboxes:
        return []

    # Ensure writable contiguous memory before converting to torch tensor.
    img_np = np.array(img_np, copy=True, order="C")
    img = torch.from_numpy(img_np).to(device).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    H, W = img_np.shape[:2]

    boxes = torch.tensor(bboxes, device=device).float()

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2

    theta, phi = erp_pixel_centers_to_spherical(cx, cy, W, H)

    w_pix = boxes[:, 2] - boxes[:, 0]
    h_pix = boxes[:, 3] - boxes[:, 1]

    angular_w = torch.clamp(w_pix / W * 2 * torch.pi, min=1e-4)
    angular_h = torch.clamp(h_pix / H * torch.pi, min=1e-4)

    ratio = w_pix / (h_pix + 1e-6)

    fov_x = angular_w.clone()
    fov_y = angular_h.clone()

    wide = ratio > 1
    tall = ~wide

    fov_y[wide] = fov_x[wide] / ratio[wide]
    fov_x[tall] = fov_y[tall] * ratio[tall]

    base_x, base_y = build_base_grid(out_size, device)

    theta_map, phi_map = gnomonic_projection_batch(
        theta, phi, fov_x, fov_y, base_x, base_y
    )

    grid = spherical_to_grid(theta_map, phi_map)

    all_patches = []
    for start in range(0, grid.shape[0], batch):
        grid_chunk = grid[start : start + batch]

        img_batch = img.repeat(grid_chunk.shape[0], 1, 1, 1)

        all_patches.append(
            F.grid_sample(
                img_batch,
                grid_chunk,
                mode="bilinear",
                align_corners=False,
            )
        )

        # DELTA vs the mirror. Without it the next iteration's `img.repeat(...)` is
        # evaluated while this tensor is still bound, so two full ERP replicas are
        # alive at once and the peak doubles. `grid_sample` returned a new tensor,
        # so nothing in all_patches references this one.
        del img_batch

    patches = torch.cat(all_patches, dim=0)

    # Keep the original proposal aspect ratio with zero padding.
    patches = patches * build_padding_mask(base_x, base_y, ratio)

    patch_images = (
        (patches.clamp(0, 1) * 255.0)
        .detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .to(torch.uint8)
        .numpy()
    )
    return [
        ProposalCutout(
            image=patch_image,
            theta_center=float(theta_i.item()),
            phi_center=float(phi_i.item()),
            angular_width=float(angular_w_i.item()),
            angular_height=float(angular_h_i.item()),
        )
        for patch_image, theta_i, phi_i, angular_w_i, angular_h_i in zip(
            patch_images, theta, phi, angular_w, angular_h
        )
    ]


def install(batch: int = DEFAULT_CUTOUT_BATCH) -> None:
    """Point the mirrored `prepare` pipeline at this implementation.

    `run_prepare` resolves `create_proposal_cutouts` from its own module globals, so
    rebinding the name there is what makes the override take effect. Done explicitly
    from `prepare_runner` rather than on import, so nothing changes behaviour merely
    by being imported.

    Args:
        batch: Proposals per `grid_sample` call; see `create_proposal_cutouts`.

    Raises:
        ValueError: `batch` is not positive.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}.")

    import prepare.pipeline

    def patched(
        img_np: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        device: str,
        out_size: int,
    ) -> list[ProposalCutout]:
        return create_proposal_cutouts(img_np, bboxes, device, out_size, batch=batch)

    prepare.pipeline.create_proposal_cutouts = patched
