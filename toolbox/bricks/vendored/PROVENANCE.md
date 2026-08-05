# Vendored from wemap-vision-backend

These files are **copies of production code**, pulled out of the backend's Django
app so the bricks can run without Django. They are not part of the strict mirror
(the `third_party/object_search/` submodule) because they live
outside the backend's object-search tree — but they carry the same obligation:
**the backend is the source of truth. If it changes, re-sync here.**

Synced from `wemap/wemap-vision-backend` @ `365e6bc` on 2026-08-04.

| Here | Backend path | Changed? |
|---|---|---|
| `maths/{__init__,vector3,matrix3,matrix4,quaternion}.py` | `backend/utils/maths/` | Verbatim, except docstring import examples retargeted. Zero Django. |
| `geo_transform.py` | `backend/utils/geo_transform.py` | `from_geo_ref` → `from_georef_db`; GEOS `Point`/`.intersects()` → GeoJSON dict + pure-Python ray-cast. All frame/ellipsoid math verbatim. |
| `erp.py` | `backend/utils/erp.py` | Import path only. |
| `depth_decode.py` | `backend/depth/service/decode.py` | Dropped the `VideoKeyframe`/S3 readers; kept path-based read + decode math verbatim. |
| `viewer360_headings.py` | `backend/api/viewer360/v1_legacy.py` | `headings_from_orientations` only. |
| `candidate_orientation.py` | `backend/object_search/v1_legacy.py` | `candidate_orientation` only; the v1 kiosk +180° flip left behind on purpose. |
| `proposal_cutouts.py` | *(the mirror — see below)* | `create_proposal_cutouts` only: added `del img_batch`, made `BATCH` a `batch` parameter (default 10, unchanged). Memory-only; output bitwise identical. |

## The one override on mirrored code

`proposal_cutouts.py` is different in kind from every other row above: it overrides a
function that **is** in the strict mirror
(`third_party/object_search/prepare/proposal_cutouts.py`, in the submodule). The mirror itself stays
byte-identical — `check-mirror.sh` still passes — and `install()` rebinds the name in
`prepare.pipeline` at runtime instead.

It copies **only the orchestration function**. `ProposalCutout` and every geometric
helper (`erp_pixel_centers_to_spherical`, `build_base_grid`,
`gnomonic_projection_batch`, `spherical_to_grid`, `build_padding_mask`) are imported
from the mirror, so the projection maths is not duplicated and upstream changes to it
are inherited.

**This exists because the mirrored version keeps two replicated ERPs alive at once**
(`img_batch = img.repeat(...)` evaluates the new tensor before rebinding), doubling
peak GPU memory for nothing and OOMing an 8 GB card. That is a bug on any GPU. **It is
owed to `wemap-vision-backend`**; deliberately not filed there yet, on the
maintainer's call. When it lands upstream, delete this file and the `install()` call
in `prepare_runner.run`.

## Not vendored, on purpose

`backend/api/utils/spatial_sampling.py` — it is *itself* a vendored copy of
`third_party/object_search/indexing/grid.py` (its own docstring says so). We call
the mirror's `indexing.grid.filter_by_distance` directly instead, so this port
removes a vendoring rather than adding one. The two differ only in the public
wrapper: the mirror takes `(gk_id, vk_id, vc_id, x, y, z)` tuples and returns the
kept tuples; the backend copy takes an `(N, 3)` array and returns a bool mask.

## Re-syncing

```bash
BACKEND=/path/to/wemap-vision-backend
diff -u "$BACKEND/backend/utils/maths/quaternion.py" maths/quaternion.py
# …and so on per row above. Expect only the deltas listed in the "Changed?" column.
```
