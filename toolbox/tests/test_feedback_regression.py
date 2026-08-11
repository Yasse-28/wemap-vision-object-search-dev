"""The baseline guard: with the feature off, localize output must not move.

This file exists to be written *before* `localize.py` is touched, and to fail if
the routing work changes anything on the default path. Everything else about this
feature is opinion; this is the part that must be provably free.

Three shapes of "off" are covered, because they are reached differently:

- a candidate that never heard of the feature (`similarity_boosted is None`) —
  every pre-existing fixture and caller;
- `alpha = beta = 0` with annotations present, so the boost is computed and
  arithmetically neutral;
- `LocalizationParams()` defaults, so nobody has to remember to disable it.
"""

from __future__ import annotations

from typing import Any

from toolbox.bricks.candidates import EnrichedCandidate, apply_feedback_boost
from toolbox.bricks.localize import (
    LocalizationParams,
    build_localize_response,
    localize_from_enriched_candidates,
)
from toolbox.bricks.vendored.geo_transform import Coordinates, GeoTransform, Level, Pose
from toolbox.bricks.vendored.maths import quaternion, vector3


def _geo_transform() -> GeoTransform:
    return GeoTransform(
        origin=Coordinates(lng=2.3522, lat=48.8566, alt=35.0),
        levels=(Level(value=0, min_altitude=-2.0, max_altitude=4.0),),
    )


def _candidate(
    candidate_id: int,
    eus_xyz: tuple[float, float, float],
    keyframe_id: int,
    similarity: float,
    *,
    similarity_boosted: float | None = None,
    pos_sim: float = 0.0,
    neg_sim: float = 0.0,
) -> EnrichedCandidate:
    pose = Pose.from_position_orientation(
        vector3.from_xyz(0.0, 0.0, 0.0), quaternion.identity()
    )
    return EnrichedCandidate(
        id=candidate_id,
        similarity=similarity,
        eus_xyz=eus_xyz,
        lat=48.8566,
        lng=2.3522,
        alt=36.0,
        level=0,
        video_keyframe_id=keyframe_id,
        theta_center=0.1,
        phi_center=-0.2,
        geokeyframe_pose=pose,
        thumbnail="thumbs/1.jpg",
        angular_width=1.0,
        angular_height=0.8,
        vkf_lat=48.8565,
        vkf_lng=2.3521,
        vkf_alt=35.5,
        vkf_level=0,
        video_keyframe_filename="kf.jpg",
        video_keyframe_heading=12.0,
        video_keyframe_depth="kf.tif",
        similarity_boosted=similarity_boosted,
        pos_sim=pos_sim,
        neg_sim=neg_sim,
    )


def _fixture(**kwargs: Any) -> list[EnrichedCandidate]:
    """Two clusters, four keyframes — enough that ranking is not a tautology."""
    return [
        _candidate(1, (0.0, 0.5, 0.0), keyframe_id=10, similarity=0.90, **kwargs),
        _candidate(2, (0.3, 0.5, 0.0), keyframe_id=11, similarity=0.85, **kwargs),
        _candidate(3, (0.2, 0.5, 0.4), keyframe_id=12, similarity=0.70, **kwargs),
        _candidate(4, (40.0, 0.5, 0.0), keyframe_id=13, similarity=0.65, **kwargs),
        _candidate(5, (40.4, 0.5, 0.0), keyframe_id=14, similarity=0.60, **kwargs),
    ]


def _localize(candidates: list[EnrichedCandidate], **param_kwargs: Any) -> list[dict]:
    return localize_from_enriched_candidates(
        candidates,
        _geo_transform(),
        LocalizationParams(min_keyframes_per_cluster=2, **param_kwargs),
    )


# ------------------------------------------------------------------- the baseline


def test_candidates_without_the_field_localize_as_before() -> None:
    """`similarity_boosted is None` must read as "use the raw similarity".

    This is the shape every fixture written before this feature has, so if the
    fallback breaks, it breaks for all existing callers at once.
    """
    result = _localize(_fixture())

    assert result, "the fixture must produce at least one cluster"
    assert all(loc["match_score"] > 0 for loc in result)
    # Two spatially separate groups, both above min_keyframes_per_cluster=2.
    assert len(result) == 2


def test_zero_alpha_beta_matches_the_no_feedback_path_exactly() -> None:
    """Annotations present, gains zero ⇒ byte-identical localizations."""
    baseline = _localize(_fixture())

    # Same candidates, but with the boost actually computed at alpha = beta = 0.
    boosted = _fixture(pos_sim=0.9, neg_sim=0.8)
    boosted = [
        EnrichedCandidate(
            **{
                **{f: getattr(c, f) for f in c.__dataclass_fields__},
                "similarity_boosted": apply_feedback_boost(
                    c.similarity, c.pos_sim, c.neg_sim, 0.0, 0.0
                ),
            }
        )
        for c in boosted
    ]
    with_feedback = _localize(boosted, feedback_alpha=0.0, feedback_beta=0.0)

    assert with_feedback == baseline


def test_default_params_disable_the_feature() -> None:
    """Nobody should have to remember to turn it off."""
    params = LocalizationParams()
    assert params.feedback_alpha == 0.0
    assert params.feedback_beta == 0.0


def test_response_envelope_is_unchanged_on_the_default_path() -> None:
    """The HTTP benchmark parses this shape directly."""
    response = build_localize_response(
        _fixture(),
        _geo_transform(),
        params=LocalizationParams(min_keyframes_per_cluster=2),
        time_embedding_ms=12,
        time_retrieval_ms=34,
    )
    assert set(response) == {"localizations", "time_embedding_ms", "time_retrieval_ms"}


def test_localize_does_not_mutate_its_input() -> None:
    """Routing work is easy to write as an in-place rewrite of the candidate list.

    Compared field-by-field rather than with `==`: `EnrichedCandidate` holds a
    `Pose`, which wraps a numpy array, so the generated `__eq__` raises
    "truth value of an array is ambiguous" instead of returning False.
    """
    candidates = _fixture()
    before = [
        (c.id, c.similarity, c.similarity_boosted, c.pos_sim, c.neg_sim)
        for c in candidates
    ]

    _localize(candidates)

    after = [
        (c.id, c.similarity, c.similarity_boosted, c.pos_sim, c.neg_sim)
        for c in candidates
    ]
    assert after == before


# ----------------------------------------------------- the feature actually works


def test_a_negative_boost_demotes_a_cluster() -> None:
    """Guard the guards: if nothing moves under beta > 0, the tests above are vacuous.

    The far cluster is penalised hard enough to fall below the near one. Only the
    *order* is asserted — how far it moves depends on the ranking weights, which
    `test_localize.py` already pins.
    """
    penalised = _fixture()
    penalised = [
        EnrichedCandidate(
            **{
                **{f: getattr(c, f) for f in c.__dataclass_fields__},
                "neg_sim": 1.0 if c.id in (1, 2, 3) else 0.0,
                "similarity_boosted": apply_feedback_boost(
                    c.similarity, 0.0, 1.0 if c.id in (1, 2, 3) else 0.0, 0.0, 0.5
                ),
            }
        )
        for c in penalised
    ]

    baseline = _localize(_fixture())
    demoted = _localize(penalised, feedback_beta=0.5)

    assert len(baseline) == len(demoted) == 2
    # The near cluster leads without feedback and trails with it.
    assert baseline[0]["coordinates"] != demoted[0]["coordinates"]


def test_clustering_geometry_is_unaffected_by_the_boost() -> None:
    """The seed order stays on raw similarity, so the clusters themselves must not move.

    Boosting the seed order would change which detections group together — a much
    larger intervention than re-ranking, and not the one this feature is.

    The penalty is deliberately gentle (0.05 × 1.0) so that no cluster crosses
    `min_similarity`. A harder one legitimately *removes* a cluster, which is the
    feature working, not the geometry moving — that case is covered separately by
    `test_a_hard_penalty_can_drop_a_cluster_below_min_similarity`.
    """
    baseline = _localize(_fixture())
    penalised = [
        EnrichedCandidate(
            **{
                **{f: getattr(c, f) for f in c.__dataclass_fields__},
                "similarity_boosted": apply_feedback_boost(
                    c.similarity, 0.0, 1.0, 0.0, 0.05
                ),
                "neg_sim": 1.0,
            }
        )
        for c in _fixture()
    ]
    boosted = _localize(penalised, feedback_beta=0.05)

    assert sorted(loc["observation_count"] for loc in baseline) == sorted(
        loc["observation_count"] for loc in boosted
    )
    assert sorted(tuple(loc["keyframe_ids"]) for loc in baseline) == sorted(
        tuple(loc["keyframe_ids"]) for loc in boosted
    )


def test_a_hard_penalty_can_drop_a_cluster_below_min_similarity() -> None:
    """The only way the boost *removes* a false positive rather than reordering it.

    `match_score` is the cluster's similarity as a ratio to the query's best, so a
    penalty that merely lowers similarity reshuffles the list — and if it hits the
    *top* cluster it rescales every score without removing anything. Pushing a cluster
    under `min_similarity` is what makes it disappear.
    """
    weak_ids = {4, 5}  # the far cluster, similarities 0.65 / 0.60
    penalised = [
        EnrichedCandidate(
            **{
                **{f: getattr(c, f) for f in c.__dataclass_fields__},
                "neg_sim": 1.0 if c.id in weak_ids else 0.0,
                "similarity_boosted": apply_feedback_boost(
                    c.similarity, 0.0, 1.0 if c.id in weak_ids else 0.0, 0.0, 0.5
                ),
            }
        )
        for c in _fixture()
    ]

    assert len(_localize(_fixture())) == 2
    assert len(_localize(penalised, feedback_beta=0.5)) == 1
