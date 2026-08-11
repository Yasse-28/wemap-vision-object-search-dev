"""The two localize entry points must agree on their parameters.

`/localize` takes JSON for a text query and multipart for an image one. The image
branch used to build `LocalizationParams` by hand from two form fields, so every
other knob the UI sent — `min_similarity`, `clustering_eps_m`,
`min_keyframes_per_cluster` — was silently dropped. Both branches now validate the
same `LocalizeParams` model; these tests are what keeps them from drifting apart
again.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolbox.bricks.service import LocalizeParams, LocalizeRequest

# A multipart form as Starlette hands it over: every value is a string.
_IMAGE_FORM = {
    "num_results": "50",
    "candidate_count": "800",
    "min_similarity": "0.25",
    "clustering_eps_m": "2.5",
    "min_keyframes_per_cluster": "3",
    "max_observations_per_cluster": "1000",
    "level_strategy": "median",
    "association": "incremental",
    "combination": "conjunctive",
    "association_sim_threshold": "1.25",
    "descriptor": "seed",
}


def test_form_strings_reach_the_localization_params() -> None:
    """The regression itself: a knob sent as a form field must arrive as a number."""
    params = LocalizeParams.model_validate(_IMAGE_FORM).to_params()

    assert params.min_similarity == 0.25
    assert params.clustering_eps_m == 2.5
    assert params.min_keyframes_per_cluster == 3
    assert params.max_observations_per_cluster == 1000
    assert params.level_strategy == "median"
    assert params.association == "incremental"
    assert params.combination == "conjunctive"
    assert params.association_sim_threshold == 1.25
    assert params.descriptor == "seed"


def test_both_branches_share_one_set_of_defaults() -> None:
    """An omitted field must mean the same thing whichever branch received it."""
    assert LocalizeParams().to_params() == LocalizeRequest(text="a").to_params()


def test_text_is_required_on_the_json_branch_only() -> None:
    """`LocalizeParams` exists to be the query-less half — it must stay that way."""
    LocalizeParams.model_validate({"num_results": 10})
    with pytest.raises(ValidationError):
        LocalizeRequest.model_validate({"num_results": 10})


def test_unknown_form_entries_are_ignored() -> None:
    """`text` rides along in the image form, and callers send retired knobs."""
    params = LocalizeParams.model_validate(
        {**_IMAGE_FORM, "text": "query.jpg", "robust_centroid": "false"}
    ).to_params()
    assert params.num_results == 50


def test_an_invalid_normalization_is_rejected_not_ignored() -> None:
    """A typo must fail loudly rather than silently measure the raw term."""
    with pytest.raises(ValidationError):
        LocalizeParams.model_validate({"feedback_normalization": "zscore"})
