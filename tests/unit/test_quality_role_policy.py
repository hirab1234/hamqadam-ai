"""Per-role quality policy: the composite floor *and* the critical floor.

Module 2 already varied ``min_overall`` by role. Building Module 6 showed that
half the policy was still role-blind: ``critical_components`` was global, so a
CNIC portrait tripped the focus floor and was marked unusable before its own
carefully-set threshold of 25.0 was ever consulted.

That is not a hypothetical. Measured on a print-degraded reference face:

    downsample   quality  sharpness   cosine(same)  cosine(impostor)
    1x              89.3      67.93          0.976             0.004
    4x              67.2       0.14          0.710            -0.009
    6x              66.9       0.00          0.551            -0.028

Sharpness reaches zero while ArcFace still separates the same person at 0.551
from an impostor at -0.028, against a CNIC strong-match threshold of 0.42.
Enforcing the focus floor there refuses a decisive match on the grounds that a
print looks like a print.
"""

from __future__ import annotations

import pytest

from hamqadam_ai.core.config import (
    QualityAggregationConfig,
    QualityConfig,
    RoleQualityConfig,
    get_settings,
)
from hamqadam_ai.quality.aggregator import QualityAggregator
from hamqadam_ai.quality.base import MetricResult


def metric(name: str, score: float) -> MetricResult:
    """One measured metric family at a given 0-1 score."""
    return MetricResult(name=name, score=score, measured=True)


#: A set where everything is comfortable except focus, which is on the floor -
#: the signature of a printed portrait.
PRINT_LIKE = [
    metric("blur", 0.62),
    metric("sharpness", 0.003),
    metric("brightness", 0.90),
    metric("contrast", 0.78),
    metric("noise", 0.95),
    metric("resolution", 0.71),
    metric("pixelation", 0.88),
    metric("distortion", 0.96),
]


@pytest.fixture
def aggregator() -> QualityAggregator:
    return QualityAggregator(QualityAggregationConfig())


# --------------------------------------------------------------------------- #
# The aggregator
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_global_critical_list_applies_by_default(
    aggregator: QualityAggregator,
) -> None:
    assessment = aggregator.aggregate(PRINT_LIKE, min_required=25.0)

    assert "sharpness" in assessment.critical_failures
    assert assessment.usable is False


@pytest.mark.unit
def test_a_role_can_narrow_what_counts_as_critical(
    aggregator: QualityAggregator,
) -> None:
    """The same image, judged as a print rather than as a live capture."""
    assessment = aggregator.aggregate(
        PRINT_LIKE, min_required=25.0, critical_components=["resolution"]
    )

    assert assessment.critical_failures == []
    assert assessment.usable is True


@pytest.mark.unit
def test_narrowing_the_list_does_not_change_the_score(
    aggregator: QualityAggregator,
) -> None:
    """Only the verdict moves. A print still scores like a print, and the
    caller can still see that its focus is on the floor."""
    strict = aggregator.aggregate(PRINT_LIKE, min_required=25.0)
    lenient = aggregator.aggregate(
        PRINT_LIKE, min_required=25.0, critical_components=["resolution"]
    )

    assert strict.overall_score == pytest.approx(lenient.overall_score)


@pytest.mark.unit
def test_the_composite_floor_still_bites(aggregator: QualityAggregator) -> None:
    """Excusing focus must not excuse everything. A card genuinely
    photographed out of focus fails on the composite, which is why the role
    threshold has to stay meaningful rather than be set to zero."""
    ruined = [metric(m.name, 0.05) for m in PRINT_LIKE]

    assessment = aggregator.aggregate(
        ruined, min_required=25.0, critical_components=["resolution"]
    )

    assert assessment.usable is False


@pytest.mark.unit
def test_resolution_stays_critical_for_a_print(
    aggregator: QualityAggregator,
) -> None:
    """A portrait genuinely too small to resolve is a real failure, not a
    property of the medium - so it keeps its floor."""
    too_small = [
        metric(m.name, 0.02 if m.name == "resolution" else m.score)
        for m in PRINT_LIKE
    ]

    assessment = aggregator.aggregate(
        too_small, min_required=25.0, critical_components=["resolution"]
    )

    assert "resolution" in assessment.critical_failures
    assert assessment.usable is False


@pytest.mark.unit
def test_an_empty_critical_list_enforces_nothing(
    aggregator: QualityAggregator,
) -> None:
    """Distinct from ``None``, which inherits the global list. A deployment
    saying "enforce no critical floor for this role" must be obeyed."""
    assessment = aggregator.aggregate(
        PRINT_LIKE, min_required=25.0, critical_components=[]
    )

    assert assessment.critical_failures == []


# --------------------------------------------------------------------------- #
# The configuration lookup
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_an_unlisted_role_inherits_the_global_list() -> None:
    config = QualityConfig(
        aggregation=QualityAggregationConfig(
            critical_components=["blur", "sharpness", "resolution"]
        ),
        roles={},
    )
    assert config.critical_components_for("anything") == [
        "blur", "sharpness", "resolution"
    ]


@pytest.mark.unit
def test_a_role_without_an_override_inherits_the_global_list() -> None:
    config = QualityConfig(roles={"live_selfie": RoleQualityConfig(min_overall=55.0)})
    assert config.critical_components_for("live_selfie") == list(
        config.aggregation.critical_components
    )


@pytest.mark.unit
def test_a_role_override_wins() -> None:
    config = QualityConfig(
        roles={
            "cnic_portrait": RoleQualityConfig(
                min_overall=25.0, critical_components=["resolution"]
            )
        }
    )
    assert config.critical_components_for("cnic_portrait") == ["resolution"]


# --------------------------------------------------------------------------- #
# The shipped configuration
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_the_shipped_config_excuses_focus_for_the_cnic_roles() -> None:
    """The deployed policy, not just the mechanism. This is the assertion that
    would fail if somebody reverted the YAML without reverting the reasoning.
    """
    quality = get_settings().quality

    for role in ("cnic_image", "cnic_portrait"):
        enforced = quality.critical_components_for(role)
        assert "sharpness" not in enforced
        assert "blur" not in enforced
        assert "resolution" in enforced


@pytest.mark.unit
def test_the_shipped_config_still_enforces_focus_on_live_captures() -> None:
    """Where softness means camera shake and the image really is worthless."""
    enforced = get_settings().quality.critical_components_for("live_selfie")

    assert "sharpness" in enforced
    assert "blur" in enforced


@pytest.mark.unit
def test_a_print_is_held_to_a_lower_composite_bar_than_a_selfie() -> None:
    quality = get_settings().quality

    assert quality.min_overall_for("cnic_portrait") < quality.min_overall_for(
        "live_selfie"
    )
