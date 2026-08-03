"""Turning module results into fraud signals.

The interesting problem here is not extraction, it is **not missing anything**.
Every module emits warnings and error codes; if a new one appears and nobody
adds it to the catalogue, an additive risk engine silently ignores it and no
test notices, because the request still produces a plausible-looking score.

So an unrecognised code is not dropped. It is collected, reported on the
response as ``unrecognised_findings``, and logged - a gap in the risk engine
rather than a property of the request. Codes that are deliberately not scored
live in :data:`~hamqadam_ai.fraud_detection.signals.BENIGN_CODES`, so the
distinction between "decided this is harmless" and "nobody has looked at this"
survives in the code rather than in somebody's memory.

Duck-typed on purpose
---------------------
The collector reads results by attribute, not by importing every response
model. Module 9 sits downstream of seven modules; importing all of their
schemas would make it the most coupled component in the service, and would
force a change here every time one of them gained a field. It reads what it
needs and tolerates absence, because a caller legitimately may not have run
every stage.
"""

from __future__ import annotations

from typing import Any

from hamqadam_ai.fraud_detection.signals import (
    FraudSignal,
    SignalDefinition,
    SignalFamily,
    is_benign,
    is_infrastructure,
    lookup,
)

#: Extra codes this module derives itself, rather than reading from a warning.
#: They describe relationships *between* module results, which is something no
#: single upstream module is in a position to notice.
DERIVED_CODES = (
    "FACE_MISMATCH",
    "CNIC_FACE_MISMATCH",
    "SECONDARY_IMAGE_MISMATCH",
    "DUPLICATE_FACE_DETECTED",
    "DUPLICATE_FACE_REVIEW",
    "CNIC_FOREIGN_FACE_PRESENT",
)


class SignalCollector:
    """Gathers fraud signals from whichever module results are available.

    Args:
        weight_overrides: Per-code weight overrides from configuration, so an
            operator can retune the policy without a code change.
    """

    __slots__ = ("_overrides", "_signals", "_unavailable", "_unrecognised")

    def __init__(self, weight_overrides: dict[str, float] | None = None) -> None:
        self._overrides = dict(weight_overrides or {})
        self._signals: list[FraudSignal] = []
        self._unavailable: list[str] = []
        self._unrecognised: list[str] = []

    # -- Results ------------------------------------------------------------- #

    @property
    def signals(self) -> list[FraudSignal]:
        """Everything collected so far."""
        return list(self._signals)

    @property
    def unavailable(self) -> list[str]:
        """Checks that could not run, each named once.

        Deduplicated because one absent module is one absent check however
        many readers noticed: ``cnic_face`` is read twice, and counting it
        twice would understate ``assessment_confidence``.
        """
        seen: list[str] = []
        for check in self._unavailable:
            if check not in seen:
                seen.append(check)
        return seen

    @property
    def unrecognised(self) -> list[str]:
        """Finding codes with no catalogue entry."""
        return sorted(set(self._unrecognised))

    # -- Collection ---------------------------------------------------------- #

    def add_code(
        self,
        code: str,
        *,
        confidence: float = 1.0,
        stage: str = "",
        detail: dict[str, Any] | None = None,
    ) -> bool:
        """Score one finding by its code.

        Returns:
            Whether the code was recognised and scored. ``False`` covers both
            "deliberately benign" and "not in the catalogue"; the two are
            distinguished on the collector's reports rather than here.
        """
        if not code or is_benign(code):
            return False

        if is_infrastructure(code):
            # A check that could not run. Recording it as a finding would score
            # an outage as suspicion; ignoring it entirely would score the
            # resulting silence as innocence. It is neither.
            self.mark_unavailable(stage or code)
            return False

        definition = lookup(code)
        if definition is None:
            self._unrecognised.append(code)
            return False

        clamped = min(max(confidence, 0.0), 1.0)

        # One finding reported twice is still one finding. Two callers legitimately
        # observe the same thing - the pipeline sees an absent CNIC portrait
        # directly, and `collect_cnic_face` sees it in the service's own
        # warnings - so `CNIC_FACE_NOT_FOUND` and `CNIC_TOO_FEW_FIELDS` were each
        # appearing twice in the response. The score was unaffected, because the
        # aggregator takes the maximum within a family rather than summing, but a
        # reviewer shown the same objection twice reasonably reads it as two
        # separate problems. The higher confidence wins, since a second observer
        # who is more certain should not be discarded in favour of the first.
        for index, existing in enumerate(self._signals):
            if existing.code == code:
                if clamped > existing.confidence:
                    self._signals[index] = FraudSignal(
                        definition=existing.definition,
                        confidence=clamped,
                        stage=existing.stage or stage,
                        detail={**existing.detail, **dict(detail or {})},
                    )
                return True

        self._signals.append(
            FraudSignal(
                definition=self._with_override(definition),
                confidence=clamped,
                stage=stage,
                detail=dict(detail or {}),
            )
        )
        return True

    def mark_unavailable(self, check: str) -> None:  # noqa: D401
        """Record that a check could not run.

        Never contributes to the score. An absent check is not evidence of
        innocence, and treating it as such is how a fraud engine gets quietly
        disabled by an outage.
        """
        self._unavailable.append(check)

    def _with_override(self, definition: SignalDefinition) -> SignalDefinition:
        """Apply a configured weight override, if there is one."""
        override = self._overrides.get(definition.code)
        if override is None:
            return definition
        return SignalDefinition(
            code=definition.code,
            family=definition.family,
            weight=min(max(override, 0.0), 1.0),
            message=definition.message,
            decisive=definition.decisive,
        )

    # -- Module readers ------------------------------------------------------ #

    def collect_warnings(self, result: Any, *, stage: str) -> None:
        """Score every warning and error code on a module result.

        Args:
            result: Any module response carrying ``warnings`` and/or
                ``error_code``.
            stage: Which module it came from.
        """
        if result is None:
            self.mark_unavailable(stage)
            return

        for warning in getattr(result, "warnings", None) or []:
            code = getattr(warning, "code", None)
            detail = getattr(warning, "detail", None) or {}
            confidence = float(detail.get("confidence", 1.0))
            self.add_code(
                str(code), confidence=confidence, stage=stage, detail=dict(detail)
            )

        error_code = getattr(result, "error_code", None)
        if error_code is not None:
            self.add_code(str(error_code), stage=stage)

        for finding in getattr(result, "findings", None) or []:
            code = getattr(finding, "code", None)
            if code is None:
                continue
            confidence = float(getattr(finding, "confidence", 1.0) or 1.0)
            severity = str(getattr(finding, "severity", "") or "")
            # An OCR validation finding carries a severity rather than a
            # confidence. Informational ones are observations, not evidence.
            if severity == "info":
                continue
            self.add_code(str(code), confidence=confidence, stage=stage)

    def collect_matching(self, result: Any) -> None:
        """Read Module 4's identity verdicts.

        These are *relationships*, not warnings: no single comparison knows
        that the submission as a whole is inconsistent, so the codes here are
        derived rather than forwarded.
        """
        if result is None:
            self.mark_unavailable("matching")
            return

        for comparison in getattr(result, "comparisons", None) or []:
            if not getattr(comparison, "compared", False):
                continue
            if str(getattr(comparison, "decision", "")) != "FAILED":
                continue

            kind = str(getattr(comparison, "comparison", ""))
            confidence = float(getattr(comparison, "confidence", 1.0) or 1.0)
            code = {
                "cnic": "CNIC_FACE_MISMATCH",
                "secondary": "SECONDARY_IMAGE_MISMATCH",
            }.get(kind, "FACE_MISMATCH")
            self.add_code(
                code,
                confidence=confidence,
                stage="matching",
                detail={"comparison": kind},
            )

    def collect_duplicate(self, result: Any) -> None:
        """Read Module 8's gallery verdict."""
        if result is None:
            self.mark_unavailable("duplicate")
            return

        if not getattr(result, "searched", False):
            self.mark_unavailable("duplicate")
            return

        if getattr(result, "duplicate_found", False):
            self.add_code(
                "DUPLICATE_FACE_DETECTED",
                stage="duplicate",
                detail={"gallery_size": int(getattr(result, "gallery_size", 0))},
            )
        elif getattr(result, "needs_review", False):
            self.add_code(
                "DUPLICATE_FACE_REVIEW",
                stage="duplicate",
                detail={"gallery_size": int(getattr(result, "gallery_size", 0))},
            )

    def collect_cnic_face(self, result: Any) -> None:
        """Read Module 6's portrait findings.

        ``foreign_face_count`` does not arrive as a warning code in every path,
        so it is read directly. It is the signal for a card held up in front of
        somebody, which is the attack that module exists to stop.
        """
        if result is None:
            self.mark_unavailable("cnic_face")
            return

        portrait = getattr(result, "portrait", None)
        if portrait is None:
            return

        foreign = int(getattr(portrait, "foreign_face_count", 0) or 0)
        if foreign:
            self.add_code(
                "CNIC_FOREIGN_FACE_PRESENT",
                stage="cnic_face",
                detail={"count": foreign},
            )

        if not getattr(portrait, "found", True):
            self.add_code("CNIC_FACE_NOT_FOUND", stage="cnic_face")


__all__ = ["DERIVED_CODES", "SignalCollector", "SignalFamily"]
