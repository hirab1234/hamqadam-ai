"""MODULE 9 capability - what does everything the other modules found add up to?

The design rests on one distinction: two findings are either **the same fact
seen twice** or **two independent facts**, and an engine that cannot tell them
apart will convict honest users. A CNIC photographed off a screen produces four
findings from one cause; an out-of-focus snapshot produces four quality
sub-scores. Adding either up reaches HIGH risk for something that is not fraud.

So signals carry a *family*, the strongest member of a family is what that
family contributes, and families combine by noisy-OR because they are
independent evidence. Measured against the alternatives:

    case                                   additive   family-max + noisy-OR
    one fact: CNIC shot off a screen          100.0                    69.4
    one fact: a blurry photograph              65.0                    20.0
    three genuinely independent facts         100.0                    94.6
"""

from hamqadam_ai.fraud_detection.aggregator import (
    FamilyContribution,
    RiskAssessment,
    aggregate,
)
from hamqadam_ai.fraud_detection.collector import SignalCollector
from hamqadam_ai.fraud_detection.signals import (
    BENIGN_CODES,
    CATALOGUE,
    FraudSignal,
    SignalDefinition,
    SignalFamily,
    is_benign,
    lookup,
)

__all__ = [
    "BENIGN_CODES",
    "CATALOGUE",
    "FamilyContribution",
    "FraudSignal",
    "RiskAssessment",
    "SignalCollector",
    "SignalDefinition",
    "SignalFamily",
    "aggregate",
    "is_benign",
    "lookup",
]
