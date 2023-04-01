"""
Definition of the Time Formatter feature. It is a feature that can re-format a date feature.
(c) 2023 tsm
"""
from dataclasses import dataclass, field
from typing import List, Any, Dict

from ..common.learningcategory import LearningCategory
from ..common.typechecking import enforce_types
from ..common.feature import Feature, FeatureWithBaseFeature, FeatureCategorical


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureDateTimeFormat(FeatureWithBaseFeature):
    """
    Feature that formats a datetime feature. It can for instance be used to extract the day-of-month from a date
    """
    format: str

    def __post_init__(self):
        self.val_base_feature_is_time_based()
        # By default; return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    @property
    def inference_ready(self) -> bool:
        # A DateTimeFormat feature is always ready for inference
        return True

    @property
    def learning_category(self) -> LearningCategory:
        # Return the learning category of the type.
        return self.type.learning_category

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any],
                         embedded_features: List['Feature'], pkl: Any) -> 'FeatureDateTimeFormat':
        name, tp, fb = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        fmt = fields['format']
        return FeatureDateTimeFormat(name, tp, fb, fmt)
