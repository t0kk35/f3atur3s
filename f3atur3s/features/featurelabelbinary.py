"""
Definition of the Binary Label feature. It is a feature directly found in a source.
(c) 2023 tsm
"""
from dataclasses import dataclass
from typing import Dict, Any, List

from ..common.typechecking import enforce_types
from ..common.feature import Feature, FeatureLabel, FeatureWithBaseFeature


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureLabelBinary(FeatureLabel):
    """
    Feature to indicate what the label needs to be during training. This feature will assume it is binary of type
    INT and will contain values 0 and 1.
    """

    def __post_init__(self):
        # Do post init validation
        self.val_int_type()
        self.val_base_feature_is_numerical()
        # By default, return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any],
                         embedded_features: List[Feature], pkl: Any) -> 'FeatureLabelBinary':
        name, tp, fb = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        flb = FeatureLabelBinary(name, tp, fb)
        return flb
