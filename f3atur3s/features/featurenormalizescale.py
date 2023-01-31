"""
Definition of the Scaling Normalizer feature. It is a feature directly found in a source.
(c) 2023 tsm
"""
from dataclasses import dataclass

from ..common.typechecking import enforce_types
from ..common.feature import FeatureNormalizeLogBase


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureNormalizeScale(FeatureNormalizeLogBase):
    """
    Normalizing feature. Feature that scales a base feature between 0 and 1 with a min/max logic.
    """
    minimum: float = None
    maximum: float = None

    def __post_init__(self):
        self.val_float_type()
        self.val_base_feature_is_float()
        self.log_base_valid()
        # By default, return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    @property
    def inference_ready(self) -> bool:
        return self.minimum is not None and self.maximum is not None
