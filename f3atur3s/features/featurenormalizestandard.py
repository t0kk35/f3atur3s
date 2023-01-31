"""
Definition of the Standardizing Normalizer feature. It is a feature directly found in a source.
(c) 2023 tsm
"""
from dataclasses import dataclass

from ..common.typechecking import enforce_types
from ..common.feature import FeatureNormalizeLogBase


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureNormalizeStandard(FeatureNormalizeLogBase):
    """
    Normalizing feature. Feature that standardises a base feature around mean zero and unit standard deviation.
    """
    mean: float = None
    stddev: float = None

    def __post_init__(self):
        self.val_float_type()
        self.val_base_feature_is_float()
        self.log_base_valid()
        # By default, return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    @property
    def inference_ready(self) -> bool:
        return self.mean is not None and self.stddev is not None
