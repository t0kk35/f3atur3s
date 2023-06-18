"""
Definition of the Time wave feature. It encodes aspects of a date-time as sines/cosine waves.
(c) 2023 tsm
"""
from dataclasses import dataclass
from typing import List, Any, Dict

from ..common.learningcategory import LearningCategory
from ..common.typechecking import enforce_types
from ..common.feature import Feature, FeatureWithBaseFeature, FeatureExpander
from .featurevirtual import FeatureVirtual


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureDateTimeWave(FeatureExpander, FeatureWithBaseFeature):
    """
    Encodes a date time feature as sine/cosine wave
    """
    format: str
    period: int
    frequencies: int

    def __post_init__(self):
        self.val_base_feature_is_time_based()
        self.val_float_type()
        # Set the expand_names
        self.expand_names = [
            f'{self.base_feature.name}{self.delimiter}{w}{self.delimiter}{f}'
            for f in range(self.frequencies) for w in ('sin', 'cos')
        ]
        # By default; return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    def expand(self) -> List[Feature]:
        if self.expand_names is not None:
            return [FeatureVirtual(name=n, type=self.type) for n in self.expand_names]
        else:
            return []

    @property
    def inference_ready(self) -> bool:
        # A DateTimeFormat feature is always ready for inference
        return True

    @property
    def delimiter(self) -> str:
        return '__'

    @property
    def learning_category(self) -> LearningCategory:
        # Return the learning category of the type.
        return self.type.learning_category

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any],
                         embedded_features: List['Feature'], pkl: Any) -> 'FeatureDateTimeWave':
        name, tp, fb = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        fmt = fields['format']
        pr = fields['period']
        fr = fields['frequencies']
        return FeatureDateTimeWave(name, tp, fb, fmt, pr, fr)
