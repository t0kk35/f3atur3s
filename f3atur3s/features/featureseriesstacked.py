"""
Definition of the Stacked Series feature.
(c) 2023 tsm
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List

from ..common.learningcategory import LearningCategory, LEARNING_CATEGORY_NONE
from ..common.feature import Feature, FeatureSeriesBased
from ..common.typechecking import enforce_types


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureSeriesStacked(FeatureSeriesBased):
    """
    Feature that builds out a stacked series.
    """
    series_depth: int
    key_feature: Feature
    series_features: List[Feature] = field(default_factory=list, hash=False)

    def __post_init__(self):
        # Do validation
        self.embedded_features.extend(self.series_features)
        for sf in self.param_features:
            self.embedded_features.extend(sf.embedded_features)
        self.embedded_features = list(set(self.embedded_features))

    def __dict__(self) -> Dict[str, Any]:
        json = super().__dict__()
        # Only need the name of the key_feature
        json['key_feature'] = json['key_feature']['name']
        return json

    @classmethod
    def create_from_save(
            cls, fields: Dict[str, Any], embedded_features: List['Feature'], pkl: Any) -> 'FeatureSeriesStacked':
        pass

    @property
    def inference_ready(self) -> bool:
        return all(f.inference_ready for f in self.series_features)

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_NONE
