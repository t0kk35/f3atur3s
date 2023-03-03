"""
Definition of the concatenating feature. It is a feature directly that concatenates 2 string dataframebuilder.
(c) 2023 tsm
"""
from dataclasses import dataclass
from typing import Dict, Any, List

from ..common.typechecking import enforce_types
from ..common.feature import FeatureWithBaseFeature, Feature, LearningCategory, FeatureTypeString
from ..common.feature import FeatureDefinitionException, FeatureWithBaseFeature


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureConcat(FeatureWithBaseFeature):
    """
    Feature to concatenate 2 dataframebuilder. Both feature must be string type, the result will be a string
    """
    concat_feature: Feature

    def __post_init__(self):
        self.val_string_type()
        self.val_base_feature_is_string()
        self._val_concat_feature_is_string()
        # Add base and concat to the embedded dataframebuilder list
        self.embedded_features = self.get_base_and_base_embedded_features()
        self.embedded_features.append(self.concat_feature)
        self.embedded_features.extend(self.concat_feature.embedded_features)

    def __dict__(self) -> Dict[str, Any]:
        json = super().__dict__()
        # Need to keep the name only of the concat feature
        json['concat_feature'] = json['concat_feature']['name']
        return json

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the learning category of the type of the source feature
        return self.type.learning_category

    @property
    def inference_ready(self) -> bool:
        # A concat feature has no inference attributes
        return True

    def _val_concat_feature_is_string(self):
        if not isinstance(self.concat_feature.type, FeatureTypeString):
            raise FeatureDefinitionException(
                f'The concat feature {self.concat_feature.name} of a FeatureRatio should be a string. ' +
                f'Got {self.concat_feature.type} '
            )

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List['Feature'], pkl: Any) -> 'FeatureConcat':
        n, tp, fb = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        cf = [f for f in embedded_features if f.name == fields['concat_feature']][0]
        return FeatureConcat(n, tp, fb, cf)
