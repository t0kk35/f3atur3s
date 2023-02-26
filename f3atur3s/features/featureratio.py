"""
Definition of the Ratio feature. It calculated a ratio between 2 other numerical features.
(c) 2023 tsm
"""
from dataclasses import dataclass
from typing import Dict, Any, List

from ..common.typechecking import enforce_types
from ..common.feature import Feature, FeatureWithBaseFeature, Feature
from ..common.learningcategory import LearningCategory
from ..common.featuretype import FeatureTypeNumerical
from ..common.exception import FeatureDefinitionException


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureRatio(FeatureWithBaseFeature):
    """
    Feature to calculate a ratio between 2 numbers. It will take the first input number and divide it by the second.
    It will avoid division by 0. If 0 is the denominator, the result will be 0 and not an error.
    """
    denominator_feature: Feature

    def __post_init__(self):
        self.val_float_type()
        self.val_base_feature_is_numerical()
        self._val_denominator_is_numerical()
        # Add base and denominator to the embedded features list
        self.embedded_features = self.get_base_and_base_embedded_features()
        self.embedded_features.append(self.denominator_feature)
        self.embedded_features.extend(self.denominator_feature.embedded_features)
        self.embedded_features = list(set(self.embedded_features))

    def __dict__(self) -> Dict[str, Any]:
        json = super().__dict__()
        # Just need to keep the name of the denominator_features
        json['denominator_feature'] = json['denominator_feature']['name']
        return json

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the learning category of the type of the source feature
        return self.type.learning_category

    @property
    def inference_ready(self) -> bool:
        # A ratio feature has no inference attributes
        return True

    def _val_denominator_is_numerical(self):
        if not isinstance(self.denominator_feature.type, FeatureTypeNumerical):
            raise FeatureDefinitionException(
                f'The denominator feature {self.denominator_feature.name} of a FeatureRatio should be numerical. ' +
                f'Got {self.denominator_feature.type} '
            )

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List[Feature], pkl: Any) -> 'FeatureRatio':
        name, tp, fb = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        fd = [f for f in embedded_features if f.name == fields['denominator_feature']]
        return FeatureRatio(name, tp, fb, fd[0])
