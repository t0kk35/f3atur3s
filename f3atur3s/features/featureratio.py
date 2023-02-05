"""
Definition of the Ratio feature. It calculated a ratio between 2 other numerical features.
(c) 2023 tsm
"""
from dataclasses import dataclass

from ..common.typechecking import enforce_types
from ..common.feature import FeatureWithBaseFeature, Feature
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