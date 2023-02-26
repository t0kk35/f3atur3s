"""
Definition of the Indexing feature. It is a feature directly found in a source.
(c) 2023 tsm
"""
from dataclasses import dataclass
from typing import List, Dict, Any

from ..common.typechecking import enforce_types
from ..common.feature import Feature, FeatureWithBaseFeature, FeatureExpander, LearningCategory
from ..common.featuretype import FeatureTypeHelper
from ..common.learningcategory import LEARNING_CATEGORY_BINARY
from .featurevirtual import FeatureVirtual


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureOneHot(FeatureExpander):
    """
    A One Hot feature. This will take a base feature and one hot encode it. It will create as many additional
    virtual features as there are input values. The virtual feature will have a specific name for instance
    <base_feature>__<input_value>
    """
    def __post_init__(self):
        self.val_int_type()
        self.val_base_feature_is_string_or_integer()
        # By default, return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    def expand(self) -> List[FeatureVirtual]:
        if self.expand_names is not None:
            return [FeatureVirtual(name=n, type=self.type) for n in self.expand_names]
        else:
            return []

    @property
    def inference_ready(self) -> bool:
        return self.expand_names is not None

    @property
    def learning_category(self) -> LearningCategory:
        # Treat One Hot Features as 'Binary' learning category. Even though they are encoded as integers.
        return LEARNING_CATEGORY_BINARY

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List[Feature], pkl: Any) -> 'FeatureOneHot':
        name, tp, fb = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        oh = FeatureOneHot(name, tp, fb)
        oh.expand_names = fields['expand_names']
        return oh
