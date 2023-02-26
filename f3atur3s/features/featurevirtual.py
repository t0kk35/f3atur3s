"""
Definition of the virtual feature. It is a feature with an identity problem
(c) 2023 tsm
"""
from dataclasses import dataclass
from typing import Dict, Any, List

from ..common.typechecking import enforce_types
from ..common.feature import Feature
from ..common.learningcategory import LearningCategory, LEARNING_CATEGORY_NONE


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureVirtual(Feature):
    """
    A placeholder feature without actual definition. Sometimes we might want to refer to a feature that is not
    an actual feature. Fluffy, true, this is a feature without actually being one.
    Virtual features should be created by providing a name and type
    """
    @property
    def inference_ready(self) -> bool:
        # A virtual feature has no inference attributes
        return True

    @property
    def learning_category(self) -> LearningCategory:
        # Virtual features are never used for learning. No matter what their type is.
        return LEARNING_CATEGORY_NONE

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List['Feature'], pkl: Any) -> 'FeatureVirtual':
        name, tp = cls.extract_dict(fields, embedded_features)
        return FeatureVirtual(name, tp)
