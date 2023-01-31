"""
Definition of the learning categories
(c) 2023 tsm
"""
from dataclasses import dataclass, field
from typing import List

from .typechecking import enforce_types


@enforce_types
@dataclass(frozen=True, order=True)
class LearningCategory:
    """
    Describes the feature category. The category will be used to drive what sort of layers and learning models
    can be used on a specific feature.
    """
    key: int = field(repr=False)
    name: str
    default_panda_type: str = field(repr=False)
    sort_index: int = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'sort_index', self.key)


LEARNING_CATEGORY_BINARY: LearningCategory = LearningCategory(0, 'Binary', 'int8')
LEARNING_CATEGORY_CATEGORICAL: LearningCategory = LearningCategory(1, 'Categorical', 'int32')
LEARNING_CATEGORY_CONTINUOUS: LearningCategory = LearningCategory(2, 'Continuous', 'float64')
LEARNING_CATEGORY_LABEL: LearningCategory = LearningCategory(3, 'Label', 'float32')
LEARNING_CATEGORY_NONE: LearningCategory = LearningCategory(4, 'None', 'None')

LEARNING_CATEGORIES: List[LearningCategory] = [
    LEARNING_CATEGORY_BINARY,
    LEARNING_CATEGORY_CONTINUOUS,
    LEARNING_CATEGORY_CATEGORICAL,
    LEARNING_CATEGORY_LABEL,
    LEARNING_CATEGORY_NONE
]

# List of Learning Categories that can be used in models. Should be all LCs excluding the None.
LEARNING_CATEGORIES_MODEL: List[LearningCategory] = [
    lc for lc in LEARNING_CATEGORIES if lc.name != 'None'
]

# List of Learning Categories that can be used in models as input. Should be all LCs excluding the None and the labels
LEARNING_CATEGORIES_MODEL_INPUT: List[LearningCategory] = [
    lc for lc in LEARNING_CATEGORIES_MODEL if lc.name != 'Label'
]
