"""
Definition of the FeatureTypes categories
(c) 2023 tsm
"""
from dataclasses import dataclass, field

from .typechecking import enforce_types
from .learningcategory import LearningCategory, LEARNING_CATEGORY_NONE, LEARNING_CATEGORY_CATEGORICAL
from .learningcategory import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_BINARY


@enforce_types
@dataclass(frozen=True)
class FeatureType:
    """
    Defines the datatype of a particular Feature. It will tell us what sort of data a feature is holding, like
    string values, a float value, an integer value etc... See below for the specific implementations
    """
    key: int = field(repr=False)
    name: str
    learning_category: LearningCategory = field(repr=False)
    precision: int = field(default=0, repr=False)

    @staticmethod
    def max_precision(ft1: 'FeatureType', ft2: 'FeatureType') -> 'FeatureType':
        if ft1.precision > ft2.precision:
            return ft1
        else:
            return ft2


class FeatureTypeString(FeatureType):
    pass


class FeatureTypeNumerical(FeatureType):
    pass


class FeatureTypeInteger(FeatureTypeNumerical):
    pass


class FeatureTypeFloat(FeatureTypeNumerical):
    pass


class FeatureTypeTimeBased(FeatureType):
    pass


class FeatureTypeBool(FeatureTypeNumerical):
    pass


FEATURE_TYPE_STRING: FeatureType = FeatureTypeString(1, 'STRING', LEARNING_CATEGORY_NONE)
FEATURE_TYPE_CATEGORICAL: FeatureType = FeatureTypeString(2, 'CATEGORICAL', LEARNING_CATEGORY_CATEGORICAL)
FEATURE_TYPE_FLOAT: FeatureType = FeatureTypeFloat(3, 'FLOAT', LEARNING_CATEGORY_CONTINUOUS, 64)
FEATURE_TYPE_FLOAT_32: FeatureType = FeatureTypeFloat(4, 'FLOAT_32', LEARNING_CATEGORY_CONTINUOUS, 32)
FEATURE_TYPE_FLOAT_64: FeatureType = FEATURE_TYPE_FLOAT
FEATURE_TYPE_DATE: FeatureType = FeatureTypeTimeBased(5, 'DATE', LEARNING_CATEGORY_NONE)
FEATURE_TYPE_DATE_TIME: FeatureType = FeatureTypeTimeBased(6, 'DATETIME', LEARNING_CATEGORY_NONE)
FEATURE_TYPE_INTEGER: FeatureType = FeatureTypeInteger(7, 'INTEGER', LEARNING_CATEGORY_CATEGORICAL, 32)
FEATURE_TYPE_INT_8: FeatureType = FeatureTypeInteger(8, 'INT_8', LEARNING_CATEGORY_CATEGORICAL, 8)
FEATURE_TYPE_INT_16: FeatureType = FeatureTypeInteger(9, 'INT_16', LEARNING_CATEGORY_CATEGORICAL, 16)
FEATURE_TYPE_INT_32: FeatureType = FEATURE_TYPE_INTEGER
FEATURE_TYPE_INT_64: FeatureType = FeatureTypeInteger(10, 'INT_64', LEARNING_CATEGORY_CATEGORICAL, 64)
FEATURE_TYPE_BOOL: FeatureType = FeatureTypeBool(11, 'INT_8', LEARNING_CATEGORY_BINARY, 8)
