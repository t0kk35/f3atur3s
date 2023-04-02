"""
Definition of the FeatureTypes categories
(c) 2023 tsm
"""
from dataclasses import dataclass, field
from typing import Dict

from .typechecking import enforce_types
from .exception import FeatureDefinitionException
from .learningcategory import LearningCategory, LEARNING_CATEGORY_NONE, LEARNING_CATEGORY_CATEGORICAL
from .learningcategory import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_BINARY


@enforce_types
@dataclass(frozen=True)
class FeatureRootType:
    key: int = field(repr=False)
    name: str


FEATURE_ROOT_TYPE_STRING = FeatureRootType(0, 'STRING')
FEATURE_ROOT_TYPE_FLOAT = FeatureRootType(1, 'FLOAT')
FEATURE_ROOT_TYPE_INT = FeatureRootType(1, 'INTEGER')


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
    root_type: FeatureRootType = field(repr=False)
    precision: int = field(default=0, repr=False)

    @staticmethod
    def max_precision(ft1: 'FeatureType', ft2: 'FeatureType') -> 'FeatureType':
        if ft1.precision > ft2.precision:
            return ft1
        else:
            return ft2

    @classmethod
    def get_type(cls, key: int) -> 'FeatureType':
        try:
            return ALL_FEATURE_TYPES[key]
        except KeyError:
            raise FeatureDefinitionException(f'Could not find  FeatureType with key <{key}>')


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


FEATURE_TYPE_STRING = FeatureTypeString(1, 'STRING', LEARNING_CATEGORY_NONE, FEATURE_ROOT_TYPE_STRING)
FEATURE_TYPE_CATEGORICAL = FeatureTypeString(2, 'CATEGORICAL', LEARNING_CATEGORY_NONE, FEATURE_ROOT_TYPE_INT)
FEATURE_TYPE_FLOAT = FeatureTypeFloat(3, 'FLOAT', LEARNING_CATEGORY_CONTINUOUS, FEATURE_ROOT_TYPE_FLOAT, 64)
FEATURE_TYPE_FLOAT_32 = FeatureTypeFloat(4, 'FLOAT_32', LEARNING_CATEGORY_CONTINUOUS, FEATURE_ROOT_TYPE_FLOAT, 32)
FEATURE_TYPE_FLOAT_64 = FEATURE_TYPE_FLOAT
FEATURE_TYPE_DATE = FeatureTypeTimeBased(5, 'DATE', LEARNING_CATEGORY_NONE, FEATURE_ROOT_TYPE_STRING)
FEATURE_TYPE_DATE_TIME = FeatureTypeTimeBased(6, 'DATETIME', LEARNING_CATEGORY_NONE, FEATURE_ROOT_TYPE_STRING)
FEATURE_TYPE_INTEGER = FeatureTypeInteger(7, 'INTEGER', LEARNING_CATEGORY_CATEGORICAL, FEATURE_ROOT_TYPE_INT, 32)
FEATURE_TYPE_INT_8 = FeatureTypeInteger(8, 'INT_8', LEARNING_CATEGORY_CATEGORICAL, FEATURE_ROOT_TYPE_INT, 8)
FEATURE_TYPE_INT_16 = FeatureTypeInteger(9, 'INT_16', LEARNING_CATEGORY_CATEGORICAL, FEATURE_ROOT_TYPE_INT, 16)
FEATURE_TYPE_INT_32 = FEATURE_TYPE_INTEGER
FEATURE_TYPE_INT_64 = FeatureTypeInteger(10, 'INT_64', LEARNING_CATEGORY_CATEGORICAL, FEATURE_ROOT_TYPE_INT, 64)
FEATURE_TYPE_BOOL = FeatureTypeBool(11, 'INT_8', LEARNING_CATEGORY_BINARY, FEATURE_ROOT_TYPE_INT, 8)

ALL_FEATURE_TYPES: Dict[int, FeatureType] = {
    FEATURE_TYPE_STRING.key: FEATURE_TYPE_STRING,
    FEATURE_TYPE_CATEGORICAL.key: FEATURE_TYPE_CATEGORICAL,
    FEATURE_TYPE_FLOAT.key: FEATURE_TYPE_FLOAT,
    FEATURE_TYPE_FLOAT_32.key: FEATURE_TYPE_FLOAT_32,
    FEATURE_TYPE_FLOAT_64.key: FEATURE_TYPE_FLOAT_64,
    FEATURE_TYPE_DATE.key: FEATURE_TYPE_DATE,
    FEATURE_TYPE_DATE_TIME.key: FEATURE_TYPE_DATE_TIME,
    FEATURE_TYPE_INTEGER.key: FEATURE_TYPE_INTEGER,
    FEATURE_TYPE_INT_8.key: FEATURE_TYPE_INT_8,
    FEATURE_TYPE_INT_16.key: FEATURE_TYPE_INT_16,
    FEATURE_TYPE_INT_32.key: FEATURE_TYPE_INT_32,
    FEATURE_TYPE_INT_64.key: FEATURE_TYPE_INT_64,
    FEATURE_TYPE_BOOL.key: FEATURE_TYPE_BOOL
}


class FeatureTypeHelper:
    @classmethod
    def get_type(cls, key: int) -> FeatureType:
        try:
            return ALL_FEATURE_TYPES[key]
        except KeyError:
            raise FeatureDefinitionException(f'Could not find  FeatureType with key <{key}>')
