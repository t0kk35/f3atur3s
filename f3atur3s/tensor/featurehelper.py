"""
class methods seem to get lost in data classes. This class bundles some class methods which the original FeatureType
# and Feature classes had, but were no longer visible after changing them to a dataclass.
# (c) 2023 tsm
"""
from typing import Type, List, TypeVar

from ..common.feature import Feature, FeatureType

T = TypeVar('T')


class FeatureHelper:
    @classmethod
    def filter_feature(cls, feature_class: Type[T], features: List[Feature]) -> List[T]:
        """
        Class method to filter a list of features. The method will return the features from the input
        that match the current class

        Args:
            feature_class: (Type) A feature class to filter.
            features: (List of Feature) A list of features.

        Returns:
            A list of features. Contains the input features that are of the class of feature_class.
        """
        return [f for f in features if isinstance(f, feature_class)]

    @classmethod
    def filter_not_feature(cls, feature_class: Type[Feature], features: List[T]) -> List[T]:
        """
        Class method to filter a list of features. The method will return the features from the input
        that do NOT match the current class

        Args:
            feature_class (Type) A feature class to filter.
            features: A list of features.

        Returns:
             A list of features. Contains the input features that are not of the class feature_class
        """
        return [f for f in features if not isinstance(f, feature_class)]

    @classmethod
    def filter_feature_type(cls, feature_type: Type[FeatureType], features: List[T]) -> List[T]:
        """
        Class method to filter a list of features. The method will return the features from the input
        that with type that match the requested feature_type

        Args:
            feature_type: (Type) A type of feature to check.
            features: (List of Feature) A list of features.

        Returns:
             A list of features. Contains the input features that are of the type of the current class
        """
        return [f for f in features if isinstance(f.type, feature_type)]
