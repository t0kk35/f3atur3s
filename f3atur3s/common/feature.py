"""
Definition of the base dataframebuilder types. These are all helper or abstract classes
(c) 2023 tsm
"""
from dataclasses import dataclass, field, asdict
from typing import List, Type, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .typechecking import enforce_types
from .learningcategory import LearningCategory, LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_LABEL
from .learningcategory import LEARNING_CATEGORY_CONTINUOUS
from .featuretype import FeatureType, FeatureTypeInteger, FeatureTypeFloat, FeatureTypeNumerical, FeatureTypeString
from .featuretype import FeatureTypeBool
from .featuretype import FeatureTypeHelper
from .exception import FeatureDefinitionException, not_implemented


@enforce_types
@dataclass(unsafe_hash=True)
class Feature(ABC):
    """
    Base Feature class. All dataframebuilder will inherit from this class.
    It is an abstract-ish class that only defines the name and type
    """
    name: str
    type: FeatureType
    embedded_features: List['Feature'] = field(default_factory=list, init=False, hash=False, repr=False)

    def __dict__(self) -> Dict[str, Any]:
        json = asdict(self)
        # We don't need the full embedded dataframebuilder, just the names.
        json['embedded_features'] = [e['name'] for e in json['embedded_features']]
        # Don't need the learning category either, its derived
        del json['type']['learning_category']
        # But we do need to know the feature class
        json['class'] = self.__class__.__name__
        return json

    def _val_type(self, f_type: Type[FeatureType]) -> None:
        """
        Validation method to check if a feature is of a specific type. Will throw a FeatureDefinitionException
        if the feature is NOT of that type.

        Returns:
            None
        """
        if not isinstance(self.type, f_type):
            raise FeatureDefinitionException(
                f'The FeatureType of a {self.__class__.__name__} must be {f_type.__name__}. Got <{self.type.name}>'
            )

    def val_int_type(self) -> None:
        """
        Validation method to check if the feature is integer based. Will throw a FeatureDefinitionException
        if the feature is NOT integer based.

        Returns:
            None
        """
        self._val_type(FeatureTypeInteger)

    def val_float_type(self) -> None:
        """
        Validation method to check if the feature is float based. Will throw a FeatureDefinitionException
        if the feature is NOT float based.

        Returns:
             None
        """
        self._val_type(FeatureTypeFloat)

    def val_bool_type(self) -> None:
        """
        Validation method to check if the feature is bool based. Will throw a FeatureDefinitionException
        if the feature is NOT bool based.

        Return:
            None
        """
        self._val_type(FeatureTypeBool)

    def val_string_type(self) -> None:
        """
        Validation method to check if the feature is string based. Will throw a FeatureDefinitionException
        if the feature is NOT bool based.

        Return:
            None
        """
        self._val_type(FeatureTypeString)

    @property
    @abstractmethod
    def learning_category(self) -> LearningCategory:
        """
        Get the learning category of this feature. Will drive the sort of learning operations that are available on
        this feature. Learning categories are 'Continuous','Binary','Categorical' and 'Label

        Returns:
            The Learning Category. An instance of 'LearningCategory'
        """
        return not_implemented(self)

    @property
    @abstractmethod
    def inference_ready(self) -> bool:
        """
        Returns a bool indicating if the feature is ready for inference. Some dataframebuilder need to have been trained
        first or loaded. They need to know the inference attributes they will need to build the feature.

        Returns:
            True if the feature is ready for inference
        """
        return not_implemented(self)

    @classmethod
    @abstractmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List['Feature'], pkl: Any) -> 'Feature':
        pass

    @classmethod
    def extract_dict(cls, fields: Dict[str, Any], embedded_feature: List['Feature']) -> Tuple[Any, ...]:
        name = fields['name']
        tp = FeatureTypeHelper.get_type(fields['type']['key'])
        return name, tp


class FeatureCategorical(Feature, ABC):
    """
    Placeholder for dataframebuilder that are categorical in nature. They implement an additional __len__ method which
    will be used in embedding layers.
    """
    @abstractmethod
    def __len__(self):
        """Return the cardinality of the categorical feature.

        @return: Integer value, the cardinality of the categorical feature.
        """
        return not_implemented(self)

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_CATEGORICAL


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureWithBaseFeature(Feature, ABC):
    """
    Abstract class for dataframebuilder that have a base feature. These are typically dataframebuilder that are based off of another
    feature. There's a bunch of derived dataframebuilder that will have a base feature.
    """
    base_feature: Feature

    def __dict__(self) -> Dict[str, Any]:
        json = super().__dict__()
        # Just need to keep the name of the base_features
        json['base_feature'] = json['base_feature']['name']
        return json

    @classmethod
    def extract_dict(cls, fields: Dict[str, Any], embedded_features: List[Feature]) -> Tuple[Any, ...]:
        names = Feature.extract_dict(fields, embedded_features)
        fb = [f for f in embedded_features if f.name == fields['base_feature']][0]
        return names + (fb,)

    def val_base_feature_is_float(self) -> None:
        """
        Validation method to check if the base_feature is of type float. Will throw a FeatureDefinitionException if the
        base feature is NOT a float.

        Returns:
            None
        """
        if not isinstance(self.base_feature.type, FeatureTypeFloat):
            raise FeatureDefinitionException(
                f'Base feature of a {self.__class__.__name__} must be a float type. ' +
                f'Got <{type(self.base_feature.type)}>'
            )

    def val_base_feature_is_string(self):
        """
        Validation method to check if the base_feature is of type string. Will throw a FeatureDefinitionException
        if the base feature is NOT a string.

        Returns:
             None
        """
        if not isinstance(self.base_feature.type, FeatureTypeString):
            raise FeatureDefinitionException(
                f'The base feature parameter of a {self.__class__.__name__} must be a string-type ' +
                f'Got [{type(self.base_feature.type)}]'
            )

    def val_base_feature_is_string_or_integer(self):
        """
        Validation method to check if the base_feature is of type int or string. Will throw a FeatureDefinitionException
        if the base feature is NOT an int or string.

        Returns:
            None

        Raises:
            FeatureDefinitionException
        """
        if not isinstance(self.base_feature.type, FeatureTypeString) and \
                not isinstance(self.base_feature.type, FeatureTypeInteger):
            raise FeatureDefinitionException(
                f'The base feature parameter of a {self.__class__.__name__} must be a string-type or integer-type. ' +
                f'Got [{type(self.base_feature.type)}]'
            )

    def val_base_feature_is_numerical(self):
        """
        Validation method to check if the type of the base feature is of is numerical based. Will throw a
        FeatureDefinitionException if the type of the base feature is NOT numerical.

        Returns:
             None

        Raises:
            FeatureDefinitionException
        """
        if not isinstance(self.base_feature.type, FeatureTypeNumerical):
            raise FeatureDefinitionException(
                f'The base feature parameter of a {self.__class__.__name__} must be a numerical type. ' +
                f'Got [{type(self.base_feature.type)}]'
            )

    def get_base_and_base_embedded_features(self) -> List[Feature]:
        """
        Returns the base_feature + all dataframebuilder embedded in the base_feature.

        Returns:
            A list of dataframebuilder.
        """
        return list(set([self.base_feature] + self.base_feature.embedded_features))


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureExpander(FeatureWithBaseFeature, ABC):
    """
    Base class for expander dataframebuilder. Expander dataframebuilder expand when they are built. One feature in an input
    can turn into multiple dataframebuilder in output. For instance a one_hot encoded feature.
    """
    expand_names: List[str] = field(default=None, init=False, hash=False)

    @abstractmethod
    def expand(self) -> List[Feature]:
        """
        Expand the feature. This will return a list of dataframebuilder that will be built when the expander feature is
        generated

        Returns:
            A List of 'Feature' objects
        """
        return not_implemented(self)


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureLabel(FeatureWithBaseFeature, ABC):
    """
    Base class for all Features that will be used as labels during training
    """
    @property
    def inference_ready(self) -> bool:
        # A label feature has no inference attributes
        return True

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_LABEL


@enforce_types
@dataclass
class FeatureNormalize(FeatureWithBaseFeature, ABC):
    """
    Base class for dataframebuilder with normalizing logic
    """
    def __post_init__(self):
        self.val_float_type()
        self.val_base_feature_is_float()

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_CONTINUOUS


@enforce_types
@dataclass
class FeatureNormalizeLogBase(FeatureNormalize, ABC):
    log_base: Optional[str] = None
    delta: float = 1e-2

    @classmethod
    def extract_dict(cls, fields: Dict[str, Any], embedded_features: List[Feature]) -> Tuple[Any, ...]:
        names = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        lg = fields['log_base']
        dt = fields['delta']
        return names + (lg, dt)

    def log_base_valid(self):
        if self.log_base is not None and self.log_base not in self.valid_bases():
            raise FeatureDefinitionException(
                f'Error creating {self.name}. Requested log base {self.log_base}. ' +
                f'Supported bases are {self.valid_bases()}'
            )

    @staticmethod
    def valid_bases() -> List[str]:
        return ['e', '10', '2']


@enforce_types
@dataclass
class FeatureSeriesBased(Feature, ABC):
    pass
