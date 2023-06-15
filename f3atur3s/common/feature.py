"""
Definition of the base features types. These are all helper or abstract classes
(c) 2023 tsm
"""
from dataclasses import dataclass, field, asdict
from typing import List, Type, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .typechecking import enforce_types
from .learningcategory import LearningCategory, LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_LABEL
from .learningcategory import LEARNING_CATEGORY_CONTINUOUS, LEARNING_CATEGORY_NONE
from .featuretype import FeatureType, FeatureTypeInteger, FeatureTypeFloat, FeatureTypeString
from .featuretype import FeatureTypeBool, FeatureTypeNumerical, FeatureTypeTimeBased
from .featuretype import FeatureTypeHelper
from .exception import FeatureDefinitionException, not_implemented


@enforce_types
@dataclass(unsafe_hash=True)
class Feature(ABC):
    """
    Base Feature class. All features will inherit from this class.
    It is an abstract-ish class that only defines the name and type
    """
    name: str
    type: FeatureType
    embedded_features: List['Feature'] = field(default_factory=list, init=False, hash=False, repr=False)

    def __dict__(self) -> Dict[str, Any]:
        json = asdict(self)
        # We don't need the full embedded features, just the names.
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
        Returns a bool indicating if the feature is ready for inference. Some features need to have been trained
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
    Placeholder for features that are categorical in nature. They implement an additional __len__ method which
    will be used in embedding layers.
    """
    @abstractmethod
    def __len__(self):
        """
        Return the cardinality of the categorical feature.

        Returns:
            Integer value, the cardinality of the categorical feature.
        """
        return not_implemented(self)

    @property
    @abstractmethod
    def index_to_label(self) -> Dict[int, Any]:
        """
        Return a dictionary that maps an index to a label (string). This will be used in various visualizing features.
        For instance in bar charts and scatter plots of Embeddings.

        Returns:
            A Dictionary with the index (int) as key and the label (str) as value
        """
        return not_implemented(self)

    @property
    def learning_category(self) -> LearningCategory:
        return LEARNING_CATEGORY_CATEGORICAL


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureWithBaseFeature(Feature, ABC):
    """
    Abstract class for features that have a base feature. These are typically features that are based off of another
    feature. There's a bunch of derived features that will have a base feature.
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

    def val_base_feature_is_time_based(self):
        """
        Validation method to check if the type of the base feature is of is time based. Will throw a
        FeatureDefinitionException if the type of the base feature is NOT time based.

        Returns:
             None

        Raises:
            FeatureDefinitionException
        """
        if not isinstance(self.base_feature.type, FeatureTypeTimeBased):
            raise FeatureDefinitionException(
                f'The base feature parameter of a {self.__class__.__name__} must be a time based type. ' +
                f'Got [{type(self.base_feature.type)}]'
            )

    def get_base_and_base_embedded_features(self) -> List[Feature]:
        """
        Returns the base_feature + all feature embedded in the base_feature.

        Returns:
            A list of features.
        """
        return list(set([self.base_feature] + self.base_feature.embedded_features))


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureExpander(FeatureWithBaseFeature, ABC):
    """
    Base class for expander features. Expander features expand when they are built. One feature in an input
    can turn into multiple features in output. For instance a one_hot encoded feature.
    """
    expand_names: List[str] = field(default=None, init=False, hash=False)

    @abstractmethod
    def expand(self) -> List[Feature]:
        """
        Expand the feature. This will return a list of features that will be built when the expander feature is
        generated

        Returns:
            A List of 'Feature' objects
        """
        return not_implemented(self)

    @property
    @abstractmethod
    def delimiter(self) -> str:
        """
        Property that returns the string to be used to delimit names. An expander feature will create multiple new
        features. In order to make the names unique it will assign names that are [ORIGINAL_NAME] + delimiter +
        [UNIQUE_STRING]

        Returns:
            A string object, which is used to delimit the expanded names
        """
        pass


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
    Base class for features with normalizing logic
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
    series_features: List[Feature] = field(hash=False)

    def val_same_learning_type(self):
        """
        Validation method to check if all features in a list have the same LearningCategory

        Return:
            None

        Raises:
            FeatureDefinitionException
        """
        lc = list(set([f.learning_category for f in self.series_features]))
        if len(lc) > 1:
            raise FeatureDefinitionException(
                f'Found more than one LearningCategory in the list of features {lc}. This function works with' +
                f'one LearningCategory only.'
            )

    def val_learning_category_not_none(self):
        """
        Validation method to check that there is no feature that has the LEARNING_CATEGORY_NONE

        Return:
            None

        Raises:
            FeatureDefinitionException
        """
        lc = list(set([f.learning_category for f in self.series_features]))
        if LEARNING_CATEGORY_NONE in lc:
            raise FeatureDefinitionException(
                f'Found (a) Feature(s) ' +
                f' {[f.name for f in self.series_features if f.learning_category == LEARNING_CATEGORY_NONE]} ' +
                f'with LEARNING_CATEGORY_NONE. This function can not process those'
            )

    def val_same_root_feature_type(self):
        """
        Validation method to check that the root type of the features is the same. Sometimes we will want to
        all have floats or all ints.

        Return:
            None

        Raises:
            FeatureDefinitionException
        """
        rt = list(set([f.type.root_type for f in self.series_features]))
        if len(rt) > 1:
            raise FeatureDefinitionException(
                f'Found more than one Root Feature Type. {rt}. This procedure can only use similar feature root types'
            )

    def val_all_series_features_numerical(self):
        """
        Validation method to check that the root type of the features is the same. Sometimes we will want to
        all have floats or all ints.

        Return:
            None

        Raises:
            FeatureDefinitionException
        """
        if not all([isinstance(f.type, FeatureTypeNumerical) for f in self.series_features]):
            raise FeatureDefinitionException(
                f'All Series Features need to be numerical for feature {self.name} ' +
                f'Found non-numerical features + '
                f'{[f.name for f in self.series_features if not isinstance(f, FeatureTypeNumerical)]}'
            )
