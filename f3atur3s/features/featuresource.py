"""
Definition of the source feature. It is a feature directly found in a source.
(c) 2023 tsm
"""
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, List

from ..common.typechecking import enforce_types
from ..common.exception import FeatureDefinitionException
from ..common.featuretype import FeatureTypeHelper
from ..common.feature import Feature
from ..common.featuretype import FeatureTypeTimeBased
from ..common.learningcategory import LearningCategory


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureSource(Feature):
    """
    A feature found in a source. I.e a file or message or JSON or other. This is the most basic feature.
    """
    format_code: Optional[str] = None
    default: Optional[Union[str, float, int]] = None

    def __post_init__(self):
        self._val_format_code_not_none_for_time()

    @property
    def inference_ready(self) -> bool:
        # A source feature has no inference attributes
        return True

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the learning category of the type of the source feature
        return self.type.learning_category

    @classmethod
    def from_dict(cls, fields: Dict[str, Any], embedded_features: List[Feature]) -> 'FeatureSource':
        name, tp = cls.extract_dict(fields, embedded_features)
        fc = fields['format_code']
        df = fields['default']
        return FeatureSource(name, tp, fc, df)

    def _val_format_code_not_none_for_time(self):
        """
        Validation Method to check that there is a format specified for TimeBased type Features.

        Returns:
            None
        """
        if isinstance(self.type, FeatureTypeTimeBased):
            if self.format_code is None:
                raise FeatureDefinitionException(
                    f'Feature {self.name} is time based, its format_code should not be <None>'
                )
