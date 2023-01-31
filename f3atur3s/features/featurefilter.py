"""
Definition of the concatenating feature. It is a feature directly that concatenates 2 string features.
(c) 2023 tsm
"""
from dataclasses import dataclass

from common.typechecking import enforce_types
from .featureexpression import FeatureExpression


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureFilter(FeatureExpression):
    """
    Is a specialisation of the FeatureExpression. It can only output a true or false, so must have a boolean type
    and the function must return a 'bool' type. It can be used to filter certain rows.
    """
    def __post_init__(self):
        # Run post init validation.
        self.val_bool_type()
        self._val_parameters_is_features_list(self.param_features)
        self._val_expression_not_lambda()
        self._val_function_is_callable(self.expression, self.param_features)
        # The parameters needed to run the function are the embedded features
        self.embedded_features.extend(self.param_features)
        for pf in self.param_features:
            self.embedded_features.extend(pf.embedded_features)
        self.embedded_features = list(set(self.embedded_features))
