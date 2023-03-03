"""
Definition of the concatenating feature. It is a feature directly that concatenates 2 string dataframebuilder.
(c) 2023 tsm
"""
from dataclasses import dataclass, field
from inspect import signature, isfunction
from typing import Callable, List, Dict, Any
from abc import ABC

from ..common.typechecking import enforce_types
from ..common.feature import Feature, LearningCategory
from ..common.feature import FeatureDefinitionException
from ..common.featuresave import FeatureWithPickle


@dataclass(unsafe_hash=True)
class _ExpressionBased(Feature, ABC):
    """
    Base class for dataframebuilder that use and expression
    """
    expression: Callable = field(hash=False)
    param_features: List[Feature] = field(default_factory=list, hash=False)

    @staticmethod
    def _val_parameters_is_features_list(param_features: List[Feature]):
        if not isinstance(param_features, List):
            raise FeatureDefinitionException(
                f'Param_features argument to <{_ExpressionBased.__name__}> must be a list. ' +
                f'Got <{type(param_features)}>'
            )
        if not len(param_features) == 0 and not isinstance(param_features[0], Feature):
            raise FeatureDefinitionException(
                f'The elements in the param_feature list to <{_ExpressionBased.__name__}> must be Feature Objects. ' +
                f'Got <{type(param_features[0])}>'
            )

    @staticmethod
    def _val_function_is_callable(expression: Callable, param_features: List[Feature]):
        if not isfunction(expression):
            raise FeatureDefinitionException(f' Expression parameter must be function')

        expression_signature = signature(expression)
        if len(expression_signature.parameters) != len(param_features):
            raise FeatureDefinitionException(
                f'Number of arguments of function and dataframebuilder do not match '
                f'[{len(expression_signature.parameters)}]  [{len(param_features)}]'
            )

    def _val_expression_not_lambda(self):
        if self.is_lambda:
            raise FeatureDefinitionException(
                f'The expression for series expression feature <{self.name}> can not be a lambda expression. ' +
                f'Lambdas are not serializable during multi-processing. ' +
                f'The "expression" parameter for series expressions must be a function'
            )

    @property
    def inference_ready(self) -> bool:
        # An expression feature has no inference attributes
        return True

    @property
    def is_lambda(self) -> bool:
        """Flag indicating if the expression that is used to build the feature is a Lambda style callable

        Return:
             True if the expression is a Lambda.
        """
        return self.expression.__name__ == '<lambda>'

    @property
    def learning_category(self) -> LearningCategory:
        # Should be the Learning category of the type of the Expression Feature.
        return self.type.learning_category


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureExpression(_ExpressionBased, FeatureWithPickle):
    """
    Derived Feature. This is a Feature that will be built off of other dataframebuilder using a function. It can be used
    to perform all sorts of custom operations on other dataframebuilder, such as adding, formatting, calculating ratio's etc...

    The function passed to as expression must be available in the main Python Context
    """
    def __post_init__(self):
        # Run post init validation
        self._val_parameters_is_features_list(self.param_features)
        self._val_function_is_callable(self.expression, self.param_features)
        # The parameters needed to run the function are the embedded dataframebuilder
        self.embedded_features.extend(self.param_features)
        for pf in self.param_features:
            self.embedded_features.extend(pf.embedded_features)
        self.embedded_features = list(set(self.embedded_features))

    def __dict__(self) -> Dict[str, Any]:
        json = super().__dict__()
        # We only need the names of the parameter dataframebuilder
        json['param_features'] = [f['name'] for f in json['param_features']]
        # Need to remove the expression, that will be pickled
        del json['expression']
        return json

    def get_pickle(self) -> Any:
        return self.expression

    @classmethod
    def create_from_save(
            cls, fields: Dict[str, Any], embedded_features: List['Feature'], pkl: Any) -> 'FeatureExpression':
        name, tp = Feature.extract_dict(fields, embedded_features)
        param = [eb for eb in embedded_features for f in fields['param_features'] if eb.name == f]
        return FeatureExpression(name, tp, pkl, param)


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureExpressionSeries(_ExpressionBased, FeatureWithPickle):
    def __post_init__(self):
        # Run post init validation.
        self._val_expression_not_lambda()
        self._val_parameters_is_features_list(self.param_features)
        self._val_function_is_callable(self.expression, self.param_features)
        # The parameters needed to run the function are the embedded dataframebuilder
        self.embedded_features.extend(self.param_features)
        for pf in self.param_features:
            self.embedded_features.extend(pf.embedded_features)
        self.embedded_features = list(set(self.embedded_features))

    def __dict__(self) -> Dict[str, Any]:
        json = super().__dict__()
        # We only need the names of the parameter dataframebuilder
        json['param_features'] = [f['name'] for f in json['param_features']]
        # Need to remove the expression, that will be pickled
        del json['expression']
        return json

    def get_pickle(self) -> Any:
        return self.expression

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List['Feature'], pkl: Any) -> 'Feature':
        name, tp = Feature.extract_dict(fields, embedded_features)
        param = [eb for eb in embedded_features for f in fields['param_features'] if eb.name == f]
        return FeatureExpressionSeries(name, tp, pkl, param)
