"""
Definition of the TensorDefinition feature. It is used to bundle Feature objects
(c) 2023 tsm
"""

from typing import List, Tuple

from ..common.exception import TensorDefinitionException
from ..common.feature import Feature, FeatureExpander, FeatureTypeNumerical
from ..common.learningcategory import LearningCategory, LEARNING_CATEGORY_CATEGORICAL, LEARNING_CATEGORY_CONTINUOUS
from ..common.learningcategory import LEARNING_CATEGORY_BINARY, LEARNING_CATEGORY_LABEL, LEARNING_CATEGORIES_MODEL
from ..features.featureindex import FeatureIndex
from ..features.featureonehot import FeatureOneHot
from .featurehelper import FeatureHelper


class TensorDefinition:
    """ A TensorDefinition is a container of dataframebuilder. A set of dataframebuilder can be bundled in a tensor definition. That
    tensor definition can then be constructed by the engines and used in modelling.

    Args:
        name: A name for this tensor definition
        features: A list of dataframebuilder to group in this feature definition
    """
    def _val_rank_set(self):
        if self._rank is None:
            raise TensorDefinitionException(
                f'The Rank of Tensor Definition <{self.name}> has not been set. Can not retrieve it. Maybe ' +
                f'it needs to be used in a run with no "Inference"'
            )

    def _val_shapes_set(self):
        if self._shapes is None:
            raise TensorDefinitionException(
                f'The Shape of Tensor Definition <{self.name}> has not been set. Can not retrieve it. Maybe ' +
                f'it needs to be used in a run with no "Inference"'
            )

    def _val_shapes_match_lcs(self, shapes: List[Tuple[int, ...]]):
        if len(shapes) != len(self.learning_categories):
            raise TensorDefinitionException(
                f'The number of shapes <{len(shapes)}> is not the same as the number of learning categories for ' +
                f'this TensorDefinition <{self.learning_categories}>'
            )

    def _val_duplicate_entries(self):
        names = [f.name for f in self.features]
        if len(list(set(names))) != len(names):
            raise TensorDefinitionException(
                f'Tensor definition has duplicate entries ' +
                f' <{[n for n in names if names.count(n) > 1]}>'
            )

    def _val_not_empty(self):
        if len(self.features) == 0:
            raise TensorDefinitionException(
                f'Tensor definition <{self.name} has no dataframebuilder. It is empty. Can not perform action'
            )

    def _val_has_numerical_features(self):
        if len(FeatureHelper.filter_feature_type(FeatureTypeNumerical, self.features)) == 0:
            raise TensorDefinitionException(
                f'Tensor definition <{self.name} has no numerical dataframebuilder. Can not perform action'
            )

    def _val_base_feature_overlap(self):
        fi = set([f.base_feature for f in FeatureHelper.filter_feature(FeatureIndex, self.features)])
        fo = set([f.base_feature for f in FeatureHelper.filter_feature(FeatureOneHot, self.features)])
        s = fi.intersection(fo)
        if len(s) != 0:
            raise TensorDefinitionException(
                f'FeatureIndex and FeatureOneHot should not have the same base dataframebuilder. Overlap <{s}>'
            )

    def _val_inference_ready(self, operation: str):
        if not self.inference_ready:
            raise TensorDefinitionException(
                f'Tensor is not ready for inference. Can not perform operation <{operation}>'
            )

    def __init__(self, name: str, features: List[Feature] = None):
        self._name = name
        self._rank = None
        self._shapes = None
        if features is None:
            self._feature_list = []
        else:
            self._features_list = features
        self._val_duplicate_entries()
        self._val_base_feature_overlap()

    def __len__(self):
        return len(self.features)

    def __repr__(self):
        return f'Tensor Definition : {self.name}'

    @property
    def name(self):
        """
        Name of the Tensor Definition

        Returns:
             String representation of the name
        """
        return self._name

    @property
    def rank(self) -> int:
        """
        Returns the rank of this TensorDefinition. The Rank is only known after the Tensor Definition has been used
        to actually build a tensor.

        Returns:
             Rank of the Tensor as int
        """
        self._val_rank_set()
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        """
        Rank property setter

        Args:
            rank: The rank of this Tensor Definition

        Returns:
             None
        """
        self._rank = rank

    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        """
        Returns the 'expected' shape of this TensorDefinition. The Shape is only known after the Tensor Definition
        has been used to build a Numpy List.
        Because a Tensor definition can not know the batch size, the first dimension is *hard-coded to -1*.

        Returns:
             A List of int Tuples. There will be a tuple per each learning category. The tuples will contain an int
             for each dimension, each int is the size along that dimension.
        """
        self._val_shapes_set()
        return self._shapes

    @shapes.setter
    def shapes(self, shapes: List[Tuple[int, ...]]):
        """
        Shape property setter.

        Args:
             shapes: A Tuple of ints describing the length along the various dimensions.

        Returns:
            None
        """
        self._val_shapes_match_lcs(shapes)
        self._shapes = shapes

    @property
    def features(self) -> List[Feature]:
        """
        Property that lists all dataframebuilder of this tensor definition

        Returns:
             A list of dataframebuilder of the tensor definition
        """
        return self._features_list

    @property
    def feature_names(self) -> List[str]:
        """
        Helper function that returns the names of the feature in the TensorDefinition. If the feature is a
        FeatureExpander, the expanded names of the Feature are returned.

        Returns:
            The list names of the dataframebuilder included in this TensorDefinition
        """
        out = []
        for f in self.features:
            if isinstance(f, FeatureExpander):
                out.extend(f.expand_names)
            else:
                out.append(f.name)
        return out

    @property
    def embedded_features(self) -> List[Feature]:
        """
        Function which returns all dataframebuilder embedded in the base dataframebuilder + the base dataframebuilder themselves. It effectively
        returns all dataframebuilder referenced in this Tensor Definition.

        Returns:
             A list of dataframebuilder embedded in the base dataframebuilder + the base dataframebuilder
        """
        base_features = self._features_list
        embedded_features = [features.embedded_features for features in base_features]
        embedded_features_flat = [feature for features in embedded_features for feature in features]
        return list(set(embedded_features_flat + base_features))

    @property
    def inference_ready(self) -> bool:
        """
        Method that return True if the Tensor is ready for inference. It means  all embedded dataframebuilder are ready fpr
        inference, they either have no inference attributes or their inference attributes are set.

        Returns:
             Bool. True or False Indicating if the tensor is ready or not for inference.
        """
        return all([f.inference_ready for f in self.embedded_features])

    @property
    def learning_categories(self) -> List[LearningCategory]:
        res = [lc for lc in LEARNING_CATEGORIES_MODEL if len(self.filter_features(lc)) > 0]
        return res

    @staticmethod
    def _expand_features(features: List[Feature]) -> List[Feature]:
        r = []
        for f in features:
            if isinstance(f, FeatureExpander):
                r.extend(f.expand())
            else:
                r.append(f)
        return r

    @property
    def highest_precision_feature(self) -> Feature:
        """
        Return the highest precision (numerical) feature in this Tensor Definition.

        Returns:
             The feature with the highest precision
        """
        self._val_has_numerical_features()
        t = FeatureHelper.filter_feature_type(FeatureTypeNumerical, self.features)
        t.sort(key=lambda x: x.type.precision)
        # Last one has the biggest precision
        return t[-1]

    def remove(self, feature: Feature) -> None:
        self._features_list.remove(feature)

    def filter_features(self, category: LearningCategory, expand=False) -> List[Feature]:
        """
        Filter dataframebuilder in this Tensor Definition according to a Learning category.
        NOTE that with expand 'True', the Tensor Definition must be ready for inference.

        Args:
            category: The LearningCategory to filter out.
            expand: Bool value. True or False indicating if the names of expander dataframebuilder will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False. NOTE that with expand 'True', the Tensor Definition must be ready for inference.

        Returns:
             List of dataframebuilder of the specified 'LearningCategory'
        """
        if expand:
            self._val_inference_ready('filter ' + category.name)

        r = [f for f in self.features if f.learning_category == category]
        if expand:
            r = TensorDefinition._expand_features(r)
        return r

    def categorical_features(self, expand=False) -> List[Feature]:
        """
        Return the categorical dataframebuilder in this Tensor Definition.

        Args:
            expand: Bool value. True or False indicating if the names of expander dataframebuilder will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False

        Returns:
             List of categorical dataframebuilder in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_CATEGORICAL, expand)

    def binary_features(self, expand=False) -> List[Feature]:
        """
        Return the binary dataframebuilder in this Tensor Definition

        Args:
            expand: Bool value. True or False indicating if the names of expander dataframebuilder will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False

        Returns:
             List of binary dataframebuilder in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_BINARY, expand)

    def continuous_features(self, expand=False) -> List[Feature]:
        """
        Return the continuous feature in this Tensor Definition

        Args:
             expand: Bool value. True or False indicating if the names of expander dataframebuilder will be expanded. For in
             instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
             Default is False

        Returns:
            List of continuous dataframebuilder in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_CONTINUOUS, expand)

    def label_features(self, expand=False) -> List[Feature]:
        """
        Return the label feature in this Tensor Definition

        Args:
            expand: Bool value. True or False indicating if the names of expander dataframebuilder will be expanded. For in
            instance with expand = True a 'FeatureOneHot' country will be expanded to country__DE, country__FR etc...
            Default is False

        Returns:
             List of label dataframebuilder in this Tensor Definition
        """
        return self.filter_features(LEARNING_CATEGORY_LABEL, expand)

    def features_not_inference_ready(self) -> List[Feature]:
        """
        List dataframebuilder of this TensorDefinition which are not ready for inference

        Returns:
             A list of dataframebuilder that returned False to the inference_ready call
        """
        return [f for f in self.embedded_features if f.inference_ready]
