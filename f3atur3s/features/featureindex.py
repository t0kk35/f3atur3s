"""
Definition of the Indexing feature. It is a feature directly found in a source.
(c) 2023 tsm
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List

from ..common.typechecking import enforce_types
from ..common.exception import FeatureRunTimeException
from ..common.feature import Feature, FeatureWithBaseFeature, FeatureCategorical


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureIndex(FeatureWithBaseFeature, FeatureCategorical):
    """
    Indexer feature. It will turn a specific input field (the base_feature) into an index. For instance 'DE'->1,
    'FR'->2, 'GB'->3 etc... The index will have an integer type and is ideal to model in embeddings.
    """
    dictionary: Dict[str, int] = field(default=None, init=False, hash=False)

    def __post_init__(self):
        self.val_int_type()
        self.val_base_feature_is_string_or_integer()
        # By default, return set embedded features to be the base feature.
        self.embedded_features.append(self.base_feature)
        self.embedded_features.extend(self.base_feature.embedded_features)

    def __len__(self):
        return len(self.dictionary)

    @property
    def inference_ready(self) -> bool:
        return self.dictionary is not None

    @property
    def index_to_label(self) -> Dict[int, Any]:
        if self.inference_ready:
            return {v: k for k, v in self.dictionary.items()}
        else:
            raise FeatureRunTimeException(
                f'Can not access the index_to_label property of feature {self.name} before is not ready for ' +
                f'inference. Please perform an inference run first.'
            )

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List[Feature], pkl: Any) -> 'FeatureIndex':
        name, tp, fb = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        fi = FeatureIndex(name, tp, fb)
        fi.dictionary = fields['dictionary']
        return fi
