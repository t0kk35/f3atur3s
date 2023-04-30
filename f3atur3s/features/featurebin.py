"""
Definition of the binning feature. It is a feature directly found in a source.
(c) 2023 tsm
"""
from dataclasses import dataclass, field
from typing import List, Any, Dict

from ..common.typechecking import enforce_types
from ..common.exception import FeatureRunTimeException
from ..common.feature import Feature, FeatureWithBaseFeature, FeatureCategorical


@enforce_types
@dataclass(unsafe_hash=True)
class FeatureBin(FeatureWithBaseFeature, FeatureCategorical):
    """
    Feature that will 'bin' a float number. Binning means the float feature will be turned into an int/categorical
    variable. For instance values 0.0 till 0.85 will be bin 1, from 0.85 till 1.7 bin 2 etc
    """
    number_of_bins: int
    scale_type: str = 'linear'
    bins: List[int] = field(default=None, init=False, hash=False)

    def __post_init__(self):
        self.val_int_type()
        self.val_base_feature_is_float()
        # By default; return set embedded features to be the base feature.
        self.embedded_features = self.get_base_and_base_embedded_features()

    def __len__(self):
        # Need to add one, we will also have an unknown/0 value.
        return self.number_of_bins

    @property
    def index_to_label(self) -> Dict[int, Any]:
        if self.inference_ready:
            d = {0: 'UNK'}
            d.update({i+1: round(self.bins[i], 3) for i in range(0, self.number_of_bins-1)})
            return d
        else:
            raise FeatureRunTimeException(f'Can not access the index_to_label property of feature {self.name}' +
                                          f' it is not ready for inference. Please perform an inference run first')

    @property
    def range(self) -> List[int]:
        return list(range(1, len(self)))

    @property
    def inference_ready(self) -> bool:
        return self.bins is not None

    @classmethod
    def create_from_save(cls, fields: Dict[str, Any], embedded_features: List[Feature], pkl: Any) -> 'FeatureBin':
        name, tp, fb = FeatureWithBaseFeature.extract_dict(fields, embedded_features)
        nb = fields['number_of_bins']
        st = fields['scale_type']
        return FeatureBin(name, tp, fb, nb, st)
