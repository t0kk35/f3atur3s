"""
Definition of the save types. These are needed to save certain classes
(c) 2023 tsm
"""
from abc import ABC, abstractmethod
from typing import Any

from .feature import Feature


class FeatureWithPickle(Feature, ABC):
    @abstractmethod
    def get_pickle(self) -> Any:
        pass
