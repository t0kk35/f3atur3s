"""
Definition of the save types. These are needed to save certain classes
(c) 2023 tsm
"""
from abc import ABC, abstractmethod
from typing import Any


class FeatureWithPickle(ABC):
    @abstractmethod
    def name(self)-> str:
        pass

    @abstractmethod
    def get_pickle(self) -> Any:
        pass
