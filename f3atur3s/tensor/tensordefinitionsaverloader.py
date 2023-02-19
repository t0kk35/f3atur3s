"""
Helper Class for the saving and loading TensorDefinition objects
(c) 2023 tsm
"""
import importlib
import json
import os
import json as js
from pathlib import Path
from typing import List, Dict

from ..common.feature import Feature
from ..common.exception import TensorDefinitionSaverException, TensorDefinitionLoaderException
from ..common.exception import TensorDefinitionException
from .tensordefinition import TensorDefinition

FEATURE_DIR = 'features'
TENSOR_JSON_FILE = 'tensor.json'


class TensorDefinitionSaver:
    """
    Helper class for the saving of a TensorDefinition into JSON files.
    """
    @classmethod
    def save(cls, td: TensorDefinition, directory: str):
        # Check if the path exists, make if it does not exist.
        if os.path.exists(directory):
            raise TensorDefinitionSaverException(td.name, f'Path already exists {directory}')
        else:
            os.makedirs(directory)

        cls._write_tensor_json(td, directory)
        cls._write_features_jsons(td, directory)

    @staticmethod
    def _write_tensor_json(td: TensorDefinition, directory: str):
        # Save General TensorDefinition data
        try:
            rank = td.rank
        except TensorDefinitionException:
            rank = None

        td_dict = {
            'name': td.name,
            'rank': rank,
            'features': [f.name for f in td.features]
        }

        with open(os.path.join(directory, TENSOR_JSON_FILE), 'w') as j_file:
            json.dump(td_dict, j_file, indent=4)

    @staticmethod
    def _write_features_jsons(td: TensorDefinition, directory: str):
        # Save all the features
        os.makedirs(os.path.join(directory, FEATURE_DIR))
        to_save_features = td.embedded_features
        for f in to_save_features:
            with open(os.path.join(directory, FEATURE_DIR, f'{f.name}.json'), 'w') as f_file:
                json.dump(f.__dict__(), f_file, indent=4)


class TensorDefinitionLoader:
    """
    Helper class for Loading a TensorDefinition from JSON files.
    """
    @classmethod
    def load(cls, directory: str) -> TensorDefinition:
        # Check if file exists
        if not os.path.exists(directory):
            raise TensorDefinitionLoaderException(f'Can not load find directory {directory}')

        # Check we have a tensor file
        if not os.path.exists(os.path.join(directory, TENSOR_JSON_FILE)):
            raise TensorDefinitionLoaderException(f'Can not find file {TENSOR_JSON_FILE} in directory {directory}')

        with open(os.path.join(directory, TENSOR_JSON_FILE)) as j_file:
            td_dict = json.load(j_file)

        # Check we have a features directory
        if not os.path.exists(os.path.join(directory, FEATURE_DIR)):
            raise TensorDefinitionLoaderException(f'Can not find {FEATURE_DIR} in directory {directory}')

        # Read all the json feature files. This will create all feature, native and embedded.
        all_features = TensorDefinitionLoader._read_features_jsons(directory)
        # Create TensorDefinition with native features only
        td = TensorDefinition(td_dict['name'], [f for f in all_features if f.name in td_dict['features']])
        return td

    @staticmethod
    def _read_features_jsons(directory: str) -> List[Feature]:
        # Check we have a features directory
        if not os.path.exists(os.path.join(directory, FEATURE_DIR)):
            raise TensorDefinitionLoaderException(f'Can not find {FEATURE_DIR} in directory {directory}')

        # Create all features list
        to_read_features: List[Dict] = []
        read_features: List[Dict] = []
        built_features: List[Feature] = []

        # Read all files with *.json
        files = Path(os.path.join(directory, FEATURE_DIR)).glob('*.json')
        for file in files:
            with open(file) as j_file:
                to_read_features.append(json.load(j_file))

        i = 0
        while len(to_read_features) > 0:
            if i > 20:
                raise TensorDefinitionLoaderException(
                    f'Exiting. Did more that {i} iterations trying to load features from {directory}'+
                    f'Potential endless loop.'
                )
            read_names = [f['name'] for f in read_features]
            ready_to_read = [f for f in to_read_features if all(ef in read_names for ef in f['embedded_features'])]
            for f in ready_to_read:
                built_features.append(TensorDefinitionLoader._build_feature(f, built_features))
            read_features.extend(ready_to_read)
            to_read_features = [f for f in to_read_features if f not in ready_to_read]
            i = i+1

        return built_features

    @staticmethod
    def _build_feature(f_dict: Dict, built_features: List[Feature]) -> Feature:
        # Create a class instance. This will create an instance of Feature.
        f_class = getattr(importlib.import_module("f3atur3s"), f_dict['class'])
        if not issubclass(f_class, Feature):
            raise TensorDefinitionLoaderException(f'{f_class.__name__} is not an instance of Feature. Can not load')
        if not hasattr(f_class, 'from_dict') or not callable(getattr(f_class, 'from_dict')):
            raise TensorDefinitionLoaderException(f'{f_class.__name__} does not have a <from_dict> class method')

        func = getattr(f_class, 'from_dict')
        emb = [f for f in built_features if f.name in f_dict['embedded_features']]
        return func(f_dict, emb)
