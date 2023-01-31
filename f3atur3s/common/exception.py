"""
Definition of the Feature Definition Exception
(c) 2023 tsm
"""


def not_implemented(class_):
    raise NotImplementedError(f'Feature problem. Not defined for class {class_.__class__.name}')


class FeatureDefinitionException(Exception):
    """
    Exception thrown when the Definition of a feature fails
    """
    def __init__(self, message: str):
        super().__init__("Error Defining Feature: " + message)


class TensorDefinitionException(Exception):
    """ Exception thrown when the creation of a TensorDefinition fails

    Args:
        message: A free form text message describing the error
    """
    def __init__(self, message: str):
        super().__init__("Error Defining Tensor: " + message)
