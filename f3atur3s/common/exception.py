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

class FeatureRunTimeException(Exception):
    """
    Exception thrown when there is a run-time problem
    """
    def __init__(self, message: str):
        super().__init__("Error Using Feature: " + message)


class TensorDefinitionException(Exception):
    """ Exception thrown when the creation of a TensorDefinition fails

    Args:
        message: A free form text message describing the error
    """
    def __init__(self, message: str):
        super().__init__("Error Defining Tensor: " + message)


class TensorDefinitionSaverException(Exception):
    """
    Exception thrown when the saving of a TensorDefinition fails.

    Args:
        message: A free form text message describing the error
    """
    def __init__(self, tensor_name: str, message: str):
        super().__init__(f"Error Saving Tensor: {tensor_name} Message: {message}")


class TensorDefinitionLoaderException(Exception):
    """
    Exception thrown when the saving of a TensorDefinition fails.

    Args:
        message: A free form text message describing the error
    """
    def __init__(self, message: str):
        super().__init__(f"Error Loading Tensor. Message: {message}")
