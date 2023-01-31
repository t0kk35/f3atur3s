"""
Unit Tests for FeatureExpression Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s as ft

from typing import Any


def feature_expression(param: int):
    return param == 1


class TestFeatureExpression(unittest.TestCase):
    def test_creation_base(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        ef = ft.FeatureExpression(name, f_type, feature_expression, par)
        self.assertIsInstance(ef, ft.FeatureExpression, f'Not expected type {type(ef)}')
        self.assertEqual(ef.name, name, f'Feature Name should be {name}')
        self.assertEqual(ef.type, f_type, f'Feature Type incorrect. Got {ef.type}')
        self.assertEqual(len(ef.embedded_features), len(par), f'Should have had {len(par)} embedded features')
        self.assertEqual(ef.embedded_features[0], par[0], f'Embedded Features should have been the parameters')
        self.assertEqual(ef.expression, feature_expression, f'Should have gotten the expression. Got {ef.expression}')
        self.assertEqual(ef.param_features, par, f'Did not get the parameters {ef.param_features}')
        self.assertIsInstance(hash(ef), int, f'Hash function not working')

    def test_creation_base_lambda(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        ef = ft.FeatureExpression(name, f_type, lambda x: x + 1, par)
        self.assertEqual(ef.is_lambda, True, f'Should been lambda')

    def test_learning_features(self):
        name = 'expr'
        f_type_1 = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        ef = ft.FeatureExpression(name, f_type_1, feature_expression, par)
        self.assertEqual(ef.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL)
        f_type_2 = ft.FEATURE_TYPE_FLOAT
        ef = ft.FeatureExpression(name, f_type_2, feature_expression, par)
        self.assertEqual(ef.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS)
        f_type_3 = ft.FEATURE_TYPE_BOOL
        ef = ft.FeatureExpression(name, f_type_3, feature_expression, par)
        self.assertEqual(ef.learning_category, ft.LEARNING_CATEGORY_BINARY)
        f_type_4 = ft.FEATURE_TYPE_STRING
        ef = ft.FeatureExpression(name, f_type_4, feature_expression, par)
        self.assertEqual(ef.learning_category, ft.LEARNING_CATEGORY_NONE)

    def test_creation_bad_not_expression(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        bad: Any = 'bad'
        # Not an expression
        with self.assertRaises(TypeError):
            _ = ft.FeatureExpression(name, f_type, bad, par)

    def test_creation_bad_param(self):
        name = 'expr'
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par_1: Any = ''
        # Param not a list
        with self.assertRaises(TypeError):
            _ = ft.FeatureExpression(name, f_type, feature_expression, par_1)
        par_2: Any = ['']
        # Not list with Feature objects
        with self.assertRaises(TypeError):
            _ = ft.FeatureExpression(name, f_type, feature_expression, par_2)
        par_3 = [sf, sf]
        # Incorrect Number of Parameters
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureExpression(name, f_type, feature_expression, par_3)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
