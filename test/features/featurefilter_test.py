"""
Unit Tests for FeatureFilter Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s as ft


def feature_expression(param: int):
    return param == 1


class TestFeatureFilter(unittest.TestCase):
    def test_creation_base(self):
        name = 'filter'
        f_type = ft.FEATURE_TYPE_BOOL
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        ff = ft.FeatureFilter(name, f_type, feature_expression, par)
        self.assertIsInstance(ff, ft.FeatureFilter, f'Not expected type {type(ff)}')
        self.assertEqual(ff.name, name, f'Feature Name should be {name}')
        self.assertEqual(ff.type, f_type, f'Feature Type incorrect. Got {ff.type}')
        self.assertEqual(len(ff.embedded_features), len(par), f'Should have had {len(par)} embedded features')
        self.assertEqual(ff.embedded_features[0], par[0], f'Embedded Features should have been the parameters')
        self.assertEqual(ff.expression, feature_expression, f'Should have gotten the expression. Got {ff.expression}')
        self.assertEqual(ff.param_features, par, f'Did not get the parameters {ff.param_features}')
        self.assertIsInstance(hash(ff), int, f'Hash function not working')

    def test_bad_non_bool_type(self):
        name = 'filter'
        f_type = ft.FEATURE_TYPE_INT_8
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_INT_16)
        par = [sf]
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureFilter(name, f_type, feature_expression, par)

    # TODO Need equality tests


def main():
    unittest.main()


if __name__ == '__main__':
    main()
