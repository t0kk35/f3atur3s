"""
Unit Tests for FeatureNormalizeScale Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s.features as ft


class TestNormalizeScaleFeature(unittest.TestCase):
    def test_creation_base(self):
        name = 'scale'
        f_type = ft.FEATURE_TYPE_FLOAT
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        scf = ft.FeatureNormalizeScale(name, f_type, sf)
        self.assertIsInstance(scf, ft.FeatureNormalizeScale, f'Incorrect Type {type(scf)}')
        self.assertEqual(scf.name, name, f'Scale feature name incorrect {name}')
        self.assertEqual(scf.type, f_type, f'Scale feature type should have been {f_type}')
        self.assertIsNone(scf.log_base, f'Unless specified, log_base should be None {scf.log_base}')
        self.assertEqual(scf.inference_ready, False, f'Scale feature should NOT be inference ready')
        self.assertIsNone(scf.minimum, f'Scale minimum should be None')
        self.assertIsNone(scf.maximum, f'Scale maximum should be None')
        self.assertEqual(scf.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Wrong Learning category')
        self.assertIsInstance(hash(scf), int, f'Hash function not working')

    def test_creation_non_float(self):
        name = 'scale'
        f_type_str = ft.FEATURE_TYPE_STRING
        f_type_flt = ft.FEATURE_TYPE_FLOAT
        sf_flt = ft.FeatureSource('Source', f_type_flt)
        sf_str = ft.FeatureSource('Source', f_type_str)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureNormalizeScale(name, f_type_str, sf_flt)
        with self.assertRaises(ft.FeatureDefinitionException):
            ft.FeatureNormalizeScale(name, f_type_flt, sf_str)

    def test_creation_with_log_base(self):
        name = 'scale'
        f_type = ft.FEATURE_TYPE_FLOAT
        log_base = 'e'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        scf = ft.FeatureNormalizeScale(name, f_type, sf, log_base)
        self.assertEqual(scf.log_base, log_base, f'log_base not correctly set')

    def test_creation_invalid_log_base(self):
        name = 'scale'
        f_type = ft.FEATURE_TYPE_FLOAT
        log_base = '5'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureNormalizeScale(name, f_type, sf, log_base)

    def test_creation_w_delta(self):
        name = 'scale'
        f_type = ft.FEATURE_TYPE_FLOAT
        delta = 1e-2
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        scf = ft.FeatureNormalizeScale(name, f_type, sf, delta=delta)
        self.assertEqual(scf.delta, delta, f'Delta not set correctly')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
