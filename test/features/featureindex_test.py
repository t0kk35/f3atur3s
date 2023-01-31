"""
Unit Tests for FeatureIndex Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s.features as ft


class TestIndexFeature(unittest.TestCase):
    def test_creation_base(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_INT_16
        fs = ft.FeatureSource('source', f_type)
        fi = ft.FeatureIndex(name, ft.FEATURE_TYPE_INT_16, fs)
        self.assertIsInstance(fi, ft.FeatureIndex, f'Unexpected Type {type(fi)}')
        self.assertEqual(fi.name, name, f'Feature Name should be {name}')
        self.assertEqual(fi.type, f_type, f'Feature Type should be {f_type}')
        self.assertEqual(fi.base_feature, fs, f'Base Feature Should have been the source feature')
        self.assertEqual(len(fi.embedded_features), 1, f'Should only have 1 emb feature {len(fi.embedded_features)}')
        self.assertIn(fs, fi.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(fi.inference_ready, False, 'Should be not inference ready upon creation')
        self.assertEqual(fi.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Learning type should be CATEGORICAL')
        self.assertIsNone(fi.dictionary, f'Dictionary should be None')
        self.assertIsInstance(hash(fi), int, f'Hash function not working')

    def test_creation_non_int_type(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_BOOL
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureIndex(name, ft.FEATURE_TYPE_FLOAT_32, fs)

    def test_creation_base_bool(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_BOOL
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureIndex(name, ft.FEATURE_TYPE_INT_16, fs)

    def test_creation_base_float(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_FLOAT_32
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureIndex(name, ft.FEATURE_TYPE_INT_16, fs)

    def test_equality(self):
        s_name_1 = 's_test_1'
        s_name_2 = 's_test_2'
        i_name_1 = 'i_test_1'
        i_name_2 = 'i_test_2'
        f_type_1 = ft.FEATURE_TYPE_INT_16
        f_type_2 = ft.FEATURE_TYPE_INT_8
        fs1 = ft.FeatureSource(s_name_1, f_type_1)
        fs2 = ft.FeatureSource(s_name_2, f_type_1)
        fi1 = ft.FeatureIndex(i_name_1, f_type_1, fs1)
        fi2 = ft.FeatureIndex(i_name_1, f_type_1, fs1)
        fi3 = ft.FeatureIndex(i_name_2, f_type_1, fs1)
        fi4 = ft.FeatureIndex(i_name_1, f_type_1, fs2)
        fi5 = ft.FeatureIndex(i_name_1, f_type_2, fs1)
        self.assertEqual(fi1, fi2, f'Should have been equal')
        self.assertNotEqual(fi1, fi3, f'Should have been not equal')
        self.assertNotEqual(fi1, fi4, f'Should not have been equal. Different Base Feature')
        self.assertNotEqual(fi1, fi5, f'Should not have been equal. Different Type')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
