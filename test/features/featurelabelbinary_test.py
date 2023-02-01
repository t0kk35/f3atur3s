"""
Unit Tests for FeatureLabelBinary Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s as ft


class TestFeatureLabelBinary(unittest.TestCase):
    def test_create_base(self):
        name = 'label'
        f_type = ft.FEATURE_TYPE_INT_16
        fs = ft.FeatureSource('source', f_type)
        fl = ft.FeatureLabelBinary(name, ft.FEATURE_TYPE_INT_8, fs)
        self.assertIsInstance(fl, ft.FeatureLabelBinary, f'Unexpected Type {type(fl)}')
        self.assertEqual(fl.name, name, f'Feature Name should be {name}')
        self.assertEqual(fl.type, ft.FEATURE_TYPE_INT_8, f'Feature Type should be int8 {fl.type}')
        self.assertEqual(fl.base_feature, fs, f'Base Feature Should have been the source feature')
        self.assertEqual(len(fl.embedded_features), 1, f'Should only have 1 emb feature {len(fl.embedded_features)}')
        self.assertIn(fs, fl.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(fl.learning_category, ft.LEARNING_CATEGORY_LABEL, f'Learning type should be LABEL')
        self.assertIsInstance(hash(fl), int, f'Hash function not working')

    def test_creation_non_numerical_type(self):
        name = 'index'
        f_type = ft.FEATURE_TYPE_STRING
        fs = ft.FeatureSource('source', f_type)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureLabelBinary(name, f_type, fs)

    def test_equality(self):
        s_name_1 = 's_test_1'
        s_name_2 = 's_test_2'
        l_name_1 = 'l_test_1'
        l_name_2 = 'l_test_2'
        f_type_1 = ft.FEATURE_TYPE_INT_16
        f_type_2 = ft.FEATURE_TYPE_INT_8
        fs1 = ft.FeatureSource(s_name_1, f_type_1)
        fs2 = ft.FeatureSource(s_name_2, f_type_2)
        fl1 = ft.FeatureLabelBinary(l_name_1, ft.FEATURE_TYPE_INT_8, fs1)
        fl2 = ft.FeatureLabelBinary(l_name_1, ft.FEATURE_TYPE_INT_8, fs1)
        fl3 = ft.FeatureLabelBinary(l_name_2, ft.FEATURE_TYPE_INT_8, fs1)
        fl4 = ft.FeatureLabelBinary(l_name_1, ft.FEATURE_TYPE_INT_8, fs2)
        fl5 = ft.FeatureLabelBinary(l_name_1, ft.FEATURE_TYPE_INT_8, fs1)
        self.assertEqual(fl1, fl2, f'Should have been equal')
        self.assertNotEqual(fl1, fl3, f'Should have been not equal')
        self.assertNotEqual(fl1, fl4, f'Should have been equal')
        self.assertEqual(fl1, fl5, f'Should have been equal')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
