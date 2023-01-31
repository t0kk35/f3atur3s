"""
Unit Tests for FeatureConcat Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s as ft


class TestFeatureConcat(unittest.TestCase):
    def test_creation_base(self):
        name = 'Concat'
        f_type = ft.FEATURE_TYPE_STRING
        sfb = ft.FeatureSource('base', ft.FEATURE_TYPE_STRING)
        sfc = ft.FeatureSource('concat', ft.FEATURE_TYPE_STRING)
        cat = ft.FeatureConcat(name, f_type, sfb, sfc)
        self.assertIsInstance(cat, ft.FeatureConcat, f'Not expected type {type(cat)}')
        self.assertEqual(cat.name, name, f'Feature Name should be {name}')
        self.assertEqual(cat.base_feature, sfb, f'Base Feature not set correctly')
        self.assertEqual(cat.concat_feature, sfc, f'concat Feature not set correctly')
        self.assertEqual(len(cat.embedded_features), 2, f'Should have 2 emb features {len(cat.embedded_features)}')
        self.assertIn(sfb, cat.embedded_features, 'Base Feature should be in emb feature list')
        self.assertIn(sfc, cat.embedded_features, 'Concat Feature should be in emb feature list')
        self.assertEqual(cat.inference_ready, True, 'Should always be inference ready')
        self.assertEqual(cat.type, f_type, 'Must always be string type.')
        self.assertEqual(cat.learning_category, ft.LEARNING_CATEGORY_NONE, f'Must have learning cat None')
        self.assertIsInstance(hash(cat), int, f'Hash function not working')

    def test_type_non_string_is_bad(self):
        name = 'Concat'
        f_type = ft.FEATURE_TYPE_FLOAT
        sfb = ft.FeatureSource('base', ft.FEATURE_TYPE_FLOAT)
        sfc = ft.FeatureSource('concat', ft.FEATURE_TYPE_FLOAT)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureConcat(name, f_type, sfb, sfc)

    def test_base_non_string_is_bad(self):
        name = 'Concat'
        f_type = ft.FEATURE_TYPE_FLOAT
        sfb = ft.FeatureSource('base', ft.FEATURE_TYPE_FLOAT)
        sfc = ft.FeatureSource('concat', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureConcat(name, f_type, sfb, sfc)

    def test_denominator_non_numerical_is_bad(self):
        name = 'Concat'
        f_type = ft.FEATURE_TYPE_FLOAT
        sfb = ft.FeatureSource('base', ft.FEATURE_TYPE_FLOAT)
        sfc = ft.FeatureSource('concat', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureConcat(name, f_type, sfb, sfc)

    def test_equality(self):
        s_name_1 = 's_test_1'
        s_name_2 = 's_test_2'
        f_type_1 = ft.FEATURE_TYPE_STRING
        b_f_1 = ft.FeatureSource('numerator1', ft.FEATURE_TYPE_STRING)
        b_f_2 = ft.FeatureSource('numerator2', ft.FEATURE_TYPE_STRING)
        c_f_1 = ft.FeatureSource('denominator1', ft.FEATURE_TYPE_STRING)
        c_f_2 = ft.FeatureSource('denominator2', ft.FEATURE_TYPE_STRING)
        rf_1 = ft.FeatureConcat(s_name_1, f_type_1, b_f_1, c_f_1)
        rf_2 = ft.FeatureConcat(s_name_2, f_type_1, b_f_1, c_f_1)
        rf_3 = ft.FeatureConcat(s_name_1, f_type_1, b_f_2, c_f_1)
        rf_4 = ft.FeatureConcat(s_name_1, f_type_1, b_f_1, c_f_2)
        self.assertEqual(rf_1, rf_1, f'Same feature should have been equal')
        self.assertNotEqual(rf_1, rf_2, f'Should not have been equal. Different Name')
        self.assertNotEqual(rf_1, rf_3, f'Should not have been equal. Different Base-Feature')
        self.assertNotEqual(rf_1, rf_4, f'Should not have been equal. Different Concat-Feature')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
