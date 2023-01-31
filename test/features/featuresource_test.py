"""
Unit Tests for FeatureSource Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s.features as ft


class TestFeatureSource(unittest.TestCase):
    def test_creation_base(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_STRING
        f = ft.FeatureSource(name, f_type)
        self.assertIsInstance(f, ft.FeatureSource, f'Unexpected Type {type(f)}')
        self.assertEqual(f.name, name, f'Feature Name should be {name}')
        self.assertEqual(f.type, f_type, f'Feature Type should be {f_type}')
        self.assertIsNone(f.default, 'Should not have a default')
        self.assertIsNone(f.format_code, 'Should not have format code')
        self.assertEqual(len(f.embedded_features), 0, 'Should not have embedded features')
        self.assertEqual(f.learning_category, ft.LEARNING_CATEGORY_NONE, f'String should have learning type NONE')
        self.assertIsInstance(hash(f), int, f'Hash function not working')

    def test_creation_w_format_code(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_STRING
        code = 'anything'
        f = ft.FeatureSource(name, f_type, code)
        self.assertIsInstance(f, ft.FeatureSource, f'Unexpected Type {type(f)}')
        self.assertEqual(f.name, name, f'Feature Name should be {name}')
        self.assertEqual(f.type, f_type, f'Feature Type should be {f_type}')
        self.assertIsNone(f.default, 'Should not have a default')
        self.assertEqual(f.format_code, code, f'Format code should have been {code}')
        self.assertEqual(len(f.embedded_features), 0, 'Should not have embedded features')

    def test_creation_w_default(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_STRING
        default = 'NA'
        f = ft.FeatureSource(name, f_type, default=default)
        self.assertIsInstance(f, ft.FeatureSource, f'Unexpected Type {type(f)}')
        self.assertEqual(f.name, name, f'Feature Name should be {name}')
        self.assertEqual(f.type, f_type, f'Feature Type should be {f_type}')
        self.assertEqual(f.default, default, f'Default should be {default}')
        self.assertIsNone(f.format_code, 'Should not have format code')
        self.assertEqual(len(f.embedded_features), 0, 'Should not have embedded features')

    def test_create_source_time_without_format_code_bad(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_DATE_TIME
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureSource(name, f_type)

    def test_equality(self):
        name_1 = 'test_1'
        name_2 = 'test_2'
        f_type_1 = ft.FEATURE_TYPE_STRING
        f_type_2 = ft.FEATURE_TYPE_FLOAT
        f1 = ft.FeatureSource(name_1, f_type_1)
        f2 = ft.FeatureSource(name_1, f_type_1)
        f3 = ft.FeatureSource(name_2, f_type_1)
        f4 = ft.FeatureSource(name_1, f_type_2)
        self.assertEqual(f1, f2, f'Should have been equal')
        self.assertNotEqual(f1, f3, f'Should have been not equal')
        self.assertNotEqual(f1, f4, f'Should not have been equal. Different Type')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
