"""
Unit Tests for FeatureVirtual Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s.features as ft


class TestFeatureVirtual(unittest.TestCase):
    def test_creation_name_type(self):
        name = 'Virtual'
        f_type = ft.FEATURE_TYPE_STRING
        vf = ft.FeatureVirtual(name=name, type=f_type)
        self.assertIsInstance(vf, ft.FeatureVirtual, f'Not expected type {type(vf)}')
        self.assertEqual(vf.name, name, f'Name should have been {name}')
        self.assertEqual(vf.type, f_type, f'Type Should have been {f_type}')
        self.assertEqual(len(vf.embedded_features), 0, f'Virtual feature should not have embedded features')
        self.assertIsInstance(hash(vf), int, f'Hash function not working')

    def test_equality(self):
        name_1 = 'test_1'
        name_2 = 'test_2'
        f_type_1 = ft.FEATURE_TYPE_STRING
        f_type_2 = ft.FEATURE_TYPE_FLOAT
        v1 = ft.FeatureVirtual(name_1, f_type_1)
        v2 = ft.FeatureVirtual(name_1, f_type_1)
        v3 = ft.FeatureVirtual(name_2, f_type_1)
        v4 = ft.FeatureVirtual(name_1, f_type_2)
        self.assertEqual(v1, v2, f'Should have been equal')
        self.assertNotEqual(v1, v3, f'Should have been not equal')
        self.assertNotEqual(v1, v4, f'Should not have been equal. Different Type')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
