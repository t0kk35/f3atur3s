"""
Unit Tests for FeatureBin Creation
(c) 2023 tsm
"""
import unittest
import f3atur3s as ft
import shutil

SAVE_LOCATION = './data/save/'


class TestLoaderSaver(unittest.TestCase):
    def test_base_save_load(self):
        # Save a TensorDefinition
        location = SAVE_LOCATION + 'base_case'
        # Remove directory if exists
        shutil.rmtree(location, ignore_errors=True)
        f1 = ft.FeatureSource('f1', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureSource('f2', ft.FEATURE_TYPE_CATEGORICAL)
        f3 = ft.FeatureSource('f3', ft.FEATURE_TYPE_DATE, format_code='%Y%m%d')
        f4 = ft.FeatureOneHot('f4', ft.FEATURE_TYPE_INT_16, f2)
        f_list = [f1, f2, f3, f4]
        td = ft.TensorDefinition('test-td', f_list)
        ft.TensorDefinitionSaver.save(td, location)

        # Reload it
        td2 = ft.TensorDefinitionLoader.load(location)
        self.assertEqual(td.name, td2.name, f'Names not equal {td.name}, {td2.name}')
        self.assertEqual(td.inference_ready, td2.inference_ready, f'Inference mode not the same')
        self.assertListEqual(td.features, td2.features, f'Features not the same')
        self.assertListEqual(td.learning_categories, td2.learning_categories, f'Learning Categories not the same')
        # We don't care if the order is not the same.
        self.assertListEqual(
            sorted(td.embedded_features, key=lambda x: x.name),
            sorted(td2.embedded_features, key=lambda x: x.name), f'Embedded Features not the same')
        # As we did not do an inference run we can not get the rank, nor the shapes of the TensorDefinition
        with self.assertRaises(ft.TensorDefinitionException):
            _ = td.rank
        with self.assertRaises(ft.TensorDefinitionException):
            _ = td2.rank
        with self.assertRaises(ft.TensorDefinitionException):
            _ = td.shapes
        with self.assertRaises(ft.TensorDefinitionException):
            _ = td2.shapes


def main():
    unittest.main()


if __name__ == '__main__':
    main()
