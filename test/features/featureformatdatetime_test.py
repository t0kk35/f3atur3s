"""
Unit Tests for FeatureBin Creation
(c) 2023 tsm
"""
import shutil
import os
import unittest
import f3atur3s as ft


class TestFeatureDateTimeFormat(unittest.TestCase):
    def test_creation_base(self):
        name = 'DateFormat'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        fmt = ft.FeatureDateTimeFormat(name, ft.FEATURE_TYPE_INT_8, sf, '%w')
        self.assertIsInstance(fmt, ft.FeatureDateTimeFormat, f'Not expected type {type(fmt)}')
        self.assertEqual(fmt.name, name, f'Feature Name should be {name}')
        self.assertEqual(fmt.base_feature, sf, f'Base Feature not set correctly')
        self.assertEqual(len(fmt.embedded_features), 1, f'Should only have 1 emb feature {len(fmt.embedded_features)}')
        self.assertIn(sf, fmt.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(fmt.inference_ready, True, 'Should be inference ready upon creation')
        self.assertEqual(fmt.type, ft.FEATURE_TYPE_INT_8, 'Must always be int-8 type. Smallest possible')
        self.assertEqual(fmt.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Must have learning category cat')
        self.assertIsInstance(hash(fmt), int, f'Hash function not working')

    def test_creation_bad_not_date_time(self):
        name = 'DateFormat'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureDateTimeFormat(name, ft.FEATURE_TYPE_INT_8, sf, '%w')


class TestFeatureDateTimeFormatSaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-date-time-format-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_fmt = 'fmt-test'
        td_name = 'base'
        fmt = '%w'
        f_type = ft.FEATURE_TYPE_INT_16
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        f = ft.FeatureDateTimeFormat(n_fmt, f_type, fb, fmt)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_fmt}.json')), f'No {n_fmt}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_fmt}.json')), f'No {n_fmt}.json')
        # We Should have the base feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './load-date-time-format-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_fmt = 'fmt-test'
        td_name = 'base'
        fmt = '%w'
        f_type = ft.FEATURE_TYPE_INT_16
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        f = ft.FeatureDateTimeFormat(n_fmt, f_type, fb, fmt)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        td_new = ft.TensorDefinitionLoader.load(save_file)
        self.assertEqual(td_new.name, td.name, f'Names not equal {td_new.name} {td.name}')
        self.assertEqual(td_new.inference_ready, td.inference_ready, f'Inference state not equal')
        self.assertListEqual(td_new.learning_categories, td.learning_categories, f'Learning Cat not equal')
        self.assertEqual(td_new.features[0], td.features[0], 'Main Feature not the same')
        self.assertListEqual(td_new.embedded_features, td.embedded_features, f'Embedded features not the same')
        shutil.rmtree(save_file, ignore_errors=True)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
