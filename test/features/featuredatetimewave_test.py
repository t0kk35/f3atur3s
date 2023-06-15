"""
Unit Tests for FeatureBin Creation
(c) 2023 tsm
"""
import shutil
import os
import unittest
import f3atur3s as ft


class TestFeatureDateTimeWave(unittest.TestCase):
    def test_creation_base(self):
        name = 'WeekWave'
        ty = ft.FEATURE_TYPE_FLOAT
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        pr = 7
        fr = 2
        fw = ft.FeatureDateTimeWave(name, ty, sf, '%w', pr, fr)
        self.assertIsInstance(fw, ft.FeatureDateTimeWave, f'Not expected type {type(fw)}')
        self.assertEqual(fw.name, name, f'Feature Name should be {name}')
        self.assertEqual(fw.period, pr, f'Period not set')
        self.assertEqual(fw.frequencies, fr, f'Frequencies not set')
        self.assertEqual(fw.base_feature, sf, f'Base Feature not set correctly')
        self.assertEqual(len(fw.expand_names), 2 * fr, f'Expected {2*fr} frequencies')
        self.assertListEqual(fw.expand_names,
                             [f'{fw.base_feature.name}{fw.delimiter}{w}{fw.delimiter}{f}'
                              for f in range(fw.frequencies) for w in ('sin', 'cos')])
        self.assertEqual(len(fw.embedded_features), 1, f'Should only have 1 emb feature {len(fw.embedded_features)}')
        self.assertIn(sf, fw.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(fw.inference_ready, True, 'Should be inference ready upon creation')
        self.assertEqual(fw.type, ty, 'Incorrect type. Expecting {ty}')
        self.assertEqual(fw.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Must be learning continuous cat')
        self.assertIsInstance(hash(fw), int, f'Hash function not working')

    def test_creation_bad_not_float(self):
        name = 'DateFormat'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureDateTimeWave(name, ft.FEATURE_TYPE_INT_8, sf, '%w', 2, 2)

    def test_creation_bad_base_not_date_time(self):
        name = 'DateFormat'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureDateTimeWave(name, ft.FEATURE_TYPE_FLOAT, sf, '%w', 2, 2)

    def test_creation_bad_per_not_int(self):
        name = 'DateFormat'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        with self.assertRaises(TypeError):
            _ = ft.FeatureDateTimeWave(name, ft.FEATURE_TYPE_FLOAT, sf, '%w', 2.0, 2)

    def test_creation_bad_freq_not_int(self):
        name = 'DateFormat'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        with self.assertRaises(TypeError):
            _ = ft.FeatureDateTimeWave(name, ft.FEATURE_TYPE_FLOAT, sf, '%w', 2, 2.0)


class TestFeatureDateTimeWaveSaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-date-time-wave-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_fw = 'fw-test'
        td_name = 'base'
        fmt = '%w'
        period = 2
        frequency = 3
        f_type = ft.FEATURE_TYPE_FLOAT
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        f = ft.FeatureDateTimeWave(n_fw, f_type, fb, fmt, period, frequency)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_fw}.json')), f'No {n_fw}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_fw}.json')), f'No {n_fw}.json')
        # We Should have the base feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './load-date-time-wave-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_fw = 'wave-test'
        td_name = 'base'
        fmt = '%w'
        period = 2
        frequency = 3
        f_type = ft.FEATURE_TYPE_FLOAT
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_DATE_TIME, '%Y%M%d')
        f = ft.FeatureDateTimeWave(n_fw, f_type, fb, fmt, period, frequency)
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
