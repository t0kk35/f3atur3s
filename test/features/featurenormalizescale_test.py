"""
Unit Tests for FeatureNormalizeScale Creation
(c) 2023 tsm
"""
import os
import unittest
import shutil
import f3atur3s as ft


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


class TestFeatureNormalizeScaleSaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-norm-scale-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_s = 'scale-test'
        td_name = 'base'
        f_type = ft.FEATURE_TYPE_FLOAT
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_FLOAT)
        f = ft.FeatureNormalizeScale(n_s, f_type, fb)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_s}.json')), f'No {n_s}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_s}.json')), f'No {n_s}.json')
        # We Should have the base feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './save-norm-scale-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_s = 'scale-test'
        td_name = 'base'
        f_type = ft.FEATURE_TYPE_FLOAT
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_FLOAT)
        f = ft.FeatureNormalizeScale(n_s, f_type, fb)
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
