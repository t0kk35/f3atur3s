"""
Unit Tests for FeatureLabelBinary Creation
(c) 2023 tsm
"""
import unittest
import shutil
import os
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


class TestFeatureLabelBinarySaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-label-bin-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_l = 'label-test'
        td_name = 'base-label-bin'
        f_type = ft.FEATURE_TYPE_INT_16
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_INT_16)
        f = ft.FeatureLabelBinary(n_l, f_type, fb)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_l}.json')), f'No {n_l}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_l}.json')), f'No {n_l}.json')
        # We Should have the base feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './load-label-bin-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_l = 'label-bin-test'
        td_name = 'base'
        f_type = ft.FEATURE_TYPE_INT_16
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_INT_16)
        f = ft.FeatureIndex(n_l, f_type, fb)
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
