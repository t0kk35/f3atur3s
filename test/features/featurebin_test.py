"""
Unit Tests for FeatureBin Creation
(c) 2023 tsm
"""
import shutil
import os
import unittest
import f3atur3s as ft


class TestFeatureBin(unittest.TestCase):
    def test_creation_base(self):
        name = 'Bin'
        nr_bin = 10
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        bn = ft.FeatureBin(name, f_type, sf, nr_bin)
        self.assertIsInstance(bn, ft.FeatureBin, f'Not expected type {type(bn)}')
        self.assertEqual(bn.name, name, f'Feature Name should be {name}')
        self.assertEqual(bn.base_feature, sf, f'Base Feature not set correctly')
        self.assertEqual(len(bn.embedded_features), 1, f'Should only have 1 emb feature {len(bn.embedded_features)}')
        self.assertIn(sf, bn.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(bn.inference_ready, False, 'Should be not inference ready upon creation')
        self.assertEqual(bn.type, f_type, 'Must always be int-8 type. Smallest possible')
        self.assertEqual(bn.learning_category, ft.LEARNING_CATEGORY_CATEGORICAL, f'Must have learning cat Categorical')
        self.assertEqual(bn.number_of_bins, nr_bin, f'Number of bins is wrong. Got {bn.number_of_bins}')
        self.assertEqual(len(bn), nr_bin, f'Length of feature should have been nr_bins. Should have. Got {len(bn)}')
        self.assertIsNone(bn.bins, f'Bins should not only be set after inference. Got {bn.bins}')
        self.assertEqual(bn.scale_type, 'linear', f'Default ScaleType should have been Linear')
        self.assertEqual(bn.range, list(range(1, nr_bin)), f'Unexpected Range. Got {bn.range}')
        self.assertIsInstance(hash(bn), int, f'Hash function not working')
        with self.assertRaises(ft.FeatureRunTimeException):
            _ = bn.index_to_label

    def test_creation_bad_base(self):
        name = 'Bin'
        nr_bin = 10
        f_type = ft.FEATURE_TYPE_INT_16
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureBin(name, f_type, sf, nr_bin)

    def test_creation_bad_type(self):
        name = 'Bin'
        nr_bin = 10
        f_type = ft.FEATURE_TYPE_FLOAT
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_FLOAT)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureBin(name, f_type, sf, nr_bin)

    def test_equality(self):
        s_name_1 = 's_test_1'
        s_name_2 = 's_test_2'
        b_name_1 = 'b_test_1'
        b_name_2 = 'b_test_2'
        f_type_1 = ft.FEATURE_TYPE_FLOAT
        f_type_2 = ft.FEATURE_TYPE_INT_8
        f_type_3 = ft.FEATURE_TYPE_INT_16
        fs1 = ft.FeatureSource(s_name_1, f_type_1)
        fs2 = ft.FeatureSource(s_name_2, f_type_1)
        fb1 = ft.FeatureBin(b_name_1, f_type_2, fs1, 10)
        fb2 = ft.FeatureBin(b_name_1, f_type_2, fs1, 10)
        fb3 = ft.FeatureBin(b_name_2, f_type_2, fs1, 10)
        fb4 = ft.FeatureBin(b_name_1, f_type_2, fs2, 10)
        fb5 = ft.FeatureBin(b_name_1, f_type_3, fs1, 10)
        self.assertEqual(fb1, fb2, f'Should have been equal')
        self.assertNotEqual(fb1, fb3, f'Should have been not equal')
        self.assertNotEqual(fb1, fb4, f'Should not have been equal. Different Base Feature')
        self.assertNotEqual(fb1, fb5, f'Should not have been equal. Different Type')


class TestFeatureBinSaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-bin-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_bin = 'bin-test'
        td_name = 'base'
        bins = 3
        f_type = ft.FEATURE_TYPE_INT_16
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_FLOAT)
        f = ft.FeatureBin(n_bin, f_type, fb, bins)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_bin}.json')), f'No {n_bin}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_bin}.json')), f'No {n_bin}.json')
        # We Should have the base feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './load-bin-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_bin = 'bin-test'
        td_name = 'base'
        bins = 3
        f_type = ft.FEATURE_TYPE_INT_16
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_FLOAT)
        f = ft.FeatureBin(n_bin, f_type, fb, bins)
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
