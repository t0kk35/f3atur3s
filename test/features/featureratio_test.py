"""
Unit Tests for FeatureSource Creation
(c) 2023 tsm
"""
import os
import unittest
import shutil
import f3atur3s as ft


class TestFeatureRatio(unittest.TestCase):
    def test_creation_base(self):
        name = 'Ratio'
        f_type = ft.FEATURE_TYPE_FLOAT
        sfn = ft.FeatureSource('numerator', ft.FEATURE_TYPE_FLOAT)
        sfd = ft.FeatureSource('denominator', ft.FEATURE_TYPE_FLOAT)
        rto = ft.FeatureRatio(name, f_type, sfn, sfd)
        self.assertIsInstance(rto, ft.FeatureRatio, f'Not expected type {type(rto)}')
        self.assertEqual(rto.name, name, f'Feature Name should be {name}')
        self.assertEqual(rto.base_feature, sfn, f'Base Feature not set correctly')
        self.assertEqual(rto.denominator_feature, sfd, f'Denominator Feature not set correctly')
        self.assertEqual(len(rto.embedded_features), 2, f'Should have 2 emb features {len(rto.embedded_features)}')
        self.assertIn(sfn, rto.embedded_features, 'Base Feature should be in emb feature list')
        self.assertIn(sfd, rto.embedded_features, 'Denominator Feature should be in emb feature list')
        self.assertEqual(rto.inference_ready, True, 'Should always be inference ready')
        self.assertEqual(rto.type, f_type, 'Must always be float type.')
        self.assertEqual(rto.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'Must have learning cat Categorical')
        self.assertIsInstance(hash(rto), int, f'Hash function not working')

    def test_type_non_numerical_is_bad(self):
        name = 'Ratio'
        f_type = ft.FEATURE_TYPE_STRING
        sfn = ft.FeatureSource('numerator', ft.FEATURE_TYPE_FLOAT)
        sfd = ft.FeatureSource('denominator', ft.FEATURE_TYPE_FLOAT)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureRatio(name, f_type, sfn, sfd)

    def test_base_non_numerical_is_bad(self):
        name = 'Ratio'
        f_type = ft.FEATURE_TYPE_FLOAT
        sfn = ft.FeatureSource('numerator', ft.FEATURE_TYPE_STRING)
        sfd = ft.FeatureSource('denominator', ft.FEATURE_TYPE_FLOAT)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureRatio(name, f_type, sfn, sfd)

    def test_base_int_is_also_good(self):
        name = 'Ratio'
        f_type = ft.FEATURE_TYPE_FLOAT
        sfn = ft.FeatureSource('numerator', ft.FEATURE_TYPE_INT_16)
        sfd = ft.FeatureSource('denominator', ft.FEATURE_TYPE_FLOAT)
        rto = ft.FeatureRatio(name, f_type, sfn, sfd)
        self.assertEqual(rto.name, name, f'Feature Name should be {name}')

    def test_denominator_non_numerical_is_bad(self):
        name = 'Ratio'
        f_type = ft.FEATURE_TYPE_FLOAT
        sfn = ft.FeatureSource('numerator', ft.FEATURE_TYPE_FLOAT)
        sfd = ft.FeatureSource('denominator', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureRatio(name, f_type, sfn, sfd)

    def test_denominator_int_is_also_good(self):
        name = 'Ratio'
        f_type = ft.FEATURE_TYPE_FLOAT
        sfn = ft.FeatureSource('numerator', ft.FEATURE_TYPE_FLOAT)
        sfd = ft.FeatureSource('denominator', ft.FEATURE_TYPE_INT_16)
        rto = ft.FeatureRatio(name, f_type, sfn, sfd)
        self.assertEqual(rto.name, name, f'Feature Name should be {name}')

    def test_equality(self):
        s_name_1 = 's_test_1'
        s_name_2 = 's_test_2'
        f_type_1 = ft.FEATURE_TYPE_FLOAT
        f_type_2 = ft.FEATURE_TYPE_FLOAT_32
        b_f_1 = ft.FeatureSource('numerator1', ft.FEATURE_TYPE_FLOAT)
        b_f_2 = ft.FeatureSource('numerator2', ft.FEATURE_TYPE_FLOAT)
        n_f_1 = ft.FeatureSource('denominator1', ft.FEATURE_TYPE_FLOAT)
        n_f_2 = ft.FeatureSource('denominator2', ft.FEATURE_TYPE_FLOAT)
        rf_1 = ft.FeatureRatio(s_name_1, f_type_1, b_f_1, n_f_1)
        rf_2 = ft.FeatureRatio(s_name_2, f_type_1, b_f_1, n_f_1)
        rf_3 = ft.FeatureRatio(s_name_1, f_type_2, b_f_1, n_f_1)
        rf_4 = ft.FeatureRatio(s_name_1, f_type_1, b_f_2, n_f_1)
        rf_5 = ft.FeatureRatio(s_name_1, f_type_1, b_f_1, n_f_2)
        self.assertEqual(rf_1, rf_1, f'Same feature should have been equal')
        self.assertNotEqual(rf_1, rf_2, f'Should not have been equal. Different Name')
        self.assertNotEqual(rf_1, rf_3, f'Should not have been equal. Different Type')
        self.assertNotEqual(rf_1, rf_4, f'Should not have been equal. Different Base-Feature')
        self.assertNotEqual(rf_1, rf_5, f'Should not have been equal. Different Denominator-Feature')


class TestFeatureRatioSaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-ratio-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_r = 'ratio-test'
        n_d = 'denominator-test'
        td_name = 'base'
        f_type = ft.FEATURE_TYPE_FLOAT
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_FLOAT)
        fd = ft.FeatureSource(n_d, ft.FEATURE_TYPE_FLOAT)
        f = ft.FeatureRatio(n_r, f_type, fb, fd)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_r}.json')), f'No {n_r}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_r}.json')), f'No {n_r}.json')
        # We Should have the base feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './load-ratio-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_r = 'ratio-test'
        n_d = 'denominator-test'
        td_name = 'base'
        f_type = ft.FEATURE_TYPE_FLOAT
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_FLOAT)
        fd = ft.FeatureSource(n_d, ft.FEATURE_TYPE_FLOAT)
        f = ft.FeatureRatio(n_r, f_type, fb, fd)
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
