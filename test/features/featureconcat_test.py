"""
Unit Tests for FeatureConcat Creation
(c) 2023 tsm
"""
import os
import unittest
import shutil
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
        sfb = ft.FeatureSource('base', ft.FEATURE_TYPE_STRING)
        sfc = ft.FeatureSource('concat', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureConcat(name, f_type, sfb, sfc)

    def test_base_non_string_is_bad(self):
        name = 'Concat'
        f_type = ft.FEATURE_TYPE_STRING
        sfb = ft.FeatureSource('base', ft.FEATURE_TYPE_FLOAT)
        sfc = ft.FeatureSource('concat', ft.FEATURE_TYPE_STRING)
        with self.assertRaises(ft.FeatureDefinitionException):
            _ = ft.FeatureConcat(name, f_type, sfb, sfc)

    def test_concat_non_string_is_bad(self):
        name = 'Concat'
        f_type = ft.FEATURE_TYPE_STRING
        sfb = ft.FeatureSource('base', ft.FEATURE_TYPE_STRING)
        sfc = ft.FeatureSource('concat', ft.FEATURE_TYPE_FLOAT)
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


class TestFeatureConcatSaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-concat-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_base2 = 'base2-test'
        n_c = 'concat-test'
        td_name = 'base'
        f_type = ft.FEATURE_TYPE_STRING
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource(n_base2, ft.FEATURE_TYPE_STRING)
        f = ft.FeatureConcat(n_c, f_type, fb, fc)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_c}.json')), f'No {n_c}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_c}.json')), f'No {n_c}.json')
        # We Should have the base feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{n_base}.json')), f'No {n_base}.json')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './load-concat-base'
        shutil.rmtree(save_file, ignore_errors=True)
        n_base = 'base-test'
        n_base2 = 'base2-test'
        n_c = 'concat-test'
        td_name = 'base'
        f_type = ft.FEATURE_TYPE_STRING
        fb = ft.FeatureSource(n_base, ft.FEATURE_TYPE_STRING)
        fc = ft.FeatureSource(n_base2, ft.FEATURE_TYPE_STRING)
        f = ft.FeatureConcat(n_c, f_type, fb, fc)
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
