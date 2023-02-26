"""
Unit Tests for FeatureVirtual Creation
(c) 2023 tsm
"""
import os
import unittest
import shutil
import f3atur3s as ft


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


class TestFeatureVirtualSaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-virtual-base'
        shutil.rmtree(save_file, ignore_errors=True)
        name = 'test'
        td_name = 'base'
        f_type = ft.FEATURE_TYPE_STRING
        f = ft.FeatureVirtual(name, f_type)
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{name}.json')), f'No {name}.json found')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{name}.json')), f'No {name}.json found')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './load-virtual-base'
        shutil.rmtree(save_file, ignore_errors=True)
        f_type = ft.FEATURE_TYPE_STRING
        f = ft.FeatureVirtual('test', f_type)
        td = ft.TensorDefinition('base', [f])
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
