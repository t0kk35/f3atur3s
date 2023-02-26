"""
Unit Tests for FeatureGrouper Creation
(c) 2023 tsm
"""
import os
import unittest
import shutil
import f3atur3s as ft


def feature_expression(param: int):
    return param == 1


class TestFeatureGrouper(unittest.TestCase):
    def test_creation_base(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_FLOAT
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fs = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        ff = ft.FeatureFilter('Filter', ft.FEATURE_TYPE_BOOL, feature_expression, [fs])
        tp = ft.TIME_PERIOD_DAY
        tw = 3
        ag = ft.AGGREGATOR_COUNT
        f = ft.FeatureGrouper(name, f_type, fa, fs, ff, tp, tw, ag)
        self.assertIsInstance(f, ft.FeatureGrouper, f'Unexpected Type {type(f)}')
        self.assertEqual(f.name, name, f'Feature Name should be {name}')
        self.assertEqual(f.type, f_type, f'Feature Type should be {f_type}')
        self.assertEqual(f.group_feature, fs, f'Group should have been {fs}')
        self.assertEqual(f.filter_feature, ff, f'Filter should have been {ff}')
        self.assertEqual(f.time_period, tp, f'TimePeriod should have been {tp}')
        self.assertEqual(f.time_window, tw, f'TimeWindow should have been {tw}')
        self.assertEqual(f.aggregator, ag, f'Aggregator should have been {ag}')
        self.assertEqual(len(f.embedded_features), 3, 'Should have had 4 embedded features')
        self.assertEqual(f.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'String should have learning type cont')

    def test_creation_base_optional_features(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_FLOAT
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fs = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        tp = ft.TIME_PERIOD_DAY
        tw = 3
        ag = ft.AGGREGATOR_COUNT
        f = ft.FeatureGrouper(name, f_type, fa, fs, None, tp, tw, ag)
        self.assertIsInstance(f, ft.FeatureGrouper, f'Unexpected Type {type(f)}')
        self.assertEqual(f.name, name, f'Feature Name should be {name}')
        self.assertEqual(f.type, f_type, f'Feature Type should be {f_type}')
        self.assertEqual(f.group_feature, fs, f'Group should have been {fs}')
        self.assertEqual(f.filter_feature, None, f'Filter should have been None')
        self.assertEqual(f.time_period, tp, f'TimePeriod should have been {tp}')
        self.assertEqual(f.time_window, tw, f'TimeWindow should have been {tw}')
        self.assertEqual(f.aggregator, ag, f'Aggregator should have been {ag}')
        self.assertEqual(len(f.embedded_features), 2, 'Should have had 2 embedded features')
        self.assertEqual(f.learning_category, ft.LEARNING_CATEGORY_CONTINUOUS, f'String should have learning type cont')

    def test_creation_not_float_bad(self):
        name = 'test'
        f_type = ft.FEATURE_TYPE_STRING
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fs = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        ff = ft.FeatureFilter('Filter', ft.FEATURE_TYPE_BOOL, feature_expression, [fs])
        tp = ft.TIME_PERIOD_DAY
        tw = 3
        ag = ft.AGGREGATOR_COUNT
        with self.assertRaises(ft.FeatureDefinitionException):
            # Own type is not float
            _ = ft.FeatureGrouper(name, f_type, fa, fs, ff, tp, tw, ag)
        with self.assertRaises(ft.FeatureDefinitionException):
            # base is not a float
            _ = ft.FeatureGrouper(name, ft.FEATURE_TYPE_FLOAT, fs, fs, ff, tp, tw, ag)

    def test_equality(self):
        name_1 = 'test_1'
        name_2 = 'test_2'
        f_type_1 = ft.FEATURE_TYPE_FLOAT_64
        f_type_2 = ft.FEATURE_TYPE_FLOAT_32
        fa_1 = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fa_2 = ft.FeatureSource('Amount2', ft.FEATURE_TYPE_FLOAT)
        fs_1 = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        fs_2 = ft.FeatureSource('Source2', ft.FEATURE_TYPE_STRING)
        ff_1 = ft.FeatureFilter('Filter', ft.FEATURE_TYPE_BOOL, feature_expression, [fs_1])
        ff_2 = ft.FeatureFilter('Filter2', ft.FEATURE_TYPE_BOOL, feature_expression, [fs_2])
        tp_1 = ft.TIME_PERIOD_DAY
        tp_2 = ft.TIME_PERIOD_WEEK
        tw_1 = 3
        tw_2 = 4
        ag_1 = ft.AGGREGATOR_COUNT
        ag_2 = ft.AGGREGATOR_STDDEV

        fg_1 = ft.FeatureGrouper(name_1, f_type_1, fa_1, fs_1, ff_1, tp_1, tw_1, ag_1)
        fg_9 = ft.FeatureGrouper(name_2, f_type_1, fa_1, fs_1, ff_1, tp_1, tw_1, ag_2)
        fg_2 = ft.FeatureGrouper(name_1, f_type_1, fa_1, fs_1, ff_1, tp_1, tw_1, ag_1)
        fg_3 = ft.FeatureGrouper(name_1, f_type_2, fa_1, fs_1, ff_1, tp_1, tw_1, ag_1)
        fg_4 = ft.FeatureGrouper(name_1, f_type_1, fa_1, fs_2, ff_1, tp_1, tw_1, ag_1)
        fg_5 = ft.FeatureGrouper(name_1, f_type_1, fa_1, fs_1, ff_2, tp_1, tw_1, ag_1)
        fg_6 = ft.FeatureGrouper(name_1, f_type_1, fa_1, fs_1, ff_1, tp_2, tw_1, ag_1)
        fg_7 = ft.FeatureGrouper(name_1, f_type_1, fa_1, fs_1, ff_1, tp_1, tw_2, ag_1)
        fg_8 = ft.FeatureGrouper(name_1, f_type_1, fa_1, fs_1, ff_1, tp_1, tw_1, ag_2)
        fg_10 = ft.FeatureGrouper(name_1, f_type_1, fa_2, fs_1, ff_1, tp_1, tw_1, ag_1)

        self.assertEqual(fg_1, fg_2, f'Should have been equal')
        self.assertNotEqual(fg_1, fg_9, f'Should not have been equal. Different Name')
        self.assertNotEqual(fg_1, fg_3, f'Should have been not equal. Different Type')
        self.assertNotEqual(fg_1, fg_10, f'Should have been not equal. Different Base Feature')
        self.assertNotEqual(fg_1, fg_4, f'Should not have been equal. Different Group Feature')
        self.assertNotEqual(fg_1, fg_5, f'Should not have been equal. Different Filter Feature')
        self.assertNotEqual(fg_1, fg_6, f'Should not have been equal. Different Time Period')
        self.assertNotEqual(fg_1, fg_7, f'Should not have been equal. Different Time Window')
        self.assertNotEqual(fg_1, fg_8, f'Should not have been equal. Different Aggregator')


class TestFeatureConcatSaveLoad(unittest.TestCase):
    def test_save_base(self):
        save_file = './save-grouper-base'
        shutil.rmtree(save_file, ignore_errors=True)
        name = 'test'
        a_name = 'amount'
        s_name = 'source'
        f_type = ft.FEATURE_TYPE_FLOAT
        fa = ft.FeatureSource(a_name, ft.FEATURE_TYPE_FLOAT)
        fs = ft.FeatureSource(s_name, ft.FEATURE_TYPE_STRING)
        tp = ft.TIME_PERIOD_DAY
        tw = 3
        ag = ft.AGGREGATOR_COUNT
        f = ft.FeatureGrouper(name, f_type, fa, fs, None, tp, tw, ag)
        td_name = 'base'
        td = ft.TensorDefinition(td_name, [f])
        ft.TensorDefinitionSaver.save(td, save_file)
        self.assertTrue(os.path.exists(save_file), f'File does not exist {save_file}.')
        self.assertTrue(os.path.isdir(save_file), f'{save_file} does not seem to be a directory.')
        self.assertTrue(os.path.exists(os.path.join(save_file, f'tensor.json')), f'Did not find tensor.json')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features')), f'Not <features> directory in {save_file}')
        self.assertTrue(os.path.isdir(os.path.join(save_file, 'features')), f'<features> is not a directory')
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{name}.json')), f'No {name}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{name}.json')), f'No {name}.json')
        # We Should have the base feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{a_name}.json')), f'No {a_name}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{a_name}.json')), f'No {a_name}.json')
        # Should have the grouper feature also
        self.assertTrue(os.path.exists(os.path.join(save_file, 'features', f'{s_name}.json')), f'No {s_name}.json')
        self.assertTrue(os.path.isfile(os.path.join(save_file, 'features', f'{s_name}.json')), f'No {s_name}.json')
        shutil.rmtree(save_file, ignore_errors=True)

    def test_load_base(self):
        save_file = './load-grouper-base'
        shutil.rmtree(save_file, ignore_errors=True)
        name = 'test'
        f_type = ft.FEATURE_TYPE_FLOAT
        fa = ft.FeatureSource('Amount', ft.FEATURE_TYPE_FLOAT)
        fs = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        tp = ft.TIME_PERIOD_DAY
        tw = 3
        ag = ft.AGGREGATOR_COUNT
        f = ft.FeatureGrouper(name, f_type, fa, fs, None, tp, tw, ag)
        td_name = 'base'
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
