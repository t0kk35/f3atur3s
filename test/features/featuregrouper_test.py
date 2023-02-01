"""
Unit Tests for FeatureGrouper Creation
(c) 2023 tsm
"""
import unittest
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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
