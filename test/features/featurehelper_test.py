"""
Unit Tests for the FeatureHelper Class
(c) 2023 tsm
"""
import unittest
import f3atur3s.features as ft


class TestFeatureHelper(unittest.TestCase):

    def test_filter_feature(self):
        f1 = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureOneHot('Oh', ft.FEATURE_TYPE_INT_8, f1)
        fl = ft.FeatureHelper.filter_feature(ft.FeatureSource, [f1, f2])
        self.assertEqual(len(fl), 1, f'Should only have returned 1 feature. Got {len(fl)}')
        self.assertTrue(isinstance(fl[0], ft.FeatureSource), f'Should have returned a source feature')

    def test_filter_feature_not(self):
        f1 = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureOneHot('Oh', ft.FEATURE_TYPE_INT_8, f1)
        fl = ft.FeatureHelper.filter_not_feature(ft.FeatureSource, [f1, f2])
        self.assertEqual(len(fl), 1, f'Should only have returned 1 feature. Got {len(fl)}')
        self.assertTrue(isinstance(fl[0], ft.FeatureOneHot), f'Should have returned a source feature')

    def test_filter_feature_type(self):
        f1 = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        f2 = ft.FeatureOneHot('Oh', ft.FEATURE_TYPE_INT_8, f1)
        fl = ft.FeatureHelper.filter_feature_type(ft.FeatureTypeString, [f1, f2])
        self.assertEqual(len(fl), 1, f'Should only have returned 1 feature. Got {len(fl)}')
        self.assertEqual(fl[0], f1, f'Should have returned the source feature')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
