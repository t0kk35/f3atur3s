"""
Unit Tests for FeatureSource Creation
(c) 2023 tsm
"""
import os
import unittest
import shutil
import f3atur3s as ft


class TestFeatureStackedSeries(unittest.TestCase):
    def test_creation_base(self):
        name = 'SeriesStacked'
        sf = ft.FeatureSource('Source', ft.FEATURE_TYPE_STRING)
        ss = ft.FeatureSeriesStacked(name, ft.FEATURE_TYPE_INT_8, 5, [sf])
        self.assertIsInstance(ss, ft.FeatureOneHot, f'Not expected type {type(oh)}')
        self.assertEqual(ss.name, name, f'Feature Name should be {name}')
        self.assertEqual(ss.base_feature, sf, f'Base Feature not set correctly')
        self.assertEqual(len(ss.embedded_features), 1, f'Should only have 1 emb feature {len(oh.embedded_features)}')
        self.assertIn(sf, ss.embedded_features, 'Base Feature should be in emb feature list')
        self.assertEqual(ss.inference_ready, False, 'Should be not inference ready upon creation')
        self.assertEqual(oh.type, ft.FEATURE_TYPE_INT_8, 'Must always be int-8 type. Smallest possible')
        self.assertEqual(oh.learning_category, ft.LEARNING_CATEGORY_NONE, f'Must have learning category Binary')
        self.assertIsInstance(hash(ss), int, f'Hash function not working')
