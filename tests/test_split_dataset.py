"""Regression tests for validation and normalization of split percentages.

Run without downloading data or training a model:
    PYTHONPATH=src python -m unittest discover -s tests -v
"""

import io
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import split_dataset


class SplitPercentageTests(unittest.TestCase):
    def calculate(self, train, validation, test):
        with patch.multiple(
            split_dataset,
            TRAIN_SPLIT_PERCENTAGE=train,
            VALIDATION_SPLIT_PERCENTAGE=validation,
            TEST_SPLIT_PERCENTAGE=test,
        ), redirect_stdout(io.StringIO()):
            return split_dataset.normaliser_pourcentages()

    def test_configured_split(self):
        self.assertEqual(self.calculate(70, 15, 15), (70.0, 15.0, 15.0))

    def test_normalizes_split_that_does_not_sum_to_100(self):
        result = self.calculate(7, 2, 1)
        for actual, expected in zip(result, (70.0, 20.0, 10.0)):
            self.assertAlmostEqual(actual, expected)

    def test_normalization_preserves_proportions(self):
        result = self.calculate(1, 1, 1)
        self.assertAlmostEqual(sum(result), 100.0)
        self.assertAlmostEqual(result[0], result[1])
        self.assertAlmostEqual(result[1], result[2])

    def test_rejects_negative_values(self):
        for position in range(3):
            with self.subTest(position=position):
                values = [70.0, 15.0, 15.0]
                values[position] = -1.0
                with self.assertRaises(ValueError):
                    self.calculate(*values)

    def test_rejects_nonfinite_values(self):
        for value in (math.nan, math.inf, -math.inf):
            for position in range(3):
                with self.subTest(value=value, position=position):
                    values = [70.0, 15.0, 15.0]
                    values[position] = value
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self.calculate(*values)

    def test_rejects_zero_total(self):
        with self.assertRaises(ValueError):
            self.calculate(0, 0, 0)

    def test_allows_one_nonzero_split(self):
        self.assertEqual(self.calculate(0, 0, 100), (0.0, 0.0, 100.0))
        self.assertEqual(self.calculate(100, 0, 0), (100.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
