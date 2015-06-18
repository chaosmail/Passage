import unittest

import numpy as np

from passage.preprocessing import *


class TestPreprocessingFunctions(unittest.TestCase):

    def test_one_hot(self):

        Y = [0, 1, 2]
        expect = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        assert np.allclose(one_hot(Y), expect)

    def test_flatten(self):

        X = [[1], [2], [3]]
        expect = [1, 2, 3]

        assert np.array_equal(flatten(X), expect)

    def test_lbf(self):

        X = [1, 2, 3]
        b = [True, False, True]
        expect = [1, 3]

        assert np.array_equal(lbf(X, b), expect)

    def test_list_index(self):

        X = [1, 2, 3]
        indices = [2, 0]
        expect = [3, 1]

        assert np.array_equal(list_index(X, indices), expect)

    def test_tokenize(self):

        text = "This is a test"
        expect = ['This', 'is', 'a', 'test']

        assert np.array_equal(tokenize(text), expect)

    def test_token_encoder(self):

        tokens = [
            ['this', 'is', 'a', 'test'],
            ['let', 'us', 'try', 'the', 'test'],
            ['unit', 'tests', 'should', 'pass'],
        ]
        expect = {'test': 3}

        self.assertEqual(token_encoder(tokens, min_df=2), expect)

    def test_standardize_targets(self):

        def CategoricalCrossEntropy():
            """cost function mock"""
            pass

        Y = [0, 1, 2]
        expect = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        assert np.allclose(standardize_targets(Y, CategoricalCrossEntropy), expect)

    def test_standardize_targets_hinge(self):

        def SquaredHinge():
            """cost function mock"""
            pass

        Y = [0, 1, 2]
        expect = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]

        assert np.allclose(standardize_targets(Y, SquaredHinge), expect)


if __name__ == '__main__':
    unittest.main()
