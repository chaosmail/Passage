import unittest

from passage.models import RNN
from passage.layers import Dense, GatedRecurrent, Embedding


class TestRNN(unittest.TestCase):

    def test_instanciate(self):

        layers = [
            Embedding(size=2, n_features=1),
            GatedRecurrent(size=128),
            Dense(size=2, activation='sigmoid', init='orthogonal')
        ]

        model = RNN(layers, cost='bce')

        assert hasattr(model, 'iterator')
        assert hasattr(model, 'updater')
        assert hasattr(model, 'cost')

        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')


if __name__ == '__main__':
    unittest.main()
