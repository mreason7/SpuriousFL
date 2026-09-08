from collections import Counter

import numpy as np

from src.optimizers.weighting_strategy import (
    select_noreplacement_paper_fixed,
)


def test_oracle_one_hot_selection_is_balanced_and_without_replacement():
    matrix = np.array(
        [[1.0, 0.0, 0.0]] * 16
        + [[0.0, 1.0, 0.0]] * 4
        + [[0.0, 0.0, 1.0]] * 4
    ).T
    original = matrix.copy()
    labels = np.argmax(matrix, axis=0)

    np.random.seed(12345)
    for _ in range(10000):
        selected = select_noreplacement_paper_fixed(matrix, 9)
        counts = Counter(labels[selected])

        assert len(selected) == 9
        assert len(set(selected)) == 9
        assert counts == Counter({0: 3, 1: 3, 2: 3})

    np.testing.assert_array_equal(matrix, original)
