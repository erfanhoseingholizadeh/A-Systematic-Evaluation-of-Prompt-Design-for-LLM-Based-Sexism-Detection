import random

import numpy as np

from sexism_prompting.seeding import set_seed


def test_set_seed_makes_random_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_reproducible():
    set_seed(7)
    a = np.random.rand(5)
    set_seed(7)
    b = np.random.rand(5)
    assert (a == b).all()


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = random.random()
    set_seed(2)
    b = random.random()
    assert a != b
