# -*- coding: utf-8 -*-

import pytest
from uncertainty_forest.skeleton import fib

__author__ = "Ronak Mehta"
__copyright__ = "Ronak Mehta"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
