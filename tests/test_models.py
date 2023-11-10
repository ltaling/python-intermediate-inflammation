"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import os


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_load_from_json(tmpdir):
    from inflammation.models import load_json
    example_path = os.path.join(tmpdir, 'example.json')
    with open(example_path, 'w') as temp_json_file:
        temp_json_file.write('[{"observations":[1, 2, 3]},{"observations":[4, 5, 6]}]')
    result = load_json(example_path)
    npt.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])

def test_daily_max_integer():
    """Test that checks if indeed the greatest number is selected in an array 
        of positive inegers."""
    from inflammation.models import daily_max

    test_input = np.array([[2, 4],
                           [6, 8],
                           [10, 12]])
    test_result = np.array(([10,12]))

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), test_result)

def test_daily_min_integer():
    """Test that checks if indeed the greatest number is selected in an array 
        of positive inegers."""
    from inflammation.models import daily_min

    test_input = np.array([[3, 6],
                           [9, 12],
                           [15, 18]])
    test_result = np.array(([3,6]))

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), test_result)