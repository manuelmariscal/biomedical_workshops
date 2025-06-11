import os
import sys
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from corr import compute_correlation_matrix


def test_compute_correlation_matrix_symmetry_and_diagonal():
    data1 = pd.DataFrame({
        'a_x': [1, 2, 3],
        'a_y': [4, 5, 6]
    })
    data2 = pd.DataFrame({
        'a_x': [1, 2, 3],
        'a_y': [4, 5, 6]
    })

    corr_matrix = compute_correlation_matrix(data1, data2)
    values = corr_matrix.values.astype(float)

    # Check symmetry
    assert np.allclose(values, values.T, equal_nan=True)

    # Check diagonal values are 1
    assert np.allclose(np.diag(values), 1)
