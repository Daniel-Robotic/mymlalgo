import pytest
import numpy as np
from datasets.utils import (_rng,
                            _make_design_matrix, 
                            _shuffle_columns,
                            _shuffle_rows,
                            _split_evenly)


# ---------------------------------------------------------------------------
#                               _rng
# ---------------------------------------------------------------------------

def test_rng_reproducibility():
    rng1 = _rng(42)
    rng2 = _rng(42)

    assert isinstance(rng1, np.random.RandomState)
    assert isinstance(rng2, np.random.RandomState)

    assert np.array_equal(rng1.rand(5), rng2.rand(5)), "Random outputs differ for same seed"


def test_rng_independence():
    rng1 = _rng(42)
    rng2 = _rng(43)

    assert not np.array_equal(rng1.rand(5), rng2.rand(5)), "Random outputs should differ for different seeds"

# ---------------------------------------------------------------------------
#                               _make_design_matrix
# ---------------------------------------------------------------------------

def test_make_design_matrix_no_rank():
    rng = _rng(0)
    X = _make_design_matrix(10, 5, None, 0.1, rng)
    assert X.shape == (10, 5)
    assert np.isfinite(X).all()

def test_make_design_matrix_with_rank():
    rng = _rng(0)
    X = _make_design_matrix(10, 5, effective_rank=3, tail_strength=0.1, rng=rng)
    assert X.shape == (10, 5)
    assert np.isfinite(X).all()

def test_make_design_matrix_deterministic():
    rng1 = _rng(0)
    rng2 = _rng(0)
    X1 = _make_design_matrix(10, 5, effective_rank=3, tail_strength=0.1, rng=rng1)
    X2 = _make_design_matrix(10, 5, effective_rank=3, tail_strength=0.1, rng=rng2)
    assert np.allclose(X1, X2)

# ---------------------------------------------------------------------------
#                               _shuffle_columns
# ---------------------------------------------------------------------------

def test_shuffle_columns_shape():
    rng = _rng(0)
    X = np.arange(12).reshape(4, 3)
    X_shuffled, _ = _shuffle_columns(X.copy(), rng)
    assert X_shuffled.shape == (4, 3)

def test_shuffle_columns_coef_alignment():
    rng = _rng(0)
    X = np.arange(12).reshape(4, 3)
    coef = np.array([1, 2, 3])
    X_shuffled, coef_shuffled = _shuffle_columns(X.copy(), rng, coef.copy())
    # Проверяем что порядок поменялся одинаково
    assert X_shuffled.shape == (4, 3)
    assert coef_shuffled.shape == (3,)
    # Очень важно: колонки X должны соответствовать перестановке coef
    for col_idx, coef_val in enumerate(coef_shuffled):
        assert (X_shuffled[:, col_idx] == X[:, coef == coef_val].flatten()).all()

def test_shuffle_columns_single_feature():
    rng = _rng(0)
    X = np.arange(4).reshape(4, 1)
    X_shuffled, _ = _shuffle_columns(X.copy(), rng)
    assert np.array_equal(X, X_shuffled), "Single-column matrix must remain the same"

# ---------------------------------------------------------------------------
#                               _shuffle_rows
# ---------------------------------------------------------------------------

def test_shuffle_rows_correctness():
    rng = _rng(42)
    X = np.arange(12).reshape(6, 2)
    y = np.array([0, 1, 2, 3, 4, 5])

    X_shuffled, y_shuffled = _shuffle_rows(X.copy(), y.copy(), rng)
    assert X_shuffled.shape == (6, 2)
    assert y_shuffled.shape == (6,)
    # Проверяем, что порядок соответствует
    for xi, yi in zip(X_shuffled, y_shuffled):
        idx = np.where((X == xi).all(axis=1))[0]
        assert y[idx] == yi

def test_shuffle_rows_mismatch_error():
    rng = _rng(0)
    X = np.zeros((5, 3))
    y = np.zeros(4)  # неправильная длина
    with pytest.raises(ValueError):
        _shuffle_rows(X, y, rng)

# ---------------------------------------------------------------------------
#                               _split_evenly
# ---------------------------------------------------------------------------

def test_split_evenly_exact():
    parts = _split_evenly(10, 5)
    assert parts == [2, 2, 2, 2, 2]

def test_split_evenly_with_remainder():
    parts = _split_evenly(10, 3)
    assert parts == [4, 3, 3]

def test_split_evenly_zero_total():
    parts = _split_evenly(0, 3)
    assert parts == [0, 0, 0]

def test_split_evenly_error_on_zero_parts():
    with pytest.raises(ValueError):
        _split_evenly(10, 0)
