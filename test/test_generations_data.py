import numpy as np
import pytest

from datasets.generations_data import make_regression, make_classification, make_blobs


@pytest.mark.parametrize("n_samples, n_features, n_informative, n_targets, coef_flag", [
    (50, 10, 5, 1, False),
    (123, 7, 5, 3, True),
])
def test_make_regression_basic(n_samples, n_features, n_informative, n_targets, coef_flag):
    X, y, *rest = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        coef=coef_flag,
        random_state=42,
    )
    assert X.shape == (n_samples, n_features)
    if n_targets == 1:
        assert y.shape == (n_samples,)
    else:
        assert y.shape == (n_samples, n_targets)

    if coef_flag:
        coef_ = rest[0]
        if n_targets == 1:
            assert coef_.shape == (n_features,)
        else:
            assert coef_.shape == (n_features, n_targets)

def test_make_regression_deterministic():
    params = dict(
        n_samples=80, n_features=15, n_informative=5,
        noise=0.5, random_state=2025, coef=True
    )
    out1 = make_regression(**params)
    out2 = make_regression(**params)
    # все три выхода должны совпадать
    for a, b in zip(out1, out2):
        assert np.allclose(a, b)


@pytest.mark.parametrize("n_classes, weights", [
    (2, None),
    (3, [0.2, 0.5, 0.3]),
])
def test_make_classification_shape(n_classes, weights):
    n_samples, n_features = 150, 30
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=3,
        n_classes=n_classes,
        weights=weights,
        random_state=0,
    )

    assert X.shape == (n_samples, n_features)
    assert y.shape == (n_samples,)
    assert set(np.unique(y)) <= set(range(n_classes))

def test_make_classification_informative_overflow():
    with pytest.raises(ValueError):
        make_classification(
            n_samples=20,
            n_features=5,
            n_informative=4,
            n_redundant=2,
            n_classes=2,
        )

def test_make_classification_deterministic():
    params = dict(
        n_samples=200, n_features=25,
        n_informative=4, n_redundant=3, n_repeated=2,
        n_classes=4, random_state=77
    )
    X1, y1 = make_classification(**params)
    X2, y2 = make_classification(**params)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_make_blobs_basic():
    centers = [(-5, -5), (0, 0), (5, 5)]
    X, y, ctrs = make_blobs(
        n_samples=[30, 40, 50],
        n_features=2,
        centers=centers,
        cluster_std=[0.4, 0.8, 1.2],
        return_centers=True,
        random_state=123,
    )
    assert X.shape == (120, 2)
    assert y.shape == (120,)
    assert np.array_equal(ctrs, np.asarray(centers))
    assert set(np.unique(y)) == {0, 1, 2}

def test_make_blobs_std_mismatch():
    with pytest.raises(ValueError):
        make_blobs(
            n_samples=100,
            centers=4,
            n_features=3,
            cluster_std=[1.0, 0.5],
        )

def test_make_blobs_deterministic():
    params = dict(
        n_samples=90,
        centers=5,
        n_features=2,
        cluster_std=0.7,
        random_state=555,
    )
    X1, y1 = make_blobs(**params)
    X2, y2 = make_blobs(**params)
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)
