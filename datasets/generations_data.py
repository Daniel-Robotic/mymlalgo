import numpy as np
import datasets.utils as utils
from typing import Sequence, Union, Tuple


def make_regression(
    n_samples: int = 100,
    n_features: int = 100,
    n_informative: int = 10,
    n_targets: int = 1,
    bias: float = 0.0,
    effective_rank: int | None = None,
    tail_strength: float = 0.5,
    noise: float = 0.0,
    shuffle: bool = True,
    coef: bool = False,
    random_state: int | None = None):
    """Generate a random regression problem (sklearn-like)."""
    rng = utils._rng(random_state)
    X = utils._make_design_matrix(n_samples, n_features, effective_rank, tail_strength, rng)

    coef_mat = np.zeros((n_features, n_targets))
    informative_idx = rng.choice(n_features, n_informative, replace=False)
    coef_mat[informative_idx] = rng.standard_normal(size=(n_informative, n_targets))

    y = X @ coef_mat + bias
    if noise:
        y += rng.normal(scale=noise, size=y.shape)

    if n_targets == 1:
        y = y.ravel()
        coef_mat = coef_mat.ravel()

    if shuffle:
        X, coef_mat = utils._shuffle_columns(X, rng, coef_mat)

    return (X, y, coef_mat) if coef else (X, y)


def make_classification(
    n_samples: int = 100,
    n_features: int = 20,
    *,
    n_informative: int = 2,
    n_redundant: int = 2,
    n_repeated: int = 0,
    n_classes: int = 2,
    n_clusters_per_class: int = 2,
    weights: Sequence[float] | None = None,
    flip_y: float = 0.01,
    class_sep: float = 1.0,
    hypercube: bool = True,
    shift: Union[float, np.ndarray] = 0.0,
    scale: Union[float, np.ndarray] = 1.0,
    shuffle: bool = True,
    random_state: int | None = None,
    return_centers: bool = False):
    """Simplified clone of sklearn.datasets.make_classification."""

    rng = utils._rng(random_state)

    if weights is None:
        weights = [1.0 / n_classes] * n_classes
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    samples_per_class = np.floor(weights * n_samples).astype(int)
    while samples_per_class.sum() < n_samples:
        samples_per_class[np.argmin(samples_per_class)] += 1

    samples_per_cluster = []
    for cls_total in samples_per_class:
        base, rem = divmod(cls_total, n_clusters_per_class)
        samples_per_cluster.extend([base + (i < rem) for i in range(n_clusters_per_class)])

    X_inf_parts, y_parts, centers = [], [], []
    for cluster_idx, ns in enumerate(samples_per_cluster):
        cls = cluster_idx // n_clusters_per_class
        if ns == 0:
            continue
        center = (rng.choice([-class_sep, class_sep], n_informative) if hypercube
                  else rng.uniform(-class_sep, class_sep, n_informative))
        centers.append(center)
        X_inf_parts.append(rng.normal(loc=center, scale=1.0, size=(ns, n_informative)))
        y_parts.append(np.full(ns, cls, dtype=int))

    X_inf = np.vstack(X_inf_parts)
    y = np.concatenate(y_parts)

    if n_redundant:
        B = rng.standard_normal(size=(n_informative, n_redundant))
        X_red = X_inf @ B + 0.01 * rng.normal(size=(X_inf.shape[0], n_redundant))
    else:
        X_red = np.empty((X_inf.shape[0], 0))

    if n_repeated:
        parent = np.hstack([X_inf, X_red])
        reps_idx = rng.choice(parent.shape[1], n_repeated, replace=True)
        X_rep = parent[:, reps_idx]
    else:
        X_rep = np.empty((X_inf.shape[0], 0))

    n_noise = n_features - n_informative - n_redundant - n_repeated
    if n_noise < 0:
        raise ValueError("Sum of informative, redundant and repeated exceeds n_features")
    X_noise = rng.standard_normal(size=(X_inf.shape[0], n_noise))

    X = np.hstack([X_inf, X_red, X_rep, X_noise])

    if flip_y:
        n_flip = int(flip_y * n_samples)
        idx = rng.choice(n_samples, n_flip, replace=False)
        y[idx] = rng.randint(0, n_classes, size=n_flip)

    X = X * scale + shift
    X, y = utils._shuffle_rows(X, y, rng)
    if shuffle:
        X, _ = utils._shuffle_columns(X, rng)

    if return_centers:
        return X, y, np.asarray(centers)
    return X, y


def make_blobs(
    n_samples: int | Sequence[int] = 100,
    n_features: int = 2,
    *,
    centers: int | np.ndarray | None = None,
    cluster_std: float | Sequence[float] = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    shuffle: bool = True,
    random_state: int | None = None,
    return_centers: bool = False,
):
    """Аналог *make_blobs* из scikit‑learn.

    Генерирует изотропные гауссовы «пятна» (blobs) — удобный датасет для
    демонстрации алгоритмов кластеризации.
    """

    rng = utils._rng(random_state)

    # --- 1. Определяем центры кластеров --------------------------------
    if centers is None:
        centers = 3  # по умолчанию как в sklearn

    if isinstance(centers, int):
        n_centers = centers
        centers = rng.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
    else:
        centers = np.asarray(centers, dtype=float)
        n_centers = centers.shape[0]
        if centers.shape[1] != n_features:
            raise ValueError("centers.shape[1] != n_features")

    # --- 2. Сколько объектов в каждом кластере -------------------------
    if isinstance(n_samples, int):
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        samples_per_center = utils._split_evenly(n_samples, n_centers)
        n_samples_total = n_samples
    else:
        if len(n_samples) != n_centers:
            raise ValueError("len(n_samples) must match number of centers")
        samples_per_center = list(n_samples)
        n_samples_total = sum(samples_per_center)

    # --- 3. Std для каждого кластера -----------------------------------
    if np.isscalar(cluster_std):
        cluster_std = [cluster_std] * n_centers
    elif len(cluster_std) != n_centers:
        raise ValueError("cluster_std length must equal number of centers")

    # --- 4. Генерируем данные -----------------------------------------
    X_parts, y_parts = [], []
    for idx, (ns, std, center) in enumerate(zip(samples_per_center, cluster_std, centers)):
        if ns == 0:
            continue
        X_parts.append(rng.normal(loc=center, scale=std, size=(ns, n_features)))
        y_parts.append(np.full(ns, idx, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # --- 5. Shuffle -----------------------------------------------------
    if shuffle:
        perm = rng.permutation(n_samples_total)
        X, y = X[perm], y[perm]

    if return_centers:
        return X, y, centers
    return X, y

