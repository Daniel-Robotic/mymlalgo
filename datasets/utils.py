import requests
import numpy as np


def _rng(random_state: int|None = None) -> np.random.RandomState:
    """Возвращает NumPy RandomState

	Args:
			 random_state (int | None, optional): Сдвиг генератора случайных чисел. Defaults to None.

	Returns:
			 np.random.RandomState: Генератор случайных чисел
	"""
    
    return np.random.RandomState(random_state)

def _make_design_matrix(n_samples: int,
                        n_features: int,
                        effective_rank: int | None,
                        tail_strength: float, rng) -> np.ndarray:
    """Создает матрицу дизайна с опциональной низкоранговой структурой (по аналогии со sklearn).

	Args:
			 n_samples (int): Количество примеров.
			 n_features (int): Количество признаков.
			 effective_rank (int | None): Эффективный ранг матрицы.
			 tail_strength (float): Сила хвостового шума.
			 rng: Генератор случайных чисел.

	Returns:
			 np.ndarray: Матрица дизайна.
	"""
    if effective_rank is None:
        return rng.standard_normal(size=(n_samples, n_features))

    k = min(n_samples, n_features)
    U, _ = np.linalg.qr(rng.standard_normal(size=(n_samples, k)))
    V, _ = np.linalg.qr(rng.standard_normal(size=(n_features, k)))
    singular = np.exp(-np.arange(k) / effective_rank)  # экспоненциально убывающий спектр
    X = U @ (singular * V.T)
    X += tail_strength * rng.standard_normal(size=(n_samples, n_features))

    return X


def _shuffle_columns(X: np.ndarray, rng, coef: np.ndarray | None = None):
    """Перемешивает столбцы матрицы X и вектора коэффициентов.

	 Args:
			 X (np.ndarray): Матрица признаков.
			 rng: Генератор случайных чисел.
			 coef (np.ndarray | None, optional): Вектор коэффициентов. Defaults to None.

	 Returns:
			 tuple: Перемешанные матрица X и вектор коэффициентов.
	 """
    
    n_features = X.shape[1]

    if n_features > 1:
        perm = rng.permutation(n_features)

        # гарантируем, что перестановка не тождественная
        if np.all(perm == np.arange(n_features)):
            # самый дешёвый способ «сдвинуть» порядок
            perm = np.roll(perm, 1)
    else:
        perm = np.arange(n_features)

    X = X[:, perm]
    if coef is not None:
        coef = coef[perm]

    return X, coef
	 
  
def _shuffle_rows(X: np.ndarray, y: np.ndarray, rng: np.random.RandomState):
    """
    Перемешать строки матрицы признаков X и соответствующие элементы целевого вектора y
    одинаковой случайной перестановкой.

    Args:
        X (np.ndarray): Матрица признаков (n_samples, n_features).
        y (np.ndarray): Вектор меток (n_samples,) или (n_samples, n_targets).
        rng (np.random.RandomState): Генератор случайных чисел.

    Returns:
        tuple: (X_shuffled, y_shuffled)
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    perm = rng.permutation(len(X))
    return X[perm], y[perm]
  

def _split_evenly(total: int, n_parts: int) -> list[int]:
    """
    Разделить `total` объектов на `n_parts` частей максимально равномерно.

    Например:
    _split_evenly(10, 3) -> [4, 3, 3]
    _split_evenly(5, 2) -> [3, 2]

    Args:
        total (int): Общее количество объектов.
        n_parts (int): На сколько частей делить.

    Returns:
        list[int]: Список количеств по каждой части.
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    base = total // n_parts
    remainder = total % n_parts

    result = [base + 1 if i < remainder else base for i in range(n_parts)]
    return result



def _load_dataset(url: str) -> None:
		pass