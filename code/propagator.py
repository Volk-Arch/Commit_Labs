"""Пропагатор: амплитуда перехода между двумя событиями.

Сумма по всем цепям от a к b, каждая с весом от накопленной фазы.
Квадрат модуля — вероятность перехода (правило Борна).
"""
import numpy as np


def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R & ((A @ A) == 0), dtype=bool)


def propagator(X, R, hop=0.5, maxlen=12):
    """Амплитуда K[a,b] = сумма по цепям a->b веса hop^длина * фазы.

    Фаза события: exp(2*pi*i*сумма координат).
    Возвращает комплексную матрицу.
    """
    N = len(X)
    L = links_of(R)
    ph = np.exp(2j*np.pi*X.sum(1))
    # шаг: из a в b по прямой связи, с фазой конечного события
    M = L.T.astype(np.complex128)*ph[None, :]*hop
    K = np.zeros((N, N), dtype=np.complex128)
    P = np.eye(N, dtype=np.complex128)
    for _ in range(maxlen):
        P = P @ M
        if not np.any(np.abs(P) > 1e-14):
            break
        K += P
    return K


def chain_length(R):
    """Число шагов по прямым связям между парами (длина максимальной цепи)."""
    L = links_of(R)
    N = len(R)
    D = np.full((N, N), -1, dtype=np.int32)
    cur = np.asarray(L, dtype=bool)
    step = 1
    while cur.any() and step < 30:
        D[(D < 0) & cur] = step
        cur = (cur.astype(np.float32) @ L.astype(np.float32)) > 0
        step += 1
    return D
