"""Причинные структуры из реальных данных: выглядят ли они как пространство-время?

Берём частичные порядки, возникающие естественно, и меряем нашими различителями.
"""
import numpy as np


def transitive_closure(E, n):
    """Замыкание графа зависимостей."""
    R = np.zeros((n, n), dtype=bool)
    for a, b in E:
        R[b, a] = True                     # R[i,j]: j раньше i
    for _ in range(60):
        Rn = R | ((R.astype(np.float32) @ R.astype(np.float32)) > 0)
        if Rn.sum() == R.sum(): break
        R = Rn
    np.fill_diagonal(R, False)
    return R


def divisibility_order(n):
    """Числа 2..n+1, a раньше b если a делит b. Естественный порядок."""
    N = n
    nums = np.arange(2, n+2)
    R = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j and nums[j] % nums[i] == 0:
                R[j, i] = True
    return R


def subset_order(k):
    """Подмножества k-элементного множества по включению. Куб."""
    N = 2**k
    R = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j and (i & j) == i:
                R[j, i] = True
    return R


def word_order(words):
    """Слова: a раньше b, если a — подпоследовательность b."""
    N = len(words)
    R = np.zeros((N, N), dtype=bool)
    def issub(a, b):
        it = iter(b)
        return all(c in it for c in a)
    for i in range(N):
        for j in range(N):
            if i != j and len(words[i]) < len(words[j]) and issub(words[i], words[j]):
                R[j, i] = True
    return R
