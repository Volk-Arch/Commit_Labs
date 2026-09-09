"""Правило, зависящее не только от пары событий.

Обычное: A раньше B  <=>  A раньше по всем шкалам.
Здесь добавлено условие на ОКРУЖЕНИЕ — тогда связь зависит от того,
что есть вокруг, и появляется возможность влияния.
"""
import numpy as np


def rel_plain(X):
    N, k = X.shape
    R = np.ones((N, N), dtype=bool)
    for m in range(k): R &= (X[None, :, m] < X[:, None, m])
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def rel_nonlocal(X, mode='support', thr=1):
    """Нелокальные варианты правила."""
    R0 = rel_plain(X)
    if mode == 'plain':
        return R0
    A = R0.astype(np.float32)
    between = (A @ A)                      # сколько событий между парой
    if mode == 'support':
        # связь есть, только если между ними ХОТЯ БЫ thr событий
        # (кроме прямых связей — их сохраняем)
        keep = R0 & ((between >= thr) | (between == 0))
        return np.asarray(keep, dtype=bool)
    if mode == 'crowd':
        # связь рвётся, если между ними СЛИШКОМ много
        return np.asarray(R0 & (between <= thr), dtype=bool)
    if mode == 'witness':
        # связь есть, если у пары достаточно общих соседей
        common = (A.T @ A)
        return np.asarray(R0 & (common >= thr), dtype=bool)
    raise ValueError(mode)
