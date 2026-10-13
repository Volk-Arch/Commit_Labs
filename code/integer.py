"""Даёт ли заряженный мир ЦЕЛУЮ размерность k-1?

Обычный мир целых не даёт: k=4 -> 3.75, k=3 -> 2.90.
Заряженный при k=4 дал 3.013 ± 0.045, при k=5 — 4.042 ± 0.075.

Проверяем на нескольких k и N. Случайное совпадение не удержится
ни по k, ни по размеру мира.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from antimatter import rel_charged, closure
from core import rel_of, dimension

SEEDS = 6


def charged_dim(N, k, seed):
    rs = np.random.default_rng(seed)
    X = rs.random((N, k))
    q = np.where(rs.random(N) < 0.5, -1, 1)
    return dimension(R=closure(rel_charged(X, q, 'B')))


def plain_dim(N, k, seed):
    rs = np.random.default_rng(seed)
    return dimension(R=rel_of(rs.random((N, k))))


def agg(fn, N, k):
    v = [fn(N, k, 12000 + 307 * s) for s in range(SEEDS)]
    return float(np.mean(v)), float(np.std(v)) / np.sqrt(SEEDS)


print(f'семян {SEEDS}, ошибка — на среднее')
print()
print(f"{'N':>6}{'k':>3}{'заряженный':>17}{'цель k-1':>10}{'откл., σ':>10}"
      f"{'обычный k-1':>17}{'откл., σ':>10}")
for N in (1000, 2000, 4000):
    for k in (4, 5, 6):
        cm, cs = agg(charged_dim, N, k)
        pm, ps = agg(plain_dim, N, k - 1)
        tgt = k - 1
        print(f'{N:>6}{k:>3}{f"{cm:.3f} ± {cs:.3f}":>17}{tgt:>10}'
              f'{abs(cm - tgt) / cs:>10.1f}'
              f'{f"{pm:.3f} ± {ps:.3f}":>17}{abs(pm - tgt) / ps:>10.1f}')
    print()
