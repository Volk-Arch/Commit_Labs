"""Гипотеза: перевёрнутая шкала перестаёт быть измерением.

Если так, заряженный мир с k шкалами (f=0.5) должен быть геометрически
неотличим от ОБЫЧНОГО мира с k-1 шкалами — по размерности, высоте,
доле пар и спектру сразу.

Следствие, если подтвердится: чтобы иметь заряд И четыре измерения,
нужно k=5 — четыре шкалы на геометрию, одна на заряд.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from antimatter import rel_charged, closure
from core import rel_of, dimension, height, spectral

SEEDS = 5
N = 1500


def charged(k, seed):
    rs = np.random.default_rng(seed)
    X = rs.random((N, k))
    q = np.where(rs.random(N) < 0.5, -1, 1)
    return closure(rel_charged(X, q, 'B'))


def plain(k, seed):
    rs = np.random.default_rng(seed)
    return rel_of(rs.random((N, k)))


def stats(builder, *args):
    d, h, fr, sp = [], [], [], []
    for s in range(SEEDS):
        R = builder(*args, 9000 + 211 * s)
        d.append(dimension(R=R))
        h.append(height(R)[0])
        fr.append(R.sum() / (N * (N - 1)))
        sp.append(spectral(R))
    f = lambda v: (float(np.mean(v)), float(np.std(v)))
    return f(d), f(h), f(fr), f(sp)


print(f'N={N}, семян {SEEDS}')
print()
print(f"{'мир':>26}{'размерность':>18}{'высота':>16}{'доля пар':>16}{'спектр':>18}")
rows = {}
for k in (4, 5):
    rows[('зар', k)] = stats(charged, k)
    print(f'{f"заряженный k={k}, f=0.5":>26}'
          + ''.join(f'{f"{m:.3f} ± {s:.3f}":>{w}}'
                    for (m, s), w in zip(rows[('зар', k)], (18, 16, 16, 18))))
for k in (3, 4):
    rows[('об', k)] = stats(plain, k)
    print(f'{f"обычный k={k}":>26}'
          + ''.join(f'{f"{m:.3f} ± {s:.3f}":>{w}}'
                    for (m, s), w in zip(rows[('об', k)], (18, 16, 16, 18))))

print()
print('СРАВНЕНИЕ: заряженный k  против  обычного k-1')
print(f"{'пара':>30}{'величина':>14}{'разность':>12}{'в сигмах':>11}")
names = ['размерность', 'высота', 'доля пар', 'спектр']
for k in (4, 5):
    a, b = rows[('зар', k)], rows[('об', k - 1)]
    for idx, nm in enumerate(names):
        diff = a[idx][0] - b[idx][0]
        sig = np.hypot(a[idx][1], b[idx][1]) / np.sqrt(SEEDS)
        label = f'зар.k={k} / об.k={k-1}' if idx == 0 else ''
        print(f'{label:>30}{nm:>14}{diff:>12.4f}'
              f'{(abs(diff) / sig if sig > 0 else 0):>11.2f}')
