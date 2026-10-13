"""Восстанавливается ли знак заряда из одного порядка?

Признак на пару слаб (AUC 0.535), но у события десятки связей.
Собираем их: строим матрицу «похоже, одного знака / разных знаков»
и ищем лучшую раскраску в два цвета через ведущий собственный вектор.

Контроль обязателен: та же процедура на мире БЕЗ заряда должна дать
случайную точность. Иначе мы измеряем свою процедуру, а не мир.
"""
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from antimatter import rel_charged, closure
from core import rel_of


def pair_feature(C):
    """Размер интервала для каждой связанной пары."""
    F = np.asarray(C, np.float32)
    # interval[i,j] = сколько событий между j и i  (C[i,l] и C[l,j])
    return (F @ F) * np.asarray(C, np.float32)


def recover(C, true_q=None):
    """Двухцветная раскраска по интервалам; точность, если знаки известны."""
    I = pair_feature(C)
    mask = np.asarray(C, bool) | np.asarray(C, bool).T
    vals = I[np.asarray(C, bool)]
    if len(vals) == 0:
        return float('nan'), float('nan')
    thr = np.median(vals)
    # W = +1 если интервал мал (ставка: одного знака), -1 если велик
    W = np.zeros_like(I)
    W[np.asarray(C, bool)] = np.where(I[np.asarray(C, bool)] <= thr, 1.0, -1.0)
    W = W + W.T                                   # симметризуем
    w, v = np.linalg.eigh(W)
    lead = v[:, np.argmax(np.abs(w))]
    guess = np.where(lead >= 0, 1, -1)
    if true_q is None:
        return float('nan'), float('nan')
    acc = max((guess == true_q).mean(), (guess == -true_q).mean())
    return acc, float((true_q > 0).mean())


print('ВОССТАНОВЛЕНИЕ ЗНАКА ИЗ ПОРЯДКА (правило B)')
print(f"{'N':>6}{'доля анти':>11}{'точность':>11}{'случайно':>11}{'выше случая':>13}")
for N in (600, 1200, 2400):
    for f in (0.5, 0.3):
        accs = []
        base = []
        for s in range(4):
            rs = np.random.default_rng(500 + 17 * s)
            X = rs.random((N, 4))
            q = np.where(rs.random(N) < f, -1, 1)
            C = closure(rel_charged(X, q, 'B'))
            a, b = recover(C, q)
            accs.append(a)
            base.append(max(b, 1 - b))            # угадывание большинством
        m, sd = float(np.mean(accs)), float(np.std(accs))
        bm = float(np.mean(base))
        print(f'{N:>6}{f:>11.2f}{m:>11.3f}{bm:>11.3f}'
              f'{("ДА" if m > bm + 3 * sd / 2 else "нет"):>13}')

print()
print('КОНТРОЛЬ: та же процедура на мире БЕЗ заряда')
print('  (знаки приписаны случайно и на порядок не влияли)')
print(f"{'N':>6}{'точность':>11}{'случайно':>11}")
for N in (600, 1200, 2400):
    accs = []
    base = []
    for s in range(4):
        rs = np.random.default_rng(900 + 17 * s)
        X = rs.random((N, 4))
        q = np.where(rs.random(N) < 0.5, -1, 1)   # метки-пустышки
        C = rel_of(X)
        a, b = recover(C, q)
        accs.append(a)
        base.append(max(b, 1 - b))
    print(f'{N:>6}{float(np.mean(accs)):>11.3f}{float(np.mean(base)):>11.3f}')
