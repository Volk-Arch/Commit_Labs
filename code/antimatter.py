#!/usr/bin/env python3
"""Заряд как инвертированная шкала: два правила, две судьбы.

Каждое событие несёт знак q = +1 или -1. Доля отрицательных — f.

ПРАВИЛО A (переворот всех шкал). Для пары одного знака обычное
    правило; для пары разных знаков сравнение обращено на всех шкалах.
    Антисимметрично, но даёт 3-циклы: i(+) < j(+) < l(-) < i(+)
    совместимо с x_i < x_l < x_j.

ПРАВИЛО B (переворот одной шкалы). Обращена только шкала 0.
    Остальные k-1 шкал всегда дают строгий порядок, который отношение
    лишь уточняет, поэтому циклов быть не может ни при каком N.
    Нетранзитивно, но транзитивное замыкание — законное причинное
    множество с двумя секторами.

Для правила B главный вопрос: можно ли восстановить знаки из одного
порядка? Если да — заряд лежит в порядке, а не приписан сверху.

  python3 antimatter.py                 # оба правила
  python3 antimatter.py --rule B --f 0.1
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import rel_of, dimension, height
from majority import has_cycle, shortest_cycle


def rel_charged(X, q, rule='A'):
    """Отношение с зарядом. R[i,j] = True, если j раньше i."""
    N, k = X.shape
    same = (q[:, None] * q[None, :]) > 0          # пара одного знака
    if rule == 'A':
        flip = slice(0, k)                         # переворот всех шкал
    else:
        flip = slice(0, 1)                         # переворот шкалы 0
    R = np.ones((N, N), dtype=bool)
    for m in range(k):
        normal = X[None, :, m] < X[:, None, m]     # j раньше i
        if flip.start <= m < flip.stop:
            # на этой шкале пара разных знаков сравнивается наоборот
            R &= np.where(same, normal, ~normal)
        else:
            R &= normal
    np.fill_diagonal(R, False)
    return np.asarray(R, dtype=bool)


def closure(R):
    A = np.asarray(R, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(len(A), 2)))) + 1):
        F = np.asarray(A, np.float32)
        B = A | ((F @ F) > 0)
        if np.array_equal(B, A):
            break
        A = B
    np.fill_diagonal(A, False)
    return A


def auc(scores, labels):
    """Площадь под ROC через ранги: насколько признак разделяет классы.
    0.5 — не разделяет вовсе, 1.0 — разделяет идеально."""
    pos = np.asarray(scores)[labels]
    neg = np.asarray(scores)[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv))
    order = np.argsort(allv, kind='mergesort')
    r[order] = np.arange(1, len(allv) + 1)
    # средние ранги для связок
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    means = np.zeros(len(cnt))
    np.add.at(means, inv, r)
    means /= cnt
    r = means[inv]
    s = r[:len(pos)].sum()
    return (s - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def run_rule_a(N, k, seeds, fracs):
    """Правило A: считаем 3-циклы как функцию доли античастиц."""
    rows = []
    for f in fracs:
        cyc = []
        acy = []
        for s in range(seeds):
            rs = np.random.default_rng(1000 + s)
            X = rs.random((N, k))
            q = np.where(rs.random(N) < f, -1, 1)
            R = rel_charged(X, q, 'A')
            A = np.asarray(R, np.float32)
            cyc.append(float(np.trace(A @ A @ A)) / 3.0)
            acy.append(not has_cycle(R))
        rows.append((f, float(np.mean(cyc)), sum(acy), seeds))
    return rows


def run_rule_b(N, k, f, seed):
    """Правило B: ациклично ли, и восстанавливается ли заряд из порядка."""
    rs = np.random.default_rng(seed)
    X = rs.random((N, k))
    q = np.where(rs.random(N) < f, -1, 1)
    R = rel_charged(X, q, 'B')
    out = dict(
        f=f,
        frac_raw=R.sum() / (N * (N - 1)),
        acyclic=not has_cycle(R),
        short=shortest_cycle(R),
        viol=0,
    )
    C = closure(R)
    A = np.asarray(C, np.float32)
    out['viol'] = int((((A @ A) > 0) & ~C).sum())
    out['frac'] = C.sum() / (N * (N - 1))
    out['dim'] = dimension(R=C)
    out['height'] = int(height(C)[0])

    # сектора по отдельности: должны выглядеть одинаково
    for name, sel in (('плюс', q > 0), ('минус', q < 0)):
        sub = C[np.ix_(sel, sel)]
        n = sel.sum()
        out['dim_' + name] = dimension(R=sub) if n > 50 else float('nan')
        out['n_' + name] = int(n)

    # ВОССТАНОВИМОСТЬ: отличима ли пара разных знаков от пары одного
    # знака по признакам, считанным ТОЛЬКО из порядка
    i, j = np.nonzero(C)                            # j раньше i
    if len(i) > 200000:                             # подвыборка для скорости
        sel = rs.choice(len(i), 200000, replace=False)
        i, j = i[sel], j[sel]
    unlike = (q[i] * q[j]) < 0
    past = np.asarray(C, np.float32)                # past[i] = строка i
    interval = ((C[i] & C[:, j].T).sum(1))          # |интервал| между j и i
    cpast = ((C[i] & C[j]).sum(1))                  # общее прошлое
    cfut = ((C[:, i] & C[:, j]).T.sum(1))           # общее будущее
    out['auc_interval'] = auc(interval, unlike)
    out['auc_cpast'] = auc(cpast, unlike)
    out['auc_cfut'] = auc(cfut, unlike)
    out['auc_phase'] = auc(cpast - cfut, unlike)
    out['n_unlike'] = int(unlike.sum())
    out['n_like'] = int((~unlike).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description='Заряд как инвертированная шкала')
    ap.add_argument('--rule', choices=['A', 'B', 'both'], default='both')
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--f', type=float, default=0.3)
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()

    if a.rule in ('A', 'both'):
        print('ПРАВИЛО A: переворот всех шкал')
        print(f"{'доля анти':>11}{'3-циклов':>13}{'ациклично':>16}")
        for f, c, ac, tot in run_rule_a(a.n, a.k, a.seeds, [0.0, 0.001, 0.01, 0.1, 0.3, 0.5]):
            print(f'{f:>11.3f}{c:>13.1f}{f"{ac} из {tot}":>16}')
        print()

    if a.rule in ('B', 'both'):
        print('ПРАВИЛО B: переворот одной шкалы')
        for f in (0.0, 0.1, 0.3, 0.5):
            r = run_rule_b(a.n, a.k, f, 11)
            print(f"  доля анти {r['f']:.2f}: ациклично {r['acyclic']}, "
                  f"кратчайший цикл {r['short']}, "
                  f"доля пар {r['frac_raw']:.4f} -> {r['frac']:.4f} после замыкания")
            print(f"      размерность {r['dim']:.3f}, высота {r['height']}, "
                  f"сектора: + {r['dim_плюс']:.3f} ({r['n_плюс']}), "
                  f"- {r['dim_минус']:.3f} ({r['n_минус']})")
            if f > 0:
                print(f"      восстановимость знака из порядка (0.5 = никак):"
                      f" интервал {r['auc_interval']:.3f},"
                      f" общ.прошлое {r['auc_cpast']:.3f},"
                      f" общ.будущее {r['auc_cfut']:.3f},"
                      f" фаза {r['auc_phase']:.3f}")
            print()


if __name__ == '__main__':
    main()
