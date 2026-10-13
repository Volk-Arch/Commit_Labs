#!/usr/bin/env python3
"""ЧАСТИЧНОЕ согласие шкал: то, что выбрасывалось во всех измерениях.

ИДЕЯ (его). Сто событий связаны между собой, и не каждая пара выигрывает
по ВСЕМ шкалам. Из частичного согласия должны образоваться кластеры, а
кластеры могут вести себя как объекты — трубка, слой, n-мерная
структура, как у графена слои складываются в объём.

ЧТО ВЫБРАСЫВАЛОСЬ. Всюду отношение бралось бинарно: по всем шкалам или
нет. Между этими краями лежит число

    m(a,b) = сколько шкал ставит a раньше b,     0 <= m <= k

причём m(a,b) + m(b,a) = k. Край m = k есть наше «раньше».

ТОЧНЫЙ СМЫСЛ. Для конуса s_u = t - u*x с направлениями u по сфере
m есть доля шапки сферы, то есть для ПРОСТРАНСТВЕННО разнесённой пары

    m/k = (1 + dt/|dx|) / 2                      (ПРЕДСКАЗАНИЕ)

Отсюда m/k = 1/2 означает dt = 0, то есть ОДНОВРЕМЕННОСТЬ, определённую
относительно. А m/k от 1/2 до 1 есть угол разноса к световому конусу.
Величины |dx| это не даёт — только направление.

ЧТО МЕРИМ.
  1. Формулу шапки: m/k против (1 + dt/|dx|)/2.
  2. Его кластеры: граф «почти согласия» (m = k-1, то есть сразу за
     конусом) плюс similarity по строкам m. Спектральная кластеризация.
  3. Пространственно ли компактны кластеры — против случайных
     подмножеств того же размера.
  4. КОНТРОЛЬ: ящик из k независимых шкал, без всякой геометрии. Если
     кластеры там такие же, это комбинаторика, а не пространство.
  5. Размерность внутри кластера: слой это или объём.

  python3 partial.py
  python3 partial.py --n 300 --k 64
"""
import argparse
import math
import time

import numpy as np

from core import mm_dim as _mm_dim

# ПОЙМАННАЯ ОШИБКА (04.10). Здесь стояла своя табличка Мирхейма–Мейера
#   [1, 0.5], [2, 0.25], [3, 0.1875], [4, 0.1536], [5, 0.1318], ...
# и своя mm_dim, ЗАТЕНЯВШАЯ правильную из core.py. Табличка неверна с
# d = 3. Проверка численная и аналитическая:
#   d = 1 — цепь, связаны ВСЕ пары, то есть f = 1, а не 0.5;
#   d = 2 — в светоконусных координатах ромб есть квадрат, пара связана
#           когда обе координаты упорядочены одинаково, то есть ровно
#           1/2, а не 0.25;
#   замер (900 событий, 6 засевов): 0.49889 / 0.22746 / 0.09905 для
#           d = 2/3/4 — против верной формулы 0.5 / 0.22857 / 0.1.
# Верно: f_d = Г(d+1) Г(d/2) / (2 Г(3d/2)) — ровно то, что в core.py.
# По старой табличке настоящая четырёхмерная россыпь (f = 0.099) читалась
# как СЕМИМЕРНАЯ (между строками [8, 0.0964] и [7, 0.1053]).
# Куда разошлось: отсюда «0.1536 для d = 4» попало в sorkin-doc.md, где
# им обоснована переразметка секторов. Вывод там верен (0.1 < 0.20, то
# есть конус тем более в бине < 0.20), обоснование — нет; в доке
# исправлено на эмпирический эталон при N = 7.


def mm_dim(f):
    """Размерность Мирхейма–Мейера. Считает core.py, здесь только охрана."""
    if not np.isfinite(f) or f <= 0 or f >= 1:
        return float('nan')
    return _mm_dim(float(f))


def diamond(N, T, rs):
    out, got = [], 0
    while got < N:
        m = int((N - got) * 9) + 64
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        ok = (Y[:, 0] >= r) & (T - Y[:, 0] >= r)
        if ok.any():
            out.append(Y[ok])
            got += int(ok.sum())
    X = np.vstack(out)[:N]
    return X[np.argsort(X[:, 0])]


def scales_cone(X, k, rs):
    """k шкал s_u = t - u*x по направлениям u на единичной сфере."""
    u = rs.normal(size=(k, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return X[:, 0:1] - X[:, 1:] @ u.T          # (N, k)


def m_matrix(S):
    """m[i,j] = число шкал, ставящих i раньше j. Память O(N^2)."""
    N, k = S.shape
    m = np.zeros((N, N), np.int16)
    for c in range(k):
        s = S[:, c]
        m += (s[:, None] < s[None, :]).astype(np.int16)
    return m


def spectral(A, nc, rs):
    """Нормированный спектральный разрез на nc кластеров."""
    d = A.sum(1)
    d[d == 0] = 1.0
    L = np.eye(len(A)) - (A / np.sqrt(d)[:, None]) / np.sqrt(d)[None, :]
    val, vec = np.linalg.eigh(L)
    V = vec[:, 1:nc]
    nv = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.maximum(nv, 1e-12)
    # k-means, несколько запусков
    best, bl = None, np.inf
    for _ in range(8):
        C = V[rs.choice(len(V), nc, replace=False)]
        lab = np.zeros(len(V), int)
        for _ in range(40):
            dd = ((V[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            nl = dd.argmin(1)
            if (nl == lab).all():
                break
            lab = nl
            for c in range(nc):
                if (lab == c).any():
                    C[c] = V[lab == c].mean(0)
        loss = float(((V - C[lab]) ** 2).sum())
        if loss < bl:
            best, bl = lab.copy(), loss
    return best


def modularity(A, lab):
    m2 = A.sum()
    if m2 <= 0:
        return float('nan')
    d = A.sum(1)
    q = 0.0
    for c in np.unique(lab):
        s = lab == c
        q += A[np.ix_(s, s)].sum() / m2 - (d[s].sum() / m2) ** 2
    return float(q)


def main():
    ap = argparse.ArgumentParser(description='Частичное согласие шкал')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--k', type=int, default=48)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--nc', type=int, default=5)
    ap.add_argument('--seeds', type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n} событий, k={a.k} шкал, кластеров {a.nc}, '
          f'засевов {a.seeds}')

    # --- 1. формула шапки -------------------------------------------
    print('\n1. ФОРМУЛА ШАПКИ:  m/k = (1 + dt/|dx|)/2  для разнесённых')
    errs, cors = [], []
    for t in range(a.seeds):
        rs = np.random.default_rng(1601000 + 977 * t)
        X = diamond(a.n, a.t, rs)
        S = scales_cone(X, a.k, rs)
        M = m_matrix(S)
        dt = X[None, :, 0] - X[:, None, 0]
        dx = np.linalg.norm(X[None, :, 1:] - X[:, None, 1:], axis=-1)
        sp = (np.abs(dt) < dx) & ~np.eye(a.n, dtype=bool)
        pred = (1 + dt[sp] / dx[sp]) / 2
        got = M[sp] / a.k
        errs.append(float(np.abs(got - pred).mean()))
        cors.append(float(np.corrcoef(got, pred)[0, 1]))
    print(f'   средняя |ошибка| = {np.mean(errs):.4f} '
          f'(биномиальный шум 1/(2*sqrt(k)) = {1/(2*math.sqrt(a.k)):.4f})')
    print(f'   корреляция = {np.mean(cors):.4f}')
    print('   значит m/k есть ОТНОСИТЕЛЬНЫЙ УГОЛ разноса; '
          'm/k = 1/2 есть одновременность')

    # --- 2-5. кластеры ----------------------------------------------
    print(f'\n2. КЛАСТЕРЫ ИЗ ЧАСТИЧНОГО СОГЛАСИЯ')
    print(f"{'мир':>12}{'граф':>14}{'модульность':>14}{'случайно':>12}"
          f"{'протяж.':>11}{'случ.протяж':>13}{'разм.внутри':>13}")

    for world in ('конус', 'ящик'):
        for gname in ('почти (m=k-1)', 'около (|m-k/2|)'):
            Q, Q0, EX, EX0, DI = [], [], [], [], []
            for t in range(a.seeds):
                rs = np.random.default_rng(1601000 + 977 * t)
                X = diamond(a.n, a.t, rs)
                if world == 'конус':
                    S = scales_cone(X, a.k, rs)
                else:
                    S = rs.random((a.n, a.k))
                M = m_matrix(S)
                k = a.k
                if gname.startswith('почти'):
                    A = ((M >= k - max(1, k // 16)) & (M < k)).astype(float)
                    A = A + A.T
                else:
                    # близость по углу: вес падает с |m/k - 1/2|
                    ang = np.abs(M / k - 0.5)
                    A = np.where((M < k) & (M.T < k),
                                 np.exp(-((ang / 0.12) ** 2)), 0.0)
                np.fill_diagonal(A, 0.0)
                if A.sum() < 10:
                    continue
                lab = spectral(A, a.nc, rs)
                Q.append(modularity(A, lab))
                perm = rs.permutation(a.n)
                Q0.append(modularity(A, lab[perm]))
                # протяжённость кластера в настоящих координатах
                for c in np.unique(lab):
                    s = np.flatnonzero(lab == c)
                    if len(s) < 8:
                        continue
                    EX.append(float(np.linalg.norm(
                        X[s][:, 1:] - X[s][:, 1:].mean(0), axis=1).mean()))
                    rnd = rs.choice(a.n, len(s), replace=False)
                    EX0.append(float(np.linalg.norm(
                        X[rnd][:, 1:] - X[rnd][:, 1:].mean(0),
                        axis=1).mean()))
                    sub = M[np.ix_(s, s)] == a.k
                    f = sub.sum() / (len(s) * (len(s) - 1) / 2)
                    DI.append(mm_dim(f))
            if not Q:
                continue
            print(f'{world:>12}{gname:>14}'
                  f'{np.mean(Q):>14.4f}{np.mean(Q0):>12.4f}'
                  f'{np.mean(EX):>11.4f}{np.mean(EX0):>13.4f}'
                  f'{np.nanmean(DI):>13.2f}', flush=True)

    print('\n  протяж. заметно меньше случайной => кластеры '
          'ПРОСТРАНСТВЕННО компактны')
    print('  ящик даёт то же => комбинаторика, а не пространство')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
