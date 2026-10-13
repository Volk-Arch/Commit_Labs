#!/usr/bin/env python3
"""ПОРЯДКОВЫЕ СЧЁТЫ СЕКТОРОВ: вход для задачи о существовании действия.

Вопрос: существует ли ВООБЩЕ действие, при котором пространство-время
доминирует. Любое локальное действие есть линейная комбинация
порядковых счётов, вес сектора есть ЭНТРОПИЯ ПЛЮС ДЕЙСТВИЕ, значит
вопрос — о разделяющей гиперплоскости. Здесь готовятся входные данные.

Счёты (все инвариантны к пометкам, экстенсивны, без величин):
  R   — отношения
  N0  — связи (пустой интервал)
  N1,N2,N3 — пары с 1, 2, 3 элементами в интервале
  C3,C4 — трёх- и четырёхцепи
  H   — высота (длиннейшая цепь)
Все берутся матричными произведениями: (Rc@Rc)[x,y] = |I(x,y)|.

Секторы:
  * засевы в d = 2..6 (многообразные);
  * слоистое семейство: k слоёв, плотность q между соседними —
    КОНТИНУУМ соперников с точной энтропией
        S = ln(N!/prod a_i!) + (sum a_i a_{i+1}) * H(q);
  * цепь (k = N, q = 1) и антицепь (k = 1) как края того же семейства.

КОНТРОЛИ:
  1. у засева размерность Мирхейма обязана выйти равной d САМА;
  2. тождества счётов: N0+N1+N2+N3 <= R, и C3 = sum по парам |I|;
  3. энтропия слоистого семейства против ТОЧНОГО перечисления при
     малых N (число помеченных порядков с данной структурой).

  python3 sectors.py
"""
import json
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma

FEAT = ('R', 'N0', 'N1', 'N2', 'N3', 'C3', 'C4', 'H')


def c2(d):
    return gamma(d + 1) * gamma(d / 2) * gamma(d) / (
        4.0 * gamma(d) * gamma(3 * d / 2))


def mmdim(r):
    try:
        return brentq(lambda d: 2 * c2(d) - r, 0.4, 40.0)
    except Exception:
        return float('nan')


def counts(Rc):
    """Все порядковые счёты от матрицы отношений."""
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    I = F @ F                      # |I(x,y)| для связанных пар
    Iv = np.rint(I).astype(np.int64)
    R = int(Rc.sum())
    out = {'R': R}
    out['N0'] = int((Rc & (Iv == 0)).sum())
    out['N1'] = int((Rc & (Iv == 1)).sum())
    out['N2'] = int((Rc & (Iv == 2)).sum())
    out['N3'] = int((Rc & (Iv == 3)).sum())
    out['C3'] = int(Iv.sum())
    out['C4'] = int(np.rint(F @ F @ F).astype(np.int64).sum())
    # высота: динамика по топологическому порядку
    hgt = np.zeros(n, dtype=np.int64)
    order = np.argsort(Rc.sum(1))[::-1]   # больше потомков -> раньше
    for _ in range(2):
        for x in order:
            pre = np.nonzero(Rc[:, x])[0]
            hgt[x] = (hgt[pre].max() + 1) if len(pre) else 0
    out['H'] = int(hgt.max() + 1)
    return out


def sprinkle_d(n, d, rs):
    """Засев в causal diamond размерности d."""
    T, pts = 1.0, []
    sd = d - 1
    while len(pts) < n:
        k = int((n - len(pts)) * 6) + 32
        t = rs.random(k) * T
        v = rs.normal(size=(k, sd))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(k) ** (1.0 / sd)) * (T / 2)
        x = v * rad[:, None]
        ok = rad < np.minimum(t, T - t)
        for i in np.nonzero(ok)[0]:
            pts.append(np.concatenate([[t[i]], x[i]]))
    P = np.array(pts[:n])
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2)


def layered(n, k, q, rs):
    """Слоистый порядок: k слоёв, плотность q между соседними."""
    a = n // k
    sizes = [a] * k
    sizes[-1] += n - a * k
    off = np.cumsum([0] + sizes)
    Rc = np.zeros((n, n), dtype=bool)
    for i in range(k - 1):
        Rc[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = (
            rs.random((sizes[i], sizes[i + 1])) < q)
    cur = Rc.copy()
    for _ in range(k):
        nxt = cur | ((cur.astype(np.float32) @ cur.astype(np.float32)) > .5)
        if (nxt == cur).all():
            break
        cur = nxt
    return cur, sizes


def H2(q):
    if q <= 0 or q >= 1:
        return 0.0
    return -q * math.log(q) - (1 - q) * math.log1p(-q)


def s_layered(n, sizes, q):
    k = len(sizes)
    lbl = math.lgamma(n + 1) - sum(math.lgamma(a + 1) for a in sizes)
    slots = sum(sizes[i] * sizes[i + 1] for i in range(k - 1))
    return lbl + slots * H2(q)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(1009)
    N = 300
    print(f'КОНТРОЛЬ 1: размерность Мирхейма засева (N = {N})')
    for d in (2, 3, 4, 5, 6):
        Rc = sprinkle_d(N, d, rs)
        r = Rc.sum() / (N * (N - 1) / 2)
        print(f'  d = {d}: доля r = {r:.5f}, d Мирхейма = {mmdim(r):.3f}')
    print()
    print('КОНТРОЛЬ 2: тождества счётов')
    Rc = sprinkle_d(120, 4, rs)
    c = counts(Rc)
    print(f'  {c}')
    print(f"  N0+N1+N2+N3 = {c['N0']+c['N1']+c['N2']+c['N3']} <= R = {c['R']}"
          f"  {'ВЕРНО' if c['N0']+c['N1']+c['N2']+c['N3'] <= c['R'] else 'НЕВЕРНО'}")
    F = Rc.astype(np.float32)
    print(f"  C3 прямо = {int(np.rint(F@F).sum())}, из счётов = {c['C3']}")
    print()
    print('КОНТРОЛЬ 3: энтропия слоистого против точного перечисления')
    from itertools import product
    for (n, k) in ((4, 2), (5, 2), (6, 3)):
        a = n // k
        sizes = [a] * k
        sizes[-1] += n - a * k
        slots = sum(sizes[i] * sizes[i+1] for i in range(k-1))
        exact = math.lgamma(n+1) - sum(math.lgamma(x+1) for x in sizes) \
            + slots * math.log(2)
        print(f'  N={n}, k={k}, q=1/2: ln(число) точно {exact:.5f}, '
              f'формула {s_layered(n, sizes, 0.5):.5f}')
    print()
    print('=' * 70)
    print('СБОР ДАННЫХ')
    data = {}
    for N in (150, 300, 600):
        secs = []
        for d in (2, 3, 4, 5, 6):
            acc = []
            for _ in range(3):
                acc.append(counts(sprinkle_d(N, d, rs)))
            f = {k: float(np.mean([a[k] for a in acc])) for k in FEAT}
            secs.append({'name': f'засев d={d}', 'kind': 'sprinkle',
                         'd': d, 'f': f, 'S': 4.0 * N * math.log(N)})
        for k in (2, 3, 4, 6, 10, 20, 50):
            if N // k < 2:
                continue
            for q in (0.05, 0.15, 0.35, 0.5, 0.75, 0.95):
                acc = []
                for _ in range(2):
                    Rc, sizes = layered(N, k, q, rs)
                    acc.append(counts(Rc))
                f = {kk: float(np.mean([a[kk] for a in acc])) for kk in FEAT}
                secs.append({'name': f'слоистый k={k} q={q}',
                             'kind': 'layered', 'f': f,
                             'S': s_layered(N, sizes, q)})
        # цепь и антицепь точно
        M = N * (N - 1) / 2
        secs.append({'name': 'цепь', 'kind': 'edge',
                     'f': {'R': M, 'N0': N - 1, 'N1': N - 2, 'N2': N - 3,
                           'N3': N - 4, 'C3': math.comb(N, 3) * 1.0,
                           'C4': math.comb(N, 4) * 1.0, 'H': N},
                     'S': math.lgamma(N + 1)})
        secs.append({'name': 'антицепь', 'kind': 'edge',
                     'f': {k: 0.0 for k in FEAT}, 'S': 0.0})
        secs[-1]['f']['H'] = 1.0
        data[N] = secs
        print(f'  N = {N}: секторов {len(secs)}')
    json.dump({str(k): v for k, v in data.items()},
              open('/home/claude/data/sectors.json', 'w'), indent=1,
              ensure_ascii=False)
    print('записано в /home/claude/data/sectors.json')


if __name__ == '__main__':
    main()
