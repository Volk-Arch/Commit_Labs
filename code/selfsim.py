#!/usr/bin/env python3
"""УСЛОВИЕ САМОПОДОБИЯ: прошлое обязано выглядеть как целое.

Третий класс правила (единственный незакрытый): условие на САМО
прошлое нового события целиком — неограниченное по размеру, но
локальное в порядке. Фаркаш его не касается (это не вес), порог
n ~ 108 не касается (это не запрет ограниченного размера).

УСЛОВИЕ: для каждого события x доля отношений внутри его прошлого
P(x) обязана совпадать с долей отношений целого.

    r(x) = |отношения внутри P(x)| / C(|P(x)|, 2)  ~  r(целого)

Проверяется при рождении, значит локально по построению.

СНАЧАЛА ПОПЫТКА УБИТЬ (вчерашний урок: улучшенное согласие
подозрительнее расхождения). Два кандидата в убийцы:

  1. ИЕРАРХИЧЕСКИЙ слоистый: блоки из блоков с плотностью q между
     соседними на каждом уровне — самоподобен по построению;
  2. ПОРЯДОК ДОМИНИРОВАНИЯ по k шкалам (собственный примитив
     проекта): «раньше по всем k шкалам». Прошлое точки в k-кубе —
     это снова равномерный k-куб, значит самоподобие ТОЧНОЕ, а
     доля отношений равна 2^{1-k} и там, и там.

Если условие проходят оба — оно пусто, и это выяснится сразу.

КОНТРОЛИ:
  1. засев ОБЯЗАН проходить: прошлое точки есть засев в меньший
     диамант, то есть тот же процесс. Если не проходит — ошибка в коде;
  2. цепь обязана давать ровно ноль отклонения (всякое прошлое — цепь);
  3. двумерное доминирование обязано статистически совпасть с засевом
     d=2, потому что светоконусные координаты u=t+x, v=t-x делают
     причинность 1+1 в точности доминированием по двум шкалам.

  python3 selfsim.py
"""
import math

import numpy as np

from sectors import sprinkle_d, layered, counts


def dominance(n, k, rs):
    """«Раньше по всем k шкалам» — порядок доминирования."""
    P = rs.random((n, k))
    return (P[:, None, :] < P[None, :, :]).all(axis=2)


def percolation(n, p, rs):
    A = np.triu(rs.random((n, n)) < p, 1)
    C = A.copy()
    while True:
        nxt = C | ((C.astype(np.float32) @ C.astype(np.float32)) > 0.5)
        if (nxt == C).all():
            return C
        C = nxt


def hier_layered(b, depth, q, rs):
    """Иерархический слоистый: блоки из блоков, плотность q между
    соседними блоками на КАЖДОМ уровне."""
    def build(j):
        if j == 0:
            return np.zeros((1, 1), dtype=bool)
        sub = [build(j - 1) for _ in range(b)]
        sz = sub[0].shape[0]
        n = b * sz
        R = np.zeros((n, n), dtype=bool)
        for i, s in enumerate(sub):
            R[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = s
        for i in range(b - 1):
            R[i * sz:(i + 1) * sz, (i + 1) * sz:(i + 2) * sz] = (
                rs.random((sz, sz)) < q)
        C = R.copy()
        while True:
            nxt = C | ((C.astype(np.float32) @ C.astype(np.float32)) > 0.5)
            if (nxt == C).all():
                break
            C = nxt
        return C
    return build(depth)


def chain(n):
    R = np.zeros((n, n), dtype=bool)
    for i in range(n):
        R[i, i + 1:] = True
    return R


def past_stats(Rc, nmin=20):
    """r(целого) и распределение r(x) по прошлым размера >= nmin."""
    n = Rc.shape[0]
    rg = Rc.sum() / (n * (n - 1) / 2)
    vals, sizes = [], []
    for x in range(n):
        idx = np.nonzero(Rc[:, x])[0]
        m = len(idx)
        if m < nmin:
            continue
        sub = Rc[np.ix_(idx, idx)]
        vals.append(sub.sum() / (m * (m - 1) / 2))
        sizes.append(m)
    if not vals:
        return rg, None, None, 0
    v = np.array(vals)
    dev = np.abs(v - rg) / max(rg, 1e-12)
    return rg, float(np.median(dev)), float(np.percentile(dev, 90)), len(v)


def report(name, Rc, nmin=20):
    rg, md, p90, cnt = past_stats(Rc, nmin)
    if md is None:
        print(f'  {name:>28}: r = {rg:.4f}, прошлых >= {nmin}: нет')
        return None
    print(f'  {name:>28}: r = {rg:.4f}, медиана откл. {md:>8.4f}, '
          f'90-й проц. {p90:>8.4f}, прошлых {cnt}')
    return md, p90


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10102026)
    N = 600
    print(f'=== КОНТРОЛИ (N = {N}) ===')
    report('цепь', chain(N))
    r2s = report('засев d=2', sprinkle_d(N, 2, rs))
    r2d = report('доминирование k=2', dominance(N, 2, rs))
    print('  (контроль 3: засев d=2 и доминирование k=2 обязаны '
          'совпасть — светоконусные координаты)')
    print()
    print('=== ЗАСЕВЫ (обязаны проходить) ===')
    for d in (2, 3, 4, 5, 6):
        report(f'засев d={d}', sprinkle_d(N, d, rs))
    print()
    print('=== ПОПЫТКА УБИТЬ 1: доминирование по k шкалам ===')
    for k in (2, 3, 4, 5, 6):
        Rc = dominance(N, k, rs)
        report(f'доминирование k={k}', Rc)
    print(f'  (теория: r = 2^(1-k) = '
          f'{", ".join(f"{2.0**(1-k):.4f}" for k in (2,3,4,5,6))})')
    print()
    print('=== ПОПЫТКА УБИТЬ 2: иерархический слоистый ===')
    for (b, depth) in ((3, 6), (5, 4), (24, 2)):
        for q in (0.1, 0.3, 0.6):
            Rc = hier_layered(b, depth, q, rs)
            report(f'иерарх. b={b} глуб={depth} q={q}', Rc)
    print()
    print('=== ОБЫЧНЫЙ СЛОИСТЫЙ (обязан проваливаться) ===')
    for k in (3, 6, 10, 20, 50):
        for q in (0.05, 0.3, 0.7):
            Rc, _ = layered(N, k, q, rs)
            report(f'слоистый k={k} q={q}', Rc)
    print()
    print('=== ПЕРКОЛЯЦИЯ в p* (проверка утверждения из файла) ===')
    ps = 1.15 * N ** -0.75
    for mult in (1.0, 2.0, 4.0):
        report(f'перколяция p={ps*mult:.4f}', percolation(N, ps * mult, rs))


if __name__ == '__main__':
    main()
