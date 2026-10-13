#!/usr/bin/env python3
"""НЕ РВАТЬ, А НЕ ДАВАТЬ ОБРАЗОВАТЬСЯ: ограничение на исходящие связи.

Разница принципиальная. Обрезание (`trunc-doc.md`) ПОРТИТ готовый засев.
Запрет ВЫРАЩИВАЕТ другой ансамбль, то есть это кинематическое
ограничение - а по главному выводу дня (`tri-doc.md`) именно они и
работают, в отличие от весов.

И тут сходятся оба замечания. В световых координатах, вставляя точки по
возрастанию u, связи В ПРОШЛОЕ фиксируются при рождении (новая точка
никогда не оказывается под старой). А связи В БУДУЩЕЕ продолжают
набегать, пока элемент «жив». Значит ограничивать надо ИСХОДЯЩИЕ - и это
ровно «смотреть в будущее».

Правило: растим по времени; новая точка НЕ ПРИНИМАЕТСЯ, если дала бы
кому-то (C+1)-ю исходящую связь.

Связи новой точки находятся за O(n) векторно: при u больше всех прошлых
её прошлое есть {w : v_w < v}, а связи - «обратные рекорды» по v, то
есть те w, у которых v_w < v и ни одна вставленная ПОЗЖЕ точка не имеет
v между v_w и v.

КОНТРОЛЬ: при C = бесконечность конструкция обязана дать обычный засев -
доля связанных 0.5, внутренняя 0.5, корон ноль.
"""
import math

import numpy as np

from crown import compat, triangles

NEG = -1e18


def links_of_candidate(vs, v):
    """индексы связей новой точки среди уже вставленных (порядок вставки)."""
    n = len(vs)
    if n == 0:
        return np.empty(0, int)
    cv = np.where(vs < v, vs, NEG)
    rev = cv[::-1]
    rm = np.maximum.accumulate(np.concatenate([[NEG], rev[:-1]]))
    isl = (rev > rm) & (rev > NEG)
    return (n - 1 - np.nonzero(isl)[0])[::-1]


def grow(n, cap, rs, maxtry=400):
    """u берём сразу как порядковые статистики, отбраковываем только v.

    Прежняя версия добирала u отбраковкой до возрастания - вероятность
    превысить последний u падает к нулю, и прогон зависал. Это модель
    РОСТА (u фиксированы, v с отбраковкой), а не точное условное
    распределение, и так и надо читать.
    """
    U = np.sort(rs.random(n))
    V = np.empty(n)
    outdeg = np.zeros(n, int)
    rej = 0
    for m in range(n):
        for _ in range(maxtry):
            v = rs.random()
            ls = links_of_candidate(V[:m], v)
            if cap >= 10 ** 8 or not len(ls) or (outdeg[ls] < cap).all():
                V[m] = v
                outdeg[ls] += 1
                break
            rej += 1
        else:
            return None, None, None
    return U, V, rej


def order(U, V):
    return (U[None, :] > U[:, None]) & (V[None, :] > V[:, None])


def stats(R, rs, lab, extra=''):
    n = R.shape[0]
    r = float(R.sum()) / (n * (n - 1) / 2)
    F = R.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    L = R & (Iv == 0)
    ii, jj = np.nonzero(R & (Iv >= 12))
    inn = float('nan')
    if len(ii):
        sel = rs.choice(len(ii), min(100, len(ii)), replace=False)
        vv = []
        for t in sel:
            i, j = ii[t], jj[t]
            mm = np.nonzero(R[i] & R[:, j])[0]
            if len(mm) >= 3:
                vv.append(float(R[np.ix_(mm, mm)].sum())
                          / (len(mm) * (len(mm) - 1) / 2))
        inn = float(np.mean(vv)) if vv else float('nan')
    G = compat(R, rs, m=1000)
    tri = triangles(G)[0] if G is not None else float('nan')
    od = L.sum(axis=1)
    print(f'{lab:>16}{r:>10.4f}{inn:>11.4f}{od.mean():>9.2f}'
          f'{od.max():>7}{float((od > 8).mean()):>10.3f}{tri:>9.1f}'
          f'  {extra}', flush=True)


def main():
    print(__doc__)
    rs = np.random.default_rng(90210)
    n = 1200
    print(f'N={n}')
    print(f"{'что':>16}{'доля св.':>10}{'внутр.':>11}{'исх.сред':>9}"
          f"{'макс':>7}{'доля>8':>10}{'корон':>9}")
    U, V, _ = grow(n, 10 ** 9, rs)
    stats(order(U, V), rs, 'контроль C=inf')
    for cap in (16, 12, 8, 6, 4):
        U, V, rej = grow(n, cap, rs)
        if U is None:
            print(f'{f"C={cap}":>16}   не выросло: отказов больше предела')
            continue
        stats(order(U, V), rs, f'C={cap}', f'отказов {rej}')


if __name__ == '__main__':
    main()
