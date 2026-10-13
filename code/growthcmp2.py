#!/usr/bin/env python3
"""ГДЕ ПЕРЕЛОМ, И САМЫЙ ПРОСТОЙ СЛУЧАЙ: цепь.

Первый прогон: при N = 120 пространство-время побеждает в полосе
p = 0.01...0.2, при N = 240 и 480 — нигде. Окно закрывается с ростом N.
Здесь: (1) число слоёв расширено, (2) оценка Йенсена дополнена МЕРЕНОЙ
E[z^r] (Йенсен — оценка снизу, то есть слоистый недооценён), (3) добавлена
ЦЕПЬ, у которой всё считается без всякого моделирования:

    ln W_цепь = ln N! + (N-1) ln p

потому что у цепи R = M (платить за несвязанность нечего) и ровно N-1
связь. Это самый жёсткий возможный тест: если даже цепь побивает
пространство-время, мера роста отбора не даёт.
"""
import math

import numpy as np

from growthst import sprinkle, rel_links
from growthcmp import layer_rho, ln_w_layer, ln_w_st


def ln_w_chain(n, p):
    return math.lgamma(n + 1) + (n - 1) * math.log(p)


def ln_w_layer_mc(sizes, p, rs, reps=24):
    """МЕРЕНОЕ ln E[z^r] вместо оценки Йенсена."""
    k = len(sizes)
    off = np.cumsum([0] + list(sizes))
    n = sum(sizes)
    lam = -math.log1p(-p)
    rs_list = []
    for _ in range(reps):
        Rc = np.zeros((n, n), dtype=bool)
        for i in range(k - 1):
            Rc[off[i]:off[i+1], off[i+1]:off[i+2]] = (
                rs.random((sizes[i], sizes[i+1])) < p)
        cur = Rc.copy()
        for _ in range(k):
            nxt = cur | ((cur.astype(np.float32)
                          @ cur.astype(np.float32)) > 0.5)
            if (nxt == cur).all():
                break
            cur = nxt
        r = 0
        for i in range(k):
            for j in range(i + 2, k):
                r += int(cur[off[i]:off[i+1], off[j]:off[j+1]].sum())
        rs_list.append(r)
    a = np.array(rs_list, dtype=float)
    mx = a.max()
    lnE = lam * mx + math.log(np.exp(lam * (a - mx)).mean())
    within = sum(x * (x - 1) // 2 for x in sizes)
    nonadj = sum(sizes[i] * sizes[j] for i in range(k)
                 for j in range(i + 2, k))
    lbl = math.lgamma(n + 1) - sum(math.lgamma(x + 1) for x in sizes)
    return lbl - lam * (within + nonadj) + lnE, float(a.mean())


def main():
    print(__doc__.strip())
    print()
    rs = np.random.default_rng(99)
    print('ПЕРЕЛОМ: лучшее по p значение каждого сектора')
    print(f"{'N':>6}{'цепь':>12}{'слоистый':>12}{'лучшее p сл.':>14}"
          f"{'простр.-время':>15}{'лучшее p пв':>13}{'кто':>14}")
    PS = [0.002, 0.004, 0.007, 0.012, 0.02, 0.035, 0.06, 0.1, 0.16,
          0.25, 0.4, 0.6, 0.8]
    for n in (60, 90, 120, 160, 240, 480):
        best = None
        for _ in range(5):
            P = sprinkle(n, rs)
            R, L = rel_links(P)
            if best is None or L < best[1]:
                best = (R, L)
        R_st, L_st = best
        ks = [k for k in (2, 3, 4, 6, 10, 16, 24, 40) if n // k >= 2]
        shapes = []
        for k in ks:
            a = n // k
            s = [a] * k
            s[-1] += n - a * k
            shapes.append(tuple(s))
        bl, blp = -1e18, None
        bs, bsp = -1e18, None
        bc = -1e18
        for p in PS:
            for s in shapes:
                v, _ = ln_w_layer_mc(s, p, rs, 16)
                if v > bl:
                    bl, blp = v, p
            v = ln_w_st(n, L_st, R_st, p)
            if v > bs:
                bs, bsp = v, p
            bc = max(bc, ln_w_chain(n, p))
        who = 'ПРОСТР-ВРЕМЯ' if bs > max(bl, bc) else (
            'цепь' if bc > bl else 'слоистый')
        print(f'{n:>6}{bc:>12.1f}{bl:>12.1f}{blp:>14.3f}'
              f'{bs:>15.1f}{bsp:>13.3f}{who:>14}', flush=True)
    print()
    print('ЧИТАЕТСЯ ТАК. «лучшее по p» для каждого сектора взято')
    print('независимо — то есть каждому дана САМАЯ выгодная для него')
    print('мера из семейства. Пространство-время при этом ещё и получает')
    print('подарок: счёт многообразных взят по границе Милнора-Тома')
    print('(оценка СВЕРХУ), а засев выбран с наименьшим числом связей.')


if __name__ == '__main__':
    main()
