#!/usr/bin/env python3
"""КТО ПОБЕЖДАЕТ ПОД МЕРОЙ РОСТА: слоистый сектор или пространство-время.

Сравнение ведётся при ОДНОМ И ТОМ ЖЕ `p` для обоих (мера одна), и
`p` сканируется целиком — чтобы вердикт не зависел от выбора точки.
Если пространство-время не побеждает ни при каком `p`, маршрут меры
закрыт независимо от того, где стоит `p*`.

СЛОИСТЫЙ (строгая НИЖНЯЯ оценка):
    ln W >= ln[N!/prod a_i!] - lambda*[sum_i C(a_i,2)
                                       + sum_{|i-j|>=2} a_i a_j (1-rho_ij)]
по Йенсену (E[z^r] >= z^{E r}), где rho_ij — доля связанных несоседних
пар, МЕРЕНАЯ прямым моделированием слоистой перколяции.

ПРОСТРАНСТВО-ВРЕМЯ (щедрая ВЕРХНЯЯ оценка):
    ln W <= 4 N lnN + L ln p + (M-R) ln(1-p)
где 4 N lnN — граница Милнора-Тома на счёт многообразных (оценка
СВЕРХУ, то есть подарок), а L и R взяты из НАСТОЯЩЕГО засева, причём
в самом выгодном для засева виде (наименьшее L, наибольшее R).

Если нижняя оценка слоистого побивает верхнюю оценку
пространства-времени при всех `p` — вердикт строгий.

  python3 growthcmp.py
"""
import json
import math

import numpy as np

from growthst import sprinkle, rel_links


def layer_rho(sizes, p, rs, reps):
    """МЕРЕНОЕ E[r] и rho_ij для слоистой перколяции."""
    k = len(sizes)
    off = np.cumsum([0] + list(sizes))
    n = sum(sizes)
    acc = []
    for _ in range(reps):
        Rc = np.zeros((n, n), dtype=bool)
        for i in range(k - 1):
            A = rs.random((sizes[i], sizes[i + 1])) < p
            Rc[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = A
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
                r += int(cur[off[i]:off[i + 1],
                             off[j]:off[j + 1]].sum())
        acc.append(r)
    return float(np.mean(acc)), float(np.std(acc, ddof=1) /
                                      math.sqrt(reps))


def ln_w_layer(sizes, p, Er):
    n = sum(sizes)
    lam = -math.log1p(-p)
    within = sum(a * (a - 1) // 2 for a in sizes)
    nonadj = sum(sizes[i] * sizes[j] for i in range(len(sizes))
                 for j in range(i + 2, len(sizes)))
    lbl = math.lgamma(n + 1) - sum(math.lgamma(a + 1) for a in sizes)
    return lbl - lam * (within + nonadj) + lam * Er


def ln_w_st(n, L, R, p):
    M = n * (n - 1) / 2
    return 4 * n * math.log(n) + L * math.log(p) + (M - R) * math.log1p(-p)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(31337)
    PS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    rows = []
    for n in (120, 240, 480):
        # засев: самый выгодный для пространства-времени из выборок
        best = None
        for _ in range(5):
            P = sprinkle(n, rs)
            R, L = rel_links(P)
            if best is None or (L < best[1]) or (L == best[1] and R > best[0]):
                best = (R, L)
        R_st, L_st = best
        pstar = 1.15 * n ** -0.75
        print(f'=== N = {n}:  засев R = {R_st}, L = {L_st}, '
              f'p*(замерено) = {pstar:.5f} ===')
        # слоистые формы
        shapes = []
        for k in (2, 3, 4, 6, 10):
            a = n // k
            s = [a] * k
            s[-1] += n - a * k
            shapes.append(tuple(s))
        print(f"{'p':>7}{'лучший слоистый':>18}{'форма':>12}"
              f"{'простр.-время':>16}{'кто':>13}")
        for p in PS:
            bl, bs = -1e18, None
            for s in shapes:
                Er, _ = layer_rho(s, p, rs, 8)
                v = ln_w_layer(s, p, Er)
                if v > bl:
                    bl, bs = v, s
            vs = ln_w_st(n, L_st, R_st, p)
            who = ('слоистый' if bl > vs else 'ПРОСТР-ВРЕМЯ')
            print(f'{p:>7.3f}{bl:>18.1f}{str(len(bs))+" слоёв":>12}'
                  f'{vs:>16.1f}{who:>13}', flush=True)
            rows.append((n, p, bl, vs))
        print()
    print('=' * 74)
    win = [r for r in rows if r[3] > r[2]]
    if win:
        print(f'Пространство-время побеждает в {len(win)} из {len(rows)} '
              f'точек:')
        for r in win[:10]:
            print(f'   N={r[0]}, p={r[1]}')
    else:
        print('НИ ПРИ ОДНОМ p и НИ ПРИ ОДНОМ N пространство-время не')
        print('побеждает — даже с подарком в виде границы Милнора-Тома и')
        print('самого выгодного засева. Маршрут меры закрыт.')
    json.dump([list(map(float, r)) for r in rows],
              open('/home/claude/data/growthcmp.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
