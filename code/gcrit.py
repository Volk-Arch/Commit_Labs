#!/usr/bin/env python3
"""КРИТЕРИЙ ОДНОЙ ФУНКЦИЕЙ: гибнет ли слоистый сектор.

Из точной факторизации (проверена против перечисления порядков):
для структуры «второй слой разбит на блоки по m, каждый блок обслуживают
d строк первого слоя» имеем a_1 = d a_2/m и ТОЧНО

    G = g_d(m)^{a_2/m},
    g_d(m) = sum_{t=0..m} C(m,t) u^t phi(t)^d
           = (1+u)^m + sum_{t=1..3} C(m,t) u^t (phi(t)^d - 1)

потому что phi(t) = 1 при t >= 4 (w(t) = 0), а блоки независимы.

Число таких `M12` экспоненциально по a_2 ln a_2 и a_1 ln a_1, то есть
даёт лишь O(N ln N). Значит ВЕСЬ вклад порядка N² сидит в a_3 ln|G|:

    ln|Z| = a_2 a_3 * ln|g_d(m)|/m + O(N ln N)

Оптимум по форме при a_1 + a_2 + a_3 = N и a_1 = (d/m) a_2 берётся
аналитически: beta* = 1/(2(1+d/m)), gamma* = 1/2, откуда

    ln|Z| ~ c N²,   c(m,d) = ln|g_d(m)| / (4(m+d))

ВЕРДИКТ СВОДИТСЯ К ОДНОМУ НЕРАВЕНСТВУ:

  если |g_d(m)| <= 1 для ВСЕХ m, d — положительного N²-вклада нет,
    сектор сбит до O(N ln N), и там с ним спорит пространство-время
    (пол 2 N lnN, счёт <= 4 N lnN по Милнору-Тому);
  если найдётся (m,d) с |g_d(m)| > 1 — сектор выживает с c > 0 и
    побивает пространство-время ОДНИМ КОЛИЧЕСТВОМ при любом N.

КОНТРОЛИ:
  1. g_d(m) против прямой суммы по 2^{a_2} подмножествам с явными
     строками (а не по формуле);
  2. m=1,d=1: g = 1 + e^{8 i th}, то есть |g| = 2|cos 4th|;
  3. точный максимум |G| по ВСЕМ композициям при a1 = 2 и 3 обязан
     достигаться внутри семейства (иначе семейство неполно).

  python3 gcrit.py
"""
import itertools
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)
WD = {1: 9.0, 2: -16.0, 3: 8.0}


def phi(t, th):
    if t <= 0:
        return 1.0 + 0.0j
    if t > 3:
        return 1.0 + 0.0j
    return np.exp(1j * th * WD[t])


def g_md(m, d, th):
    u = np.exp(-1j * th)
    tot = (1.0 + u) ** m
    for t in (1, 2, 3):
        if t > m:
            break
        tot += math.comb(m, t) * u ** t * (phi(t, th) ** d - 1.0)
    return tot


def g_direct(m, d, th, nblocks=1):
    """КОНТРОЛЬ 1: прямая сумма по подмножествам с явными строками."""
    a2 = m * nblocks
    rows = []
    for b in range(nblocks):
        mask = 0
        for j in range(m):
            mask |= 1 << (b * m + j)
        rows.extend([mask] * d)
    u = np.exp(-1j * th)
    tot = 0.0 + 0.0j
    for T in range(1 << a2):
        ph = 0.0
        for s in rows:
            k = bin(s & T).count('1')
            if 1 <= k <= 3:
                ph += WD[k]
        tot += u ** bin(T).count('1') * np.exp(1j * th * ph)
    return tot


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    th = K4 / 0.55
    print('КОНТРОЛЬ 1: g_d(m)^{блоков} против прямой суммы')
    worst = 0.0
    for (m, d, nb) in ((1, 1, 3), (2, 1, 3), (3, 2, 2), (4, 3, 2),
                       (1, 2, 4), (5, 1, 2), (2, 3, 3)):
        a = g_md(m, d, th) ** nb
        b = g_direct(m, d, th, nb)
        rel = abs(a - b) / max(abs(b), 1e-12)
        worst = max(worst, rel)
        print(f'  m={m} d={d} блоков={nb}: |G| формула {abs(a):.8f}, '
              f'прямо {abs(b):.8f}, отн. {rel:.2e}')
    print(f'  худшее {worst:.2e}')
    print()
    print('КОНТРОЛЬ 2: m=1,d=1 -> |g| = 2|cos 4th|')
    for hb in (0.45, 0.55, 0.62, 0.72):
        t = K4 / hb
        print(f'  hbar={hb}: |g_1(1)| = {abs(g_md(1,1,t)):.8f}, '
              f'2|cos 4th| = {2*abs(math.cos(4*t)):.8f}')
    print()
    print('КОНТРОЛЬ 3: точный максимум по ВСЕМ композициям внутри семейства?')
    from cancelfast import comps, g_vec
    from cancelclosed import g_closed
    for a2 in (6, 8, 10, 12):
        nc = comps(a2)
        g = np.abs(g_vec(nc, th))
        i = int(np.argmax(g))
        fam = abs(g_md(a2 // 2, 1, th)) ** 2
        print(f'  a1=2, a2={a2:>3}: max по композициям {g[i]:.6f} при '
              f'{tuple(int(x) for x in nc[:,i])}, семейство (m={a2//2},d=1) '
              f'{fam:.6f}')
    # a1 = 3: полный перебор композиций на 8 классов
    for a2 in (6, 9, 12):
        best, arg = -1.0, None
        for comb in itertools.combinations(range(a2 + 7), 7):
            prev, parts = -1, []
            for c in comb:
                parts.append(c - prev - 1)
                prev = c
            parts.append(a2 + 6 - prev)
            v = abs(g_closed(3, {c: parts[c] for c in range(8)}, th))
            if v > best:
                best, arg = v, tuple(parts)
        fam = abs(g_md(a2 // 3, 1, th)) ** 3
        print(f'  a1=3, a2={a2:>3}: max по композициям {best:.6f} при '
              f'{arg}, семейство (m={a2//3},d=1) {fam:.6f}')
    print()
    print('=' * 74)
    print('ГЛАВНОЕ: есть ли (m,d) с |g_d(m)| > 1, и при каком hbar его нет')
    print(f"{'hbar':>7}{'max |g|':>11}{'при (m,d)':>12}"
          f"{'c = ln|g|/(4(m+d))':>21}{'вердикт':>22}")
    best_hb = None
    rows = []
    for hb in np.arange(0.39, 0.785, 0.005):
        t = K4 / hb
        bg, barg, bc = 0.0, None, -1e9
        for m in range(1, 41):
            for d in range(1, 41):
                v = abs(g_md(m, d, t))
                c = math.log(max(v, 1e-300)) / (4 * (m + d))
                if c > bc:
                    bc, bg, barg = c, v, (m, d)
        rows.append((hb, bg, barg, bc))
    for (hb, bg, barg, bc) in rows:
        verdict = ('ГИБНЕТ' if bc <= 0 else 'выживает')
        print(f'{hb:>7.3f}{bg:>11.5f}{str(barg):>12}{bc:>21.6f}'
              f'{verdict:>22}')
    good = [r for r in rows if r[3] <= 0]
    print()
    if good:
        print(f'ЕСТЬ hbar, при которых сектор гибнет: {len(good)} из '
              f'{len(rows)} точек скана')
        for r in good:
            print(f'   hbar = {r[0]:.3f}, max|g| = {r[1]:.6f} при {r[2]}')
    else:
        print('НИ ПРИ ОДНОМ hbar в полосе max|g| не опускается до единицы:')
        mn = min(rows, key=lambda r: r[3])
        print(f'  лучшее hbar = {mn[0]:.3f}: c = {mn[3]:.6f} > 0 при '
              f'(m,d) = {mn[2]}, max|g| = {mn[1]:.6f}')
        print('  значит у слоистого сектора остаётся положительный вклад')
        print('  порядка N², и он побивает пространство-время счётом.')


if __name__ == '__main__':
    main()
