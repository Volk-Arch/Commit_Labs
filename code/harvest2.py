#!/usr/bin/env python3
"""НАБОР С ДВУМЕРНОЙ КВОТОЙ: по наклону И по размеру интервала.

ЗАЧЕМ. Вырождение плана ломается размахом `C_1` (арифметика плана:
ошибка отношения 0.172 при `C_1 = 200..900`, 0.021 при `50..2000`, то
есть восьмикратный выигрыш). Но пробный набор в широком окне показал,
что ОДНИХ квот по наклону недостаточно: средние `C_1` по полосам вышли
139, 103, 827, 890, 486, 272 — отбор берёт что легче, диапазон внутри
полосы не покрыт, и ошибки выросли до ±0.03, потому что сигнал идёт как
`sqrt(C_1)` и в одной полосе смешались интервалы с `C_1` = 50 и 2000.

ЧТО ЗДЕСЬ. Квота по сетке `(полоса v², полоса C_1)`. Плотность в этом
сприклинге единичная, поэтому `C_1 ~ (pi/24) tau^4`, а `tau^2` известна
ДО дорогого шага — значит полосу по размеру можно выбрать заранее, не
считая мощность интервала.

Формат строки прежний `(C_1, X_2, X_3, v^2, mid)`, чтобы разбор работал
без правок.

  python3 harvest2.py --fam lin --seeds 5 --seed0 900
"""
import argparse
import math
import pickle
import time

import numpy as np

from pencil import (SCRATCH, VB, sprinkle, fits, counts, truth,
                    C2C, C3C, a_of)

CFGW = (26.0, 20.0, 10.0, 1.6, 50, 2000)
# ЗАЩИТА ОКНА ПРИМЕНИМОСТИ, добавлена 8 октября. Разложение первого
# порядка держится до T^2 R_uu ~ 0.8. В широком окне T^2 растёт как
# sqrt(C_1), поэтому большие интервалы при высоком наклоне уходят за
# окно: при C_1 = 1153 и v^2 = 0.74 выходило 1.65, и предсказание там
# промахивалось (-0.0332 против измеренных -0.0232 +- 0.0061).
# Теперь такие интервалы отбрасываются ПРИ НАБОРЕ, и область
# применимости вырезается сама.
TMAX = 0.8
C1B = (50, 130, 330, 830, 2000)          # полосы по C_1, геометрически
# ПАРАМЕТРЫ ДЛЯ ШИРОКОГО ОКНА, 8 октября. Подобраны так, чтобы ВСЯ
# сетка (C_1 до 2000, v² до 0.78) лежала внутри окна применимости
# T^2 R_uu <= 0.8. При прежних значениях две верхние полосы по размеру
# были пусты В ПРИНЦИПЕ, и набор бесконечно пытался их заполнить.
# Сигнал падает как параметр в квадрате, зато рычаг по C_1 даёт 8 раз.
PARS = {'exp': 0.030, 'root': 0.030, 'lin': 0.020, 'quad': 0.0007,
        'flat': 0.0}


def ruu(fam, par, eta0, mid, v2):
    """R_uu * a^2 для направления интервала — для защиты окна."""
    if fam == 'flat':
        return 0.0
    a = float(a_of(np.array([mid]), fam, par, eta0)[0])
    if fam == 'lin':
        ree, rxx = 3 * par ** 2 / a ** 2, par ** 2 / a ** 2
    elif fam == 'exp':
        ree, rxx = 0.0, 2 * par ** 2
    elif fam == 'root':
        ree, rxx = 1.5 * par ** 2 / a ** 4, 0.0
    else:
        sq = par * (mid - eta0) ** 2
        ree = 6 * par * (sq - 2) / (sq + 2) ** 2
        rxx = 2 * par * (3 * sq + 2) / (sq + 2) ** 2
    return abs((ree + rxx * v2) / (1.0 - v2))


def harvest2(fam, par, cfg, rs, want, tries=40):
    L, T, eta0, half, lo, hi = cfg
    X = sprinkle(fam, par, L, T, eta0, rs)
    t = np.ascontiguousarray(X[:, 0])
    S = np.ascontiguousarray(X[:, 1:])
    tlo = (24.0 * lo / math.pi) ** 0.25
    thi = (24.0 * hi / math.pi) ** 0.25
    nb, nc = len(VB) - 1, len(C1B) - 1
    quota = want // (nb * nc) + 1
    got = np.zeros((nb, nc), dtype=int)
    out = []
    guard = 0
    while len(out) < want and guard < want * tries:
        guard += 1
        ip = int(rs.integers(np.searchsorted(t, eta0 - half - thi),
                             np.searchsorted(t, eta0 + half)))
        p0, ps = t[ip], S[ip]
        j0 = np.searchsorted(t, p0 + tlo)
        j1 = np.searchsorted(t, p0 + thi * 2.2)
        if j1 - j0 < 40:
            continue
        de = t[j0:j1] - p0
        d = S[j0:j1] - ps
        dx2 = np.einsum('ij,ij->i', d, d)
        tau2 = de * de - dx2
        ok = np.flatnonzero((tau2 > tlo * tlo) & (tau2 < thi * thi))
        if len(ok) == 0:
            continue
        v2 = dx2[ok] / (de[ok] ** 2)
        c1p = (math.pi / 24.0) * tau2[ok] ** 2          # ожидаемое C_1
        bi = np.digitize(v2, VB) - 1
        ci = np.digitize(c1p, C1B) - 1
        fill = np.array([got[b, c] if (0 <= b < nb and 0 <= c < nc)
                         else 10 ** 9 for b, c in zip(bi, ci)])
        for q in np.argsort(fill)[:10]:
            if len(out) >= want:
                break
            b, c = bi[q], ci[q]
            if not (0 <= b < nb and 0 <= c < nc) or got[b, c] >= quota:
                continue
            jj = int(ok[q])
            f0, fs = t[j0 + jj], S[j0 + jj]
            mid = 0.5 * (p0 + f0)
            if abs(mid - eta0) > half:
                continue
            # окно применимости: T^2 R_uu <= TMAX
            if tau2[ok[q]] * ruu(fam, par, eta0, mid,
                                 float(v2[q])) > TMAX:
                continue
            if not fits(np.concatenate([[p0], ps]),
                        np.concatenate([[f0], fs]), L, T):
                continue
            k0 = np.searchsorted(t, p0, 'right')
            k1 = np.searchsorted(t, f0, 'left')
            if k1 - k0 < lo:
                continue
            Sl, tl = S[k0:k1], t[k0:k1]
            u = Sl - ps
            w = fs - Sl
            m = (((tl - p0) ** 2 > np.einsum('ij,ij->i', u, u))
                 & ((f0 - tl) ** 2 > np.einsum('ij,ij->i', w, w)))
            a = int(m.sum())
            if not (lo <= a <= hi):
                continue
            _, bb, cc = counts(np.column_stack([tl[m], Sl[m]]))
            out.append((a,
                        bb / (C2C * a * (a - 1.0)) - 1.0,
                        cc / (C3C * a * (a - 1.0) * (a - 2.0)) - 1.0,
                        float(v2[q]), mid))
            got[b, c] += 1
    del X, t, S
    return out, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fam', default='lin',
                    choices=('exp', 'root', 'lin', 'quad', 'flat'))
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--per', type=int, default=480)
    a = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    par = PARS[a.fam]
    print(f'семейство {a.fam}, параметр {par}, окно C_1 = '
          f'{CFGW[4]}..{CFGW[5]}, полосы C_1 {C1B}')
    print(f'истина (R_ee, R_xx, R) = {truth(a.fam, par)}')
    rows, sdi = [], []
    tot = np.zeros((len(VB) - 1, len(C1B) - 1), dtype=int)
    for s in range(a.seed0, a.seed0 + a.seeds):
        rs = np.random.default_rng(770000 + 1931 * s + int(1e7 * par))
        r, got = harvest2(a.fam, par, CFGW, rs, a.per)
        rows += r
        sdi += [s] * len(r)
        tot += got
        if (s - a.seed0 + 1) % 5 == 0:
            print(f'  засевов {s-a.seed0+1}, интервалов {len(rows)}, '
                  f'{time.time()-t0:.0f} с', flush=True)
    A = np.array(rows)
    out = f'{SCRATCH}/{a.fam}fam2_{a.seed0}.pkl'
    with open(out, 'wb') as fh:
        pickle.dump({'A': A, 'sd': np.array(sdi), 'k': par, 'fam': a.fam,
                     'cfg': CFGW, 'c1b': C1B}, fh)
    print()
    print(f'{len(A)} интервалов. ЗАПОЛНЕНИЕ СЕТКИ (строки v², столбцы C_1):')
    for i in range(len(VB) - 1):
        print('   ' + ' '.join(f'{tot[i, j]:>5}' for j in range(len(C1B)-1)))
    print()
    print(f"{'v²':>7}{'полоса C_1':>13}{'шт':>6}{'⟨C_1⟩':>8}"
          f"{'T²R_uu':>9}{'X_2':>10}{'ст.ош':>9}")
    sd = np.array(sdi)
    for i in range(len(VB) - 1):
        for j in range(len(C1B) - 1):
            m = ((A[:, 3] >= VB[i]) & (A[:, 3] < VB[i+1])
                 & (A[:, 0] >= C1B[j]) & (A[:, 0] < C1B[j+1]))
            if m.sum() < 15:
                continue
            us = np.unique(sd[m])
            if len(us) < 3:
                continue
            j2 = np.array([A[m & (sd != u), 1].mean() for u in us])
            n = len(us)
            e2 = math.sqrt((n-1)/n*((j2-j2.mean())**2).sum())
            tm = np.mean([math.sqrt(24.0*c/math.pi)
                          * ruu(a.fam, par, CFGW[2], md, vv)
                          for c, vv, md in zip(A[m,0], A[m,3], A[m,4])])
            print(f'{A[m,3].mean():>7.3f}{f"{C1B[j]}..{C1B[j+1]}":>13}'
                  f'{m.sum():>6}{A[m,0].mean():>8.0f}{tm:>9.2f}'
                  f'{A[m,1].mean():>+10.4f}{e2:>9.4f}')
    print(f'\n  сохранено: {out}, время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
