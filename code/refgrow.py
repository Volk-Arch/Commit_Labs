#!/usr/bin/env python3
"""Рост по опорным событиям: k порядковых шкал вместо координат.

ПОВОД. step-phase-doc.md: все перебранные семейства предложений имеют по
одной эффективной степени свободы, а цель живёт в пяти измерениях
(плотность, высота, связи, спектр, перекрытие прошлых). Диагностика
достижимости показала прямо: |D| и перекрытие J лежат на одномерной
кривой, цель вне неё. Нужно семейство с несколькими степенями свободы на
событие — и порядковое, а не координатное.

КОНСТРУКЦИЯ. Опорные события r_1..r_k. Для каждого события b заводится

    v_i(b) = длина самой длинной цепи от r_i до b   (0, если недостижимо)

Новое событие задаётся целью w из Z^k, и его прошлое есть

    D = { b : v(b) <= w покомпонентно }

ДВА СВОЙСТВА БЕСПЛАТНО.

  Замкнутость вниз. Если b' < b, любая цепь от r_i до b' продолжается
  через b, значит v(b') <= v(b). Тогда из v(b) <= w следует v(b') <= w.
  Значит D — всегда корректное прошлое, и транзитивность цела.

  Коррелированность шкал. Все k чисел выведены из ОДНОГО порядка, как
  s_u = t - u.x выведены из одного 4-вектора. Именно коррелированность
  стоила +0.04 по спектру при совпадающей плотности, и именно её не
  давали ни популярность, ни высота.

И это буквально исходное правило проекта — «событие несёт k чисел, A
раньше B если раньше по всем k» — только числа берутся ИЗ ПОРЯДКА, а не
задаются извне.

  python3 refgrow.py
  python3 refgrow.py --ns 600 1200 2400 --krefs 2 3 4 6
"""
import argparse
import math
import time

import numpy as np

from core import (mm_dim, height, spectral, frac_pairs, links_of,
                  transitivity_violations)


def jacc_inc(R):
    """Среднее перекрытие прошлых у НЕСРАВНИМЫХ пар (Жаккар)."""
    A = R.astype(np.float32)
    inter = A @ A.T
    sz = A.sum(1)
    un = sz[:, None] + sz[None, :] - inter
    J = np.where(un > 0, inter / np.maximum(un, 1e-9), 0.0)
    rel = R | R.T
    iu = np.triu(np.ones_like(rel), 1).astype(bool)
    big = (sz[:, None] >= 5) & (sz[None, :] >= 5)
    m = iu & ~rel & big
    return float(J[m].mean()) if m.sum() > 50 else float('nan')


def grow_ref(N, kref, rs, spread=1.0, jitter=0.0):
    """Вырастить порядок, задавая каждому событию цель w из k чисел.

    spread — во сколько текущих максимумов шкал забрасывается цель;
    jitter — доля шкал, которым цель ставится заведомо большой
             (позволяет прошлому иногда захватывать всё по одной шкале).
    """
    R = np.zeros((N, N), bool)
    V = np.zeros((N, kref), np.int32)
    for i in range(kref):
        V[i, i] = 1                            # опора достижима из себя
    for k in range(kref, N):
        mx = V[:k].max(0).astype(np.float64) + 1.0
        w = rs.random(kref) * spread * mx
        if jitter > 0:
            big = rs.random(kref) < jitter
            w = np.where(big, mx * 1e3, w)
        d = (V[:k] <= w[None, :]).all(1)
        R[k, :k] = d
        if d.any():
            sub = V[:k][d]
            for i in range(kref):
                col = sub[:, i]
                pos = col[col > 0]
                V[k, i] = (pos.max() + 1) if len(pos) else 0
    return R


def report(R, lbl):
    f = frac_pairs(R)
    N = len(R)
    return dict(lbl=lbl, f=f, d=mm_dim(f), h=height(R)[0], s=spectral(R),
                tv=transitivity_violations(R), l=links_of(R).sum() / N,
                j=jacc_inc(R))


def main():
    ap = argparse.ArgumentParser(description='Рост по опорным событиям')
    ap.add_argument('--ns', type=int, nargs='+', default=[600])
    ap.add_argument('--krefs', type=int, nargs='+', default=[2, 3, 4, 5, 6, 8])
    ap.add_argument('--spreads', type=float, nargs='+', default=[1.0])
    ap.add_argument('--jitter', type=float, nargs='+', default=[0.0])
    ap.add_argument('--seeds', type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    print('ЦЕЛЬ (конус): доля 0.098, разм. 4.03, высота ~9 при N=800,')
    print('спектр +0.013 при с.к.о. 0.003, связей ~13, J несравн. 0.169')
    print('ящик для сравнения: спектр -0.008, J 0.138')
    print()
    print(f"{'N':>6}{'k опор':>8}{'spread':>8}{'jit':>6}{'доля':>9}"
          f"{'разм.':>7}{'высота':>7}{'<спектр>':>11}{'с.к.о.':>9}"
          f"{'наруш':>7}{'связ':>7}{'J':>8}")
    for N in a.ns:
        for kref in a.krefs:
            for sp in a.spreads:
                for jt in a.jitter:
                    rows = []
                    for t in range(a.seeds):
                        rs = np.random.default_rng(101000 + 977 * t)
                        R = grow_ref(N, kref, rs, sp, jt)
                        rows.append(report(R, ''))
                        del R
                    g = lambda key: np.array([r[key] for r in rows], float)
                    sp_v = g('s')
                    print(f'{N:>6}{kref:>8}{sp:>8.2f}{jt:>6.2f}'
                          f'{g("f").mean():>9.4f}{np.nanmean(g("d")):>7.2f}'
                          f'{g("h").mean():>7.1f}{sp_v.mean():>+11.4f}'
                          f'{sp_v.std():>9.4f}{int(g("tv").sum()):>7}'
                          f'{g("l").mean():>7.1f}'
                          f'{np.nanmean(g("j")):>8.4f}', flush=True)
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
