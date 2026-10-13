#!/usr/bin/env python3
"""РАЗДЕЛЕНИЕ ДВУХ СЛАГАЕМЫХ: его идея, что ньютоновский закон есть сумма.

ОТКУДА. `sharpen.py` показал, что показатель спадания дефицита хода часов
не −1, а около −0.53 (11 сигм от ньютоновского). Его замечание: «может,
нам перехода не хватает до ньютоновского, либо ньютоновский — это сумма
двух переменных?»

У нас буквально сумма. Ход часов есть ОТНОШЕНИЕ

    ход = h / tau  =  собственное время / радарное координатное,

и вещество портит ОБА. Разрежение замедляет часы (числитель `h`, высота
по фону) и одновременно искажает саму линейку (знаменатель `tau`,
радарное время по цепи), потому что цепь-наблюдатель проходит через центр
разрежения. В логарифмах это сумма, а в показателях — разность:

    d ln(h/tau) = d ln h − d ln tau

Значит до сих пор мерилась РАЗНОСТЬ ДВУХ ЗАКОНОВ, выданная за один. Если
у слагаемых разные показатели, их разность даёт третий, ни на что не
похожий, — ровно то, что наблюдалось.

ЧТО СЧИТАЕТСЯ. По тем же полосам радарного расстояния, в тех же трёх
миров на засев (без дырки / с дыркой / рассыпанный контроль), берутся
ПОРОЗНЬ относительные дефициты числителя и знаменателя:

    dh   = (<h>_дырка   − <h>_рассып)   / <h>_0
    dtau = (<tau>_дырка − <tau>_рассып) / <tau>_0
    dr   = (<h/tau>_дырка − <h/tau>_рассып) / <h/tau>_0

Проверка замкнутости: `dr` обязан равняться `dh − dtau` с точностью до
второго порядка. Если не равняется — разложение неверно и дальше идти
нельзя.

ЧТО ЭТО РАЗЛИЧАЕТ. Если ньютоновское `1/r` сидит в ОДНОМ из слагаемых, а
второе спадает иначе, то закон в порядке есть, а испорчен он был
смешиванием. Если оба слагаемых не ньютоновские — закон действительно не
ньютоновский, и это окончательно.

Окно и ошибка — как в `sharpen.py`: `r/ell >= 2.5` (размазка),
`r <= 0.40 R` (граница, по схлопыванию из `collapse.py`), нож по засевам.

  python3 split.py --n 60000 --radell 1.21 --seeds 12
"""
import argparse
import math
import time

import numpy as np

from bigfall import one


def bandmeans(D, bands, dens):
    """<h>, <tau> и <h/tau> по полосам радарного расстояния."""
    H, TT, R = [], [], []
    for lo, hi in bands:
        m = D['ok'] & (D['rho'] >= lo * dens) & (D['rho'] < hi * dens)
        if m.sum() < 15:
            H.append(np.nan); TT.append(np.nan); R.append(np.nan)
            continue
        h = D['h'][m]
        t = np.maximum(D['tau'][m], 1e-9)
        H.append(float(h.mean()))
        TT.append(float(t.mean()))
        R.append(float((h / t).mean()))
    return np.array(H), np.array(TT), np.array(R)


def fit(r, y):
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return np.nan
    return -np.polyfit(np.log(r[m]), np.log(y[m]), 1)[0]


def jack(r, M, win):
    """Показатель и ошибка складным ножом по засевам."""
    full = fit(r[win], -np.nanmean(M, axis=0)[win])
    jk = []
    for s in range(len(M)):
        keep = [i for i in range(len(M)) if i != s]
        b = fit(r[win], -np.nanmean(M[keep], axis=0)[win])
        if np.isfinite(b):
            jk.append(b)
    jk = np.array(jk)
    n = len(jk)
    se = (math.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum())
          if n > 2 else np.nan)
    return full, se


def main():
    ap = argparse.ArgumentParser(description='Разделение двух слагаемых')
    ap.add_argument('--n', type=int, default=60000)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--radell', type=float, default=1.21)
    ap.add_argument('--dens', type=int, default=300)
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--step', type=float, default=0.5)
    ap.add_argument('--blk', type=int, default=1500)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    rad = a.radell * ell
    Rw = a.t / 2.0
    nb = int(max(3, math.floor((0.80 * Rw - rad) / (a.step * ell))))
    bands = [(rad + k * a.step * ell, rad + (k + 1) * a.step * ell)
             for k in range(nb)]
    rc = np.array([0.5 * (lo + hi) for lo, hi in bands])
    win = (rc / ell >= 2.5) & (rc <= 0.40 * Rw)
    print(f'N = {a.n}, ell = {ell:.5f}, радиус мира {Rw/ell:.1f} ell')
    print(f'дырка {rad:.4f} = {a.radell} ell, полос {nb}, засевов {a.seeds}')
    print(f'окно (задано заранее): r/ell от {rc[win].min()/ell:.2f} '
          f'до {rc[win].max()/ell:.2f}, {win.sum()} полос')
    print()
    print('ход часов = h / tau, и вещество портит ОБА:')
    print('  h   — высота по фону, собственное время (часы замедляются)')
    print('  tau — радарное время по цепи, линейка (она тоже искажена)')
    print('d ln(h/tau) = d ln h − d ln tau  =>  мерилась РАЗНОСТЬ законов')
    print()

    DH, DT, DR = [], [], []
    for s in range(a.seeds):
        g = lambda: np.random.default_rng(95000 + 197 * s)
        A = one(a.n, a.t, g(), 0.0, a.dens, blk=a.blk)
        B = one(a.n, a.t, g(), rad, a.dens, blk=a.blk)
        S = one(a.n, a.t, g(), rad, a.dens, scatter=True, blk=a.blk)
        hA, tA, rA = bandmeans(A, bands, a.dens)
        hB, tB, rB = bandmeans(B, bands, a.dens)
        hS, tS, rS = bandmeans(S, bands, a.dens)
        DH.append((hB - hS) / hA)
        DT.append((tB - tS) / tA)
        DR.append((rB - rS) / rA)
        del A, B, S
        if s == 0 or (s + 1) % 4 == 0:
            print(f'  засев {s} ({time.time()-t0:.0f} с)', flush=True)
    DH = np.array(DH); DT = np.array(DT); DR = np.array(DR)

    mh = np.nanmean(DH, axis=0); mt = np.nanmean(DT, axis=0)
    mr = np.nanmean(DR, axis=0)
    sh = np.nanstd(DH, axis=0, ddof=1) / math.sqrt(len(DH))
    st = np.nanstd(DT, axis=0, ddof=1) / math.sqrt(len(DT))
    print()
    print('ПРОВЕРКА ЗАМКНУТОСТИ: dr обязан равняться dh − dtau')
    print(f"{'r/ell':>7}{'dh (часы)':>12}{'сигм':>7}"
          f"{'dtau (линейка)':>16}{'сигм':>7}{'dr':>11}{'dh-dtau':>11}"
          f"{'невязка':>10}")
    for k in range(nb):
        if not np.isfinite(mr[k]):
            continue
        res = mr[k] - (mh[k] - mt[k])
        print(f'{rc[k]/ell:>7.2f}{mh[k]:>+12.5f}'
              f'{abs(mh[k])/max(sh[k],1e-12):>7.1f}'
              f'{mt[k]:>+16.5f}{abs(mt[k])/max(st[k],1e-12):>7.1f}'
              f'{mr[k]:>+11.5f}{mh[k]-mt[k]:>+11.5f}{res:>+10.5f}')
    good = np.isfinite(mr)
    rel = np.nanmean(np.abs(mr[good] - (mh[good] - mt[good]))) / \
        max(np.nanmean(np.abs(mr[good])), 1e-12)
    print(f'  средняя невязка: {rel*100:.1f} % от величины — '
          f'{"разложение замкнуто" if rel < 0.15 else "РАЗЛОЖЕНИЕ НЕ СХОДИТСЯ"}')

    print()
    print('ПОКАЗАТЕЛИ ПОРОЗНЬ, в заранее заданном окне, нож по засевам')
    if win.sum() >= 3:
        for lbl, M in (('ЧАСЫ      dh  ', DH),
                       ('ЛИНЕЙКА   dtau', DT),
                       ('ОТНОШЕНИЕ dr  ', DR)):
            b, se = jack(rc, M, win)
            if not np.isfinite(b):
                print(f'  {lbl} знак не позволяет подгонку степени')
                continue
            d = (f'{abs(-b + 1)/se:.1f} сигм от -1'
                 if np.isfinite(se) and se > 0 else '')
            print(f'  {lbl} r^{-b:+.3f} +- {se:.3f}   {d}')
    print()
    print('МЕСТНЫЙ ПОКАЗАТЕЛЬ порознь (окно из трёх полос)')
    print(f"{'r/ell':>7}{'r/R':>7}{'часы':>9}{'линейка':>10}{'отнош':>9}")
    for k in range(1, nb - 1):
        w = slice(k - 1, k + 2)
        bh, bt, br = fit(rc[w], -mh[w]), fit(rc[w], -mt[w]), fit(rc[w], -mr[w])
        print(f'{rc[k]/ell:>7.2f}{rc[k]/Rw:>7.3f}{-bh:>9.3f}'
              f'{-bt:>10.3f}{-br:>9.3f}')
    print()
    print('  если -1 сидит в ОДНОМ слагаемом — закон есть, он был смешан;')
    print('  если ни в одном — закон не ньютоновский, и это окончательно.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
