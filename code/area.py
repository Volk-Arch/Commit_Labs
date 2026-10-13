#!/usr/bin/env python3
"""ПОРЯДКОВАЯ ПЛОЩАДЬ: идёт ли граничный счёт как площадь, то есть sqrt(C_1)?

ЗАЧЕМ. Якобсон и Виссер (SciPost Phys. 7, 079 (2019)): уравнение
Эйнштейна выполнено тогда и только тогда, когда обобщённая энтропия
стационарна ПРИ ФИКСИРОВАННОМ ОБЪЁМЕ в малых ромбах. В простой форме:
недостача площади границы при фиксированном объёме внутри равна энергии
внутри — как круг на апельсине короче круга той же площади на бумаге.

У интервала есть ДВА независимых счёта: объём `C_1` и граничный счёт.
Если недостача второго при фиксированном первом совпадёт с плотностью
энергии, добытой из обилий `C_2`, `C_3`, то уравнение поля окажется
условием согласования двух счётов одного интервала.

ЗДЕСЬ — только первый шаг: КАЛИБРОВКА ПЛОСКОГО. Для ромба собственного
времени `tau` объём есть `(pi/24)tau⁴`, а «талия» — двумерная сфера
радиуса `tau/2` площадью `pi tau²`. Значит в плоском

    площадь ~ tau²  ~  sqrt(C_1),   а НЕ C_1

и показатель степени решает, площадь ли это вообще. Проверяются четыре
граничные величины, все из уже считаемых массивов прошлого и будущего:

    F_0  число МАКСИМАЛЬНЫХ элементов (в будущем внутри нет ничего)
    P_0  число МИНИМАЛЬНЫХ
    F_0+F_1+F_2 и то же для прошлого — более широкий слой у конуса

ОЖИДАНИЕ. Элемент максимален с вероятностью exp(-rho V(x,f)), поэтому
счёт сосредоточен у светового конуса, и это ровно та конструкция, из
которой Бак, Доукер, Джабб и Сурья (1502.05388) получают объём стыка
коразмерности 2 — в четырёхмерии площадь.

КОНТРОЛЬ: показатель мерить по ЛЕСТНИЦЕ по размеру, а не подгонкой по
всему диапазону (правило из ARTICLE.md), и ошибку брать складным ножом
по ЗАСЕВАМ, а не по интервалам.

  python3 area.py [--seeds N] [--per M]
"""
import argparse
import math
import time

import numpy as np

from chains import c_k, fits
from pencil import sprinkle, VB

C2C, C3C = c_k(2), c_k(3)


def bcounts(I):
    """C_1, C_2, C_3 и граничные счёты F_0, P_0, F_012, P_012."""
    k = len(I)
    if k < 4:
        return None
    dt = I[:, 0][None, :] - I[:, 0][:, None]
    dx = I[:, 1:][None, :, :] - I[:, 1:][:, None, :]
    M = (dt > 0) & (dt ** 2 > (dx * dx).sum(2))
    past = M.sum(0)
    fut = M.sum(1)
    return (k, int(M.sum()), int((past * fut).sum()),
            int((fut == 0).sum()), int((past == 0).sum()),
            int((fut <= 2).sum()), int((past <= 2).sum()))


def harvest(fam, par, eta0, cfg, rs, want):
    W, L, lo_c, hi_c, band = cfg
    X = sprinkle(fam, par, L, 2 * eta0, eta0, rs)
    t = np.ascontiguousarray(X[:, 0])
    S = np.ascontiguousarray(X[:, 1:])
    tlo = (24.0 * lo_c / math.pi) ** 0.25
    thi = (24.0 * hi_c / math.pi) ** 0.25
    out = []
    guard = 0
    while len(out) < want and guard < want * 30:
        guard += 1
        ip = int(rs.integers(np.searchsorted(t, eta0 - band - thi * 1.4),
                             np.searchsorted(t, eta0 + band)))
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
        for q in rs.permutation(ok)[:6]:
            if len(out) >= want:
                break
            jj = int(q)
            f0, fs = t[j0 + jj], S[j0 + jj]
            if abs(0.5 * (p0 + f0) - eta0) > band:
                continue
            if not fits(np.concatenate([[p0], ps]),
                        np.concatenate([[f0], fs]), L, 2 * eta0):
                continue
            k0 = np.searchsorted(t, p0, 'right')
            k1 = np.searchsorted(t, f0, 'left')
            if k1 - k0 < lo_c:
                continue
            Sl, tl = S[k0:k1], t[k0:k1]
            u = Sl - ps
            w = fs - Sl
            m = (((tl - p0) ** 2 > np.einsum('ij,ij->i', u, u))
                 & ((f0 - tl) ** 2 > np.einsum('ij,ij->i', w, w)))
            a = int(m.sum())
            if not (lo_c <= a <= hi_c):
                continue
            r = bcounts(np.column_stack([tl[m], Sl[m]]))
            if r is None:
                continue
            # истинное tau из вложения — ТОЛЬКО для сверки, не для оценки
            tau = math.sqrt(max(tau2[jj], 1e-12))
            out.append(r + (tau,))
    del X, t, S
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=14)
    ap.add_argument('--per', type=int, default=500)
    a_ = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    # широкий диапазон размеров: показатель мерим лестницей
    cfg = (0.0, 26.0, 120, 2400, 4.0)
    rows, sdi = [], []
    for s in range(a_.seeds):
        rs = np.random.default_rng(991000 + 7717 * s)
        r = harvest('flat', 0.0, 10.0, cfg, rs, a_.per)
        rows += r
        sdi += [s] * len(r)
    A = np.array(rows, dtype=float)
    sd = np.array(sdi)
    print(f'плоское: {len(A)} интервалов, C_1 от {A[:,0].min():.0f} '
          f'до {A[:,0].max():.0f}   ({time.time()-t0:.0f} с)')
    print()
    names = ('F_0 максимальных', 'P_0 минимальных',
             'F_0+F_1+F_2', 'P_0+P_1+P_2')
    cols = (3, 4, 5, 6)
    print('ПОКАЗАТЕЛЬ СТЕПЕНИ: величина ~ C_1^p.  p = 1/2 -> ПЛОЩАДЬ,')
    print('p = 1 -> объём, p = 3/4 -> трёхмерный объём нулевой границы.')
    print()
    q = np.quantile(A[:, 0], np.linspace(0, 1, 6))
    print(f"{'C_1':>8}{'шт':>7}" + ''.join(f'{n:>18}' for n in names))
    bands = []
    for i in range(5):
        m = (A[:, 0] >= q[i]) & (A[:, 0] <= q[i + 1])
        if m.sum() < 30:
            continue
        bands.append((A[m][:, 0].mean(), m,
                      [A[m][:, c].mean() for c in cols]))
        print(f'{A[m][:,0].mean():>8.0f}{m.sum():>7}'
              + ''.join(f'{v:>18.1f}' for v in bands[-1][2]))
    print()
    print(f"{'величина':>18}{'показатель p':>15}{'ст.ош':>9}{'вердикт':>14}")
    for j, (nm, c) in enumerate(zip(names, cols)):
        def slope(mask):
            x = np.log(A[mask][:, 0])
            y = np.log(np.maximum(A[mask][:, c], 0.5))
            return np.polyfit(x, y, 1)[0]
        full = slope(np.ones(len(A), bool))
        us = np.unique(sd)
        jk = np.array([slope(sd != u) for u in us])
        k = len(us)
        e = math.sqrt((k - 1) / k * ((jk - jk.mean()) ** 2).sum())
        ver = ('ПЛОЩАДЬ' if abs(full - 0.5) < 3 * max(e, 1e-6)
               else ('объём' if abs(full - 1.0) < 3 * max(e, 1e-6)
                     else 'ни то ни то'))
        print(f'{nm:>18}{full:>15.4f}{e:>9.4f}{ver:>14}')
    print()
    # прямая сверка с tau²: площадь талии = pi tau²
    print('ПРЯМАЯ СВЕРКА с площадью талии pi*tau² (tau из вложения,')
    print('только для проверки — в оценке он не участвует):')
    print(f"{'C_1':>8}{'pi tau²':>10}" + ''.join(f'{n[:9]:>11}' for n in names)
          + ''.join(f'{"/pi tau²":>11}' for _ in names))
    for c1, m, vals in bands:
        at = math.pi * (A[m][:, 7] ** 2).mean()
        print(f'{c1:>8.0f}{at:>10.1f}'
              + ''.join(f'{v:>11.1f}' for v in vals)
              + ''.join(f'{v/at:>11.4f}' for v in vals))
    print()
    print('ЧТЕНИЕ. Если показатель равен 1/2 и отношение к pi*tau²')
    print('ПОСТОЯННО по полосам — граничный счёт есть площадь в единицах')
    print('шага дискретности, и первый ингредиент дороги Якобсона у нас')
    print('в руках. Если показатель 1 — это объём, и величина не та.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
