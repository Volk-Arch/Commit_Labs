#!/usr/bin/env python3
"""ЗОЛОТОЙ КВАЗИКРИСТАЛЛ 3+1 — мир из одной строки.

Точка мира — четвёрка чисел c_i = a_i + b_i τ (a_i, b_i целые), τ = (1+√5)/2.
У каждого такого числа есть «тень» c' = a − b/τ (замена τ -> τ' = −1/τ).

    ПРАВИЛО: четвёрка входит в мир  ⟺  её тень лежит в шаре радиуса r
             0.618·t'² + x'² + y'² + z'²  ≤  r²

Физические координаты: T = √τ·t, x, y, z;  причинность: ΔT > |Δx|.
Форма τt² − x² − y² − z² в нашем мире лоренцева, а у тени она
определённая (τ' < 0): симметрии формы с целыми (в Z[τ]) коэффициентами
действуют на тень вращениями и потому шар не портят. Это и есть
дискретная группа Лоренца, приближающая любой буст (кокомпактная).
Плотность: ρ = объём шара / 25 (кообъём Z[τ]⁴ равен (√5)⁴).
"""
import math, sys, time
import numpy as np
TAU = (1 + 5 ** .5) / 2; TP = -1 / TAU
ST, SI = math.sqrt(TAU), math.sqrt(-TP)          # масштабы времени: снаружи и в тени

def coord_values(lo, hi, rint):
    """все c = a + bτ с lo ≤ c ≤ hi и |c'| ≤ rint. Возвращает (c, c')."""
    # a = (c τ' ... ): решаем относительно b: c − c' = b(τ − τ') = b√5
    bmin = math.floor((lo - rint) / 5 ** .5) - 1; bmax = math.ceil((hi + rint) / 5 ** .5) + 1
    C, Cp = [], []
    for b in range(bmin, bmax + 1):
        alo = max(math.ceil(lo - b * TAU), math.ceil(-rint - b * TP))
        ahi = min(math.floor(hi - b * TAU), math.floor(rint - b * TP))
        if ahi < alo: continue
        a = np.arange(alo, ahi + 1)
        C.append(a + b * TAU); Cp.append(a + b * TP)
    if not C: return np.empty(0), np.empty(0)
    return np.concatenate(C), np.concatenate(Cp)

def r_for_density(rho):
    return (25 * rho / (math.pi ** 2 / 2)) ** 0.25

def points_in_box(box, r):
    """точки мира в физическом ящике box = [(lo,hi) для T, x, y, z]."""
    (Tl, Th), *sp = box
    t, tp = coord_values(Tl / ST, Th / ST, r / SI)
    vals = [(t * ST, (tp * SI) ** 2)]
    for lo, hi in sp:
        c, cp = coord_values(lo, hi, r)
        vals.append((c, cp ** 2))
    # декартово произведение с отсечением по шару тени
    P = np.zeros((1, 4)); W = np.zeros(1)
    for k, (c, w2) in enumerate(vals):
        P = np.repeat(P, len(c), 0); W = np.repeat(W, len(c))
        P[:, k] = np.tile(c, len(P) // len(c)); W += np.tile(w2, len(W) // len(c))
        keep = W <= r * r; P, W = P[keep], W[keep]
    return P

def diamond(N, r=None):
    """мир в причинном ромбе |T| + |x| ≤ 1 с ожидаемым числом точек ~N."""
    rho = N / (2 * math.pi / 3)
    r = r_for_density(rho)
    P = points_in_box([(-1, 1)] * 4, r)
    m = np.abs(P[:, 0]) + np.linalg.norm(P[:, 1:], axis=1) <= 1
    return P[m]

if __name__ == '__main__':
    # контроль плотности
    for rho in (1.0, 3.0):
        r = r_for_density(rho); L = 4.0
        P = points_in_box([(-L, L)] * 4, r)
        print(f'плотность: заказано {rho}, получено {len(P) / (2 * L) ** 4:.4f}  (точек {len(P)})')
