"""Устойчив ли рост: по N, по eps, по размеру пула, по силе отбора."""
import math
import numpy as np
from core import mm_dim, height, spectral, transitivity_violations, frac_pairs
from growth import grow

def run(N, beta, eps, K, seeds=3):
    out = []
    for t in range(seeds):
        rs = np.random.default_rng(4400+17*t)
        R, h = grow(N, beta, eps, K, rs)
        f = frac_pairs(R)
        out.append((f, mm_dim(f), height(R)[0], spectral(R),
                    transitivity_violations(R), h[:,0].mean()))
        del R
    A = np.array(out, float)
    return A.mean(0), A.std(0)/math.sqrt(seeds)

def show(lbl, m, e):
    print(f'{lbl:>16}{m[0]:>9.4f} ±{e[0]:<6.4f}{m[1]:>8.3f} ±{e[1]:<5.3f}'
          f'{m[2]:>7.1f}{m[3]:>+10.4f}{int(m[4]):>8}{m[5]:>11.1f}')

hdr = (f"{'':>16}{'доля пар':>16}{'размерность':>14}{'высота':>7}"
       f"{'спектр':>10}{'наруш':>8}{'|прошлое|':>11}")
print('ЦЕЛЬ: доля 0.103, размерность 4.00, спектр > 0, нарушений 0')
print('высота Минковского растёт как N^(1/4): ~7 при 300, ~9 при 800, ~11 при 1600')
print()
print('A. УСТОЙЧИВОСТЬ ПО РАЗМЕРУ МИРА (eps=0.2, beta=100, K=24)'); print(hdr)
for N in (300, 800, 1600):
    m,e = run(N, 100.0, 0.2, 24); show(f'N={N}', m, e)
print()
print('B. ЗАВИСИМОСТЬ ОТ eps (N=800, beta=100, K=24)'); print(hdr)
for eps in (0.1, 0.2, 0.4, 0.8):
    m,e = run(800, 100.0, eps, 24); show(f'eps={eps}', m, e)
print()
print('C. АРТЕФАКТ ПУЛА? (N=800, eps=0.2, beta=100)'); print(hdr)
for K in (6, 12, 24, 48, 96):
    m,e = run(800, 100.0, 0.2, K); show(f'K={K}', m, e)
print()
print('D. МЯГКИЙ ОТБОР ВМЕСТО ЖАДНОГО (N=800, eps=0.2, K=24)'); print(hdr)
for b in (3.0, 10.0, 30.0, 100.0):
    m,e = run(800, b, 0.2, 24); show(f'beta={b:g}', m, e)
