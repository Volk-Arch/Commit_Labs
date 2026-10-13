#!/usr/bin/env python3
"""Форма области решает всё: ящик, шар, причинный ромб.

На ящиках по осям Соболь даёт логарифмическое подавление флуктуаций, на
шарах — степенное (hyperuniform-doc.md). Разница в 88 порядков при
пересчёте на космологический объём, поэтому форму надо выбрать
правильно, а не удобно.

Вопрос распадается на два.

  наша модель     порядок покомпонентный, значит интервал между двумя
                  событиями — ЯЩИК по осям
  настоящий мир   причинный ромб есть пересечение двух световых
                  конусов; ящиком он является только в двух измерениях
                  (в светоконусных координатах), в четырёх — нет

Довод Соркина про Λ относится ко второму. Здесь меряются все три формы
рядом, при одной доле объёма и одних размерах сетки.

Объём причинного ромба собственного времени T в d измерениях:
    V = pi^((d-1)/2) / (2^(d-1) * d * Г((d+1)/2)) * T^d
при d=4 это (pi/24) T^4.

  python3 shapes.py --dim 4
  python3 shapes.py --dim 4 --frac 0.005 --maxm 18
"""
import argparse
import math
import time

import numpy as np
from scipy.stats import qmc


def diamond_volume_coeff(d):
    return math.pi ** ((d - 1) / 2) / (
        2 ** (d - 1) * d * math.gamma((d + 1) / 2))


def ball_volume_coeff(d):
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


# ---- три формы: каждая возвращает список областей и счётчик ----------

def make_boxes(d, frac, n_reg, rs):
    regs = []
    for _ in range(n_reg):
        e = rs.random(d)
        e /= e.sum()
        side = np.minimum(frac ** e, 1.0)
        lo = rs.random(d) * (1.0 - side)
        regs.append((lo, lo + side))
    return regs


def count_boxes(X, regs):
    return np.array([np.count_nonzero(((X >= lo) & (X < hi)).all(1))
                     for lo, hi in regs], dtype=np.int64)


def make_balls(d, frac, n_reg, rs):
    r = (frac / ball_volume_coeff(d)) ** (1.0 / d)
    span = max(1.0 - 2 * r, 0.0)
    return [(r + rs.random(d) * span, r) for _ in range(n_reg)]


def count_balls(X, regs):
    return np.array([np.count_nonzero(((X - c) ** 2).sum(1) < r * r)
                     for c, r in regs], dtype=np.int64)


def make_diamonds(d, frac, n_reg, rs):
    """Ромб задаётся центром и собственным временем T.

    Ось 0 — время. Точка внутри, если она в будущем конусе нижней
    вершины и в прошлом конусе верхней.
    """
    T = (frac / diamond_volume_coeff(d)) ** (1.0 / d)
    h = T / 2.0
    span = max(1.0 - 2 * h, 0.0)
    return [(h + rs.random(d) * span, h) for _ in range(n_reg)]


def count_diamonds(X, regs):
    out = np.empty(len(regs), dtype=np.int64)
    for t, (c, h) in enumerate(regs):
        dt = X[:, 0] - c[0]
        rsp = np.sqrt(((X[:, 1:] - c[1:]) ** 2).sum(1))
        # нижняя вершина в (c0-h, c_prostr), верхняя в (c0+h, c_prostr)
        out[t] = np.count_nonzero((dt + h >= rsp) & (h - dt >= rsp))
    return out


SHAPES = {
    'ящик по осям': (make_boxes, count_boxes),
    'шар': (make_balls, count_balls),
    'причинный ромб': (make_diamonds, count_diamonds),
}


def measure(shape, d, m, frac, n_reg, n_scr, seed):
    make, count = SHAPES[shape]
    n = 2 ** m
    regs = make(d, frac, n_reg, np.random.default_rng(seed))
    sob, poi = [], []
    for s in range(n_scr):
        eng = qmc.Sobol(d=d, scramble=True, seed=seed + 1000 * s)
        sob.append(count(eng.random(n).astype(np.float32), regs))
        Y = np.random.default_rng(seed + 7000 + s).random((n, d)).astype(np.float32)
        poi.append(count(Y, regs))
    sob = np.concatenate(sob).astype(float)
    poi = np.concatenate(poi).astype(float)
    return n, float(sob.std()), float(poi.std()), float(poi.mean())


def fits(ns, ys):
    """Подгонка степенью и логарифмом; возвращает наклоны и остатки."""
    out = {}
    for kind, x in (('power', np.log(ns)), ('log', np.log(np.log(ns)))):
        p = np.polyfit(x, np.log(ys), 1)
        r = np.log(ys) - np.polyval(p, x)
        se = math.sqrt((r ** 2).sum() / max(len(x) - 2, 1)) / \
            math.sqrt(((x - x.mean()) ** 2).sum())
        out[kind] = (float(p[0]), se, float(np.sqrt((r ** 2).mean())))
    return out


def main():
    ap = argparse.ArgumentParser(description='Ящик, шар, причинный ромб')
    ap.add_argument('--dim', type=int, default=4)
    ap.add_argument('--minm', type=int, default=12)
    ap.add_argument('--maxm', type=int, default=18)
    ap.add_argument('--frac', type=float, default=0.002)
    ap.add_argument('--regions', type=int, default=80)
    ap.add_argument('--scramble', type=int, default=3)
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()

    t0 = time.time()
    print(f'd={a.dim}, доля объёма {a.frac}, областей {a.regions}, '
          f'перемешиваний {a.scramble}')
    theory_sob = (a.dim - 1) / (2 * a.dim)
    print(f'гипероднородная теория для степени: (d-1)/2d = {theory_sob:.4f}')
    print()

    summary = {}
    for shape in SHAPES:
        ns, ss, ps = [], [], []
        print(f'  {shape}')
        print(f"{'n':>12}{'σ Соболь':>12}{'σ пуассон':>12}"
              f"{'подавление':>12}{'ср. число':>11}")
        for m in range(a.minm, a.maxm + 1):
            n, s, p, mean = measure(shape, a.dim, m, a.frac,
                                    a.regions, a.scramble, a.seed)
            ns.append(n); ss.append(s); ps.append(p)
            print(f'{n:>12}{s:>12.3f}{p:>12.3f}'
                  f'{(s/p if p else float("nan")):>12.4f}{mean:>11.1f}')
        ns = np.array(ns, float); ss = np.array(ss, float); ps = np.array(ps, float)
        fs, fp = fits(ns, ss), fits(ns, ps)
        summary[shape] = (fs, fp)
        print()

    print('ИТОГ: какая форма закона и какой показатель')
    print(f"{'форма':>16}{'Соболь: степень':>22}{'Соболь: логарифм':>22}"
          f"{'лучше':>10}{'пуассон, степень':>20}")
    for shape, (fs, fp) in summary.items():
        pw, lg, pp = fs['power'], fs['log'], fp['power']
        better = 'логарифм' if lg[2] < pw[2] else 'степень'
        print(f'{shape:>16}{f"{pw[0]:.4f} ± {pw[1]:.4f}":>22}'
              f'{f"{lg[0]:.4f} ± {lg[1]:.4f}":>22}{better:>10}'
              f'{f"{pp[0]:.4f} ± {pp[1]:.4f}":>20}')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
