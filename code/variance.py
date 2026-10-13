#!/usr/bin/env python3
"""Как растут флуктуации числа точек: степень или логарифм.

Довод про Λ (hyperuniform-doc.md) держался на измеренном показателе
степени, экстраполированном с n=32768 на 10^243. Экстраполяция на 238
порядков — слабое место.

Но теория расхождений даёт форму закона, а не подгонку. Для
квазислучайной последовательности L2-расхождение есть
O((log n)^(d-1)/n) сверху (Соболь, Нидеррайтер) и не меньше
c*(log n)^((d-1)/2)/n снизу (Рот). Дисперсия числа точек в области
связана с квадратом расхождения, умноженным на n^2, поэтому

    пуассон        σ ~ sqrt(f(1-f) n)        — корень из n
    квазислучайно  σ ~ (log n)^(d-1)         — логарифм

Это разные классы роста, не разные показатели. Если измерение
подтверждает логарифм, экстраполяция перестаёт быть экстраполяцией.

ВАЖНО про схему: ящики ставятся СЛУЧАЙНО, не по сетке. Соболевские
сетки по построению точно равнораспределены в двоичных ящиках, и на
выровненных по сетке ящиках дисперсия выходит почти нулевой — это
артефакт схемы, а не свойство мира. Плюс последовательность
перемешивается (scramble), иначе дисперсия вообще не определена.

  python3 variance.py --dim 4
  python3 variance.py --dim 4 --maxm 20
"""
import argparse
import math
import time

import numpy as np
from scipy.stats import qmc


def count_in_boxes(X, boxes):
    """Сколько точек в каждом ящике. Ящики: (lo, hi) по всем осям."""
    out = np.empty(len(boxes), dtype=np.int64)
    for t, (lo, hi) in enumerate(boxes):
        out[t] = np.count_nonzero(((X >= lo) & (X < hi)).all(1))
    return out


def random_boxes(d, frac, n_boxes, rs):
    """Случайно размещённые ящики заданной доли объёма.

    Форма ящика тоже случайна (случайные стороны с нужным
    произведением), чтобы не привязываться к кубам.
    """
    boxes = []
    for _ in range(n_boxes):
        # случайные стороны с произведением frac
        e = rs.random(d)
        e = e / e.sum()
        side = frac ** e                      # произведение = frac
        side = np.minimum(side, 1.0)
        lo = rs.random(d) * (1.0 - side)
        boxes.append((lo, lo + side))
    return boxes


def random_balls(d, frac, n_balls, rs):
    """Случайно размещённые шары той же доли объёма.

    Проверка критическая: теория расхождений — про ящики по осям, а
    причинный ромб не ящик. Если подавление у Соболя держится только на
    ящиках, довод про Λ неприменим к физике.
    """
    # объём шара радиуса r в d измерениях
    vol_unit = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
    r = (frac / vol_unit) ** (1.0 / d)
    out = []
    for _ in range(n_balls):
        c = r + rs.random(d) * (1.0 - 2 * r) if 2 * r < 1 else rs.random(d)
        out.append((c, r))
    return out, r


def count_in_balls(X, balls):
    out = np.empty(len(balls), dtype=np.int64)
    for t, (c, r) in enumerate(balls):
        out[t] = np.count_nonzero(((X - c) ** 2).sum(1) < r * r)
    return out


def measure_balls(d, m, frac, n_balls, n_scramble, seed):
    n = 2 ** m
    rs = np.random.default_rng(seed)
    balls, r = random_balls(d, frac, n_balls, rs)
    sob, poi = [], []
    for s in range(n_scramble):
        eng = qmc.Sobol(d=d, scramble=True, seed=seed + 1000 * s)
        X = eng.random(n).astype(np.float32)
        sob.append(count_in_balls(X, balls))
        Y = np.random.default_rng(seed + 7000 + s).random((n, d)).astype(np.float32)
        poi.append(count_in_balls(Y, balls))
    sob = np.concatenate(sob).astype(float)
    poi = np.concatenate(poi).astype(float)
    ms, mp = sob.mean(), poi.mean()
    return n, float(sob.std()), float(poi.std()), float(ms)


def measure(d, m, frac, n_boxes, n_scramble, seed):
    """Стандартное отклонение числа точек: Соболь против пуассона."""
    n = 2 ** m
    rs = np.random.default_rng(seed)
    boxes = random_boxes(d, frac, n_boxes, rs)

    sob, poi = [], []
    for s in range(n_scramble):
        eng = qmc.Sobol(d=d, scramble=True, seed=seed + 1000 * s)
        X = eng.random(n).astype(np.float32)
        sob.append(count_in_boxes(X, boxes))
        Y = np.random.default_rng(seed + 7000 + s).random((n, d)).astype(np.float32)
        poi.append(count_in_boxes(Y, boxes))

    # дисперсия по размещениям ящиков и по перемешиваниям, вокруг
    # ожидаемого значения frac*n
    exp = frac * n
    sob = np.concatenate(sob).astype(float)
    poi = np.concatenate(poi).astype(float)
    return n, float(np.sqrt(((sob - exp) ** 2).mean())), \
        float(np.sqrt(((poi - exp) ** 2).mean()))


def fit_power(ns, ys):
    a, _ = np.polyfit(np.log(ns), np.log(ys), 1)
    return a


def fit_log(ns, ys, d):
    """Подгонка sigma = C*(log n)^b: наклон в логарифме логарифма."""
    b, _ = np.polyfit(np.log(np.log(ns)), np.log(ys), 1)
    return b


def resid(ns, ys, kind, d):
    x = np.log(ns) if kind == 'power' else np.log(np.log(ns))
    p = np.polyfit(x, np.log(ys), 1)
    pred = np.polyval(p, x)
    return float(np.sqrt(((np.log(ys) - pred) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(description='Рост флуктуаций: степень или логарифм')
    ap.add_argument('--dim', type=int, default=4)
    ap.add_argument('--minm', type=int, default=10)
    ap.add_argument('--maxm', type=int, default=19)
    ap.add_argument('--frac', type=float, default=0.05)
    ap.add_argument('--boxes', type=int, default=150)
    ap.add_argument('--scramble', type=int, default=4)
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()

    t0 = time.time()
    print(f'd={a.dim}, доля объёма {a.frac}, ящиков {a.boxes} '
          f'(случайно размещённых), перемешиваний {a.scramble}')
    print()
    print(f"{'n':>10}{'σ Соболь':>13}{'σ пуассон':>13}"
          f"{'√(f(1-f)n)':>14}{'подавление':>13}")
    ns, ss, ps = [], [], []
    for m in range(a.minm, a.maxm + 1):
        n, s, p = measure(a.dim, m, a.frac, a.boxes, a.scramble, a.seed)
        theory = math.sqrt(a.frac * (1 - a.frac) * n)
        ns.append(n); ss.append(s); ps.append(p)
        print(f'{n:>10}{s:>13.3f}{p:>13.3f}{theory:>14.3f}{s/p:>13.5f}')

    ns = np.array(ns, float); ss = np.array(ss, float); ps = np.array(ps, float)
    print()
    print('ПУАССОН (контроль): должен дать показатель 0.5')
    print(f'  степенной показатель: {fit_power(ns, ps):.4f}')
    print()
    print('СОБОЛЬ: какая форма описывает лучше')
    pa = fit_power(ns, ss)
    lb = fit_log(ns, ss, a.dim)
    rp = resid(ns, ss, 'power', a.dim)
    rl = resid(ns, ss, 'log', a.dim)
    print(f'  степень  σ ~ n^{pa:.4f}        остаток {rp:.4f}')
    print(f'  логарифм σ ~ (log n)^{lb:.4f}  остаток {rl:.4f}')
    print(f'  теория для логарифма: показатель d-1 = {a.dim-1}')
    print(f'  лучше описывает: {"ЛОГАРИФМ" if rl < rp else "СТЕПЕНЬ"}')

    print()
    print('ПЕРЕСЧЁТ НА КОСМОЛОГИЧЕСКИЙ ОБЪЁМ  N = 4.3e243')
    N = 4.3e243
    lnN = math.log(N)
    pois = math.sqrt(a.frac * (1 - a.frac) * N)
    # привязываем обе формы к последней измеренной точке
    C_pow = ss[-1] / ns[-1] ** pa
    C_log = ss[-1] / math.log(ns[-1]) ** lb
    C_thr = ss[-1] / math.log(ns[-1]) ** (a.dim - 1)
    for nm, val in (('по измеренной степени', C_pow * N ** pa),
                    ('по измеренному логарифму', C_log * lnN ** lb),
                    ('по теоретическому (log n)^(d-1)', C_thr * lnN ** (a.dim - 1))):
        sup = val / pois
        print(f'  {nm:>34}: σ = {val:.3e}, подавление {sup:.2e} '
              f'({math.log10(sup):+.1f} порядков)')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
