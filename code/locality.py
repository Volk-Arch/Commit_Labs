#!/usr/bin/env python3
"""СЖИМАЮТСЯ ЛИ ПЕТЛИ при измельчении. Структурный факт, решающий всё.

ОТКУДА ВОПРОС. `kapcal.py` сажал на россыпь связность с заведомой
кривизной — симметричную калибровку магнитного поля, у которой холономия
по обходу равна потоку, то есть полю на ПЛОЩАДЬ. Ожидание было
kappa ~ N^(-1/2), потому что площадь клетки сжимается как ell^2, а
ell ~ N^(-1/4) в четырёх мерах. Вышло N^(-0.0427), и тут же обнаружилась
причина: средняя площадь обхода шла как N^(-0.0223), то есть почти не
менялась.

ЕСЛИ ЭТО ТАК, рушится вся холономическая программа в том виде, в каком я
её ставил. Калибровочная теория на решётке держится на том, что
кривизна * площадь -> 0 при измельчении; там холономия малой петли и ЕСТЬ
кривизна. Если петли не сжимаются, холономия по ним щупает всю область
целиком — ровно то же, что делает перемешивание. Тогда понятно, почему
перемешанный контроль весь день выходил не слабее теста:
`matchscale.py` −0.5489 (23.5σ) против −0.0613 (4.3σ), `kappa.py`
+0.0252 (4.9σ) против +0.0236 (5.6σ).

ПОЧЕМУ ТАК МОЖЕТ БЫТЬ. Связь в причинном множестве — это пара без
посредников, и такие пары почти разделены светом. Около света
пространственное расстояние не обязано быть малым: при |Δt| = |Δx|
интервал нулевой при любом |Δx|. Поэтому связи длинные, и их число на
событие растёт как N^0.75 (архивный замер) — именно потому, что событие
связано с далёкими.

ЧТО МЕРИМ. Лестница N при той же геометрии, и для каждого N:
    связи     — пространственное |Δx| и собственное время sqrt(Δt²−Δx²);
    циклы     — пространственный поперечник четвёрки и её площадь;
    эталон    — длина дискретности ell = (V/N)^(1/4).
Всё — показателями по N.

ПРЕДСКАЗАНИЯ, расходящиеся по знаку:
    локально   поперечник ~ N^(-1/4), как ell, площадь ~ N^(-1/2);
    нелокально поперечник ~ N^0, площадь ~ N^0.

Собственное время связи ОБЯЗАНО идти как ell (это и есть дискретность).
Расхождение между ним и пространственным поперечником и будет мерой
нелокальности.

  python3 locality.py
  python3 locality.py --ns 400 800 1600 3200 --seeds 10
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through
from response import warped_diamond


def stats(N, T, f_lo, f_hi, samp, ncyc, seed):
    """Длины связей и размеры циклов на одном мире."""
    rs = np.random.default_rng(7701000 + 977 * seed)
    X = warped_diamond(N, T, 0.0, 0.45, rs)
    R = cone_rel(X)
    L = links_of(R)
    U = L | L.T
    i, j = np.nonzero(L)
    if len(i) < 50:
        return None
    pick = rs.choice(len(i), min(4000, len(i)), replace=False)
    i, j = i[pick], j[pick]
    dx = np.linalg.norm(X[i, 1:] - X[j, 1:], axis=1)
    dt = np.abs(X[i, 0] - X[j, 0])
    tau = np.sqrt(np.maximum(dt ** 2 - dx ** 2, 0.0))
    rr = np.linalg.norm(X[:, 1:], axis=1)
    rmax = rr.max()
    pool = np.flatnonzero((rr >= f_lo * rmax) & (rr < f_hi * rmax) &
                          (U.sum(1) >= 3))
    if len(pool) < 25:
        return None
    pool = rs.choice(pool, min(samp, len(pool)), replace=False)
    dia, area, cnt = [], [], 0
    for v in pool:
        for c in cycles_through(U, int(v), ncyc,
                                np.random.default_rng(31 + seed)):
            P = X[list(c)]
            d = 0.0
            for p in range(len(c)):
                for q in range(p + 1, len(c)):
                    d = max(d, float(np.linalg.norm(P[p, 1:] - P[q, 1:])))
            dia.append(d)
            area.append(abs(sum(P[t, 1] * P[(t + 1) % len(c), 2] -
                                P[t, 2] * P[(t + 1) % len(c), 1]
                                for t in range(len(c)))) / 2)
            cnt += 1
    del R, L, U, X
    if cnt < 40:
        return None
    vol = math.pi / 24.0 * T ** 4  # объём ромба 4-мер: pi T^4 / 24
    return (float(np.mean(dx)), float(np.median(dx)), float(np.mean(tau)),
            float(np.mean(dia)), float(np.mean(area)),
            float((vol / N) ** 0.25), cnt)


def fit(ns, vals):
    ok = [k for k in range(len(vals))
          if vals[k] is not None and vals[k] > 1e-14]
    if len(ok) < 3:
        return float('nan'), float('nan')
    x = np.log([ns[k] for k in ok])
    y = np.log([vals[k] for k in ok])
    b, c = np.polyfit(x, y, 1)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - b * x - c) ** 2).sum() / ss if ss > 1e-18 else 1.0
    return float(b), float(r2)


def main():
    ap = argparse.ArgumentParser(description='Сжимаются ли петли')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[400, 800, 1600, 3200])
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--shell', type=float, nargs=2, default=[0.45, 0.75])
    ap.add_argument('--samp', type=int, default=260)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    lo, hi = a.shell
    print(f'лестница N = {a.ns}, ромб T={a.t}, оболочка r/rmax {lo}..{hi}')
    print(f'засевов {a.seeds}')
    print('ПРЕДСКАЗАНИЯ: локально — поперечник ~ N^-0.25, площадь ~ N^-0.50;')
    print('              нелокально — оба ~ N^0')
    print('собственное время связи обязано идти как ell ~ N^-0.25')
    print()
    cols = ['|Δx| связи', 'медиана', 'время связи', 'попереч. цикла',
            'площадь', 'ell']
    print(f"{'N':>7}" + ''.join(f'{c:>16}' for c in cols) + f"{'циклов':>8}")

    acc = [[] for _ in cols]
    for N in a.ns:
        rows = []
        for t in range(a.seeds):
            r = stats(N, a.t, lo, hi, a.samp, a.ncyc, t)
            if r:
                rows.append(r)
        if not rows:
            print(f'{N:>7}  мало данных')
            for k in range(len(cols)):
                acc[k].append(None)
            continue
        M = np.array(rows)
        line = f'{N:>7}'
        for k in range(len(cols)):
            v = float(M[:, k].mean())
            acc[k].append(v)
            line += f'{v:>16.5f}'
        print(line + f'{int(M[:, 6].sum()):>8}', flush=True)

    print()
    for k, c in enumerate(cols):
        b, r2 = fit(a.ns, acc[k])
        print(f'  {c:<16} ~ N^{b:+.4f}   R^2 = {r2:.4f}')

    be, _ = fit(a.ns, acc[5])
    bt, _ = fit(a.ns, acc[2])
    bd, _ = fit(a.ns, acc[3])
    ba, _ = fit(a.ns, acc[4])
    print()
    print(f'  время связи / ell:      {bt:+.4f} против {be:+.4f} — '
          f'дискретность на месте' if abs(bt - be) < 0.08 else
          f'  время связи {bt:+.4f} против ell {be:+.4f} — РАСХОДЯТСЯ')
    print(f'  поперечник цикла {bd:+.4f}: при локальности было бы '
          f'{be:+.4f}')
    print(f'  площадь цикла    {ba:+.4f}: при локальности было бы '
          f'{2*be:+.4f}')
    print()
    print('  поперечник около нуля => петли НЕ сжимаются, и холономия по')
    print('  ним щупает всю область, а не кривизну в точке')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
