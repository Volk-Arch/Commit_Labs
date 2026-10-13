#!/usr/bin/env python3
"""ФАЗА В ТОЧКЕ ПЕРЕХОДА: двумерная ли она по ВСЕМ наблюдаемым.

Доля связанных в точке перехода вышла 0.4939 (N=12) и 0.5023 (N=15) —
ровно значение Майрхейма–Майера для d=2 (`crit-doc.md`). Но доля 0.5
бывает и у не-двумерных порядков, значит условие необходимое и не
достаточное. Решает сравнение ВСЕХ наблюдаемых с засевом d=2,
мерянным ТЕМ ЖЕ КОДОМ при том же N.

При N = 12...18 показатели не мерятся (слишком мало), поэтому
сравниваются ЗНАЧЕНИЯ: доля связанных, высота, ширина, число связей.

Для полноты рядом стоят засевы d = 2, 3, 4 — чтобы видеть, к какому из
них фаза ближе, а не только «похожа ли на двумерный».

КОНТРОЛЬ: доля принятых ходов на рабочем beta печатается; если она
мала, точке не верить.
"""
import math

import numpy as np

from crit9 import Ch, blockerr
from crit import action_full, transitive_ok_add
from crit4 import cells_contrib
from sectors import sprinkle_d
from width3 import width

BET = [0.0, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.25]
TARGET = 0.04


def obs(Rc):
    n = Rc.shape[0]
    R = int(Rc.sum())
    F = Rc.astype(np.float32)
    L = int((Rc & ~((F @ F) > 0.5)).sum())
    h = np.zeros(n, np.int64)
    for x in np.argsort(Rc.sum(1))[::-1]:
        pre = np.nonzero(Rc[:, x])[0]
        h[x] = (h[pre].max() + 1) if len(pre) else 0
    return dict(r=R / (n * (n - 1) / 2), H=int(h.max() + 1),
                W=width(Rc), L=L)


def sample_phase(n, nsweep, rs, target=TARGET, every=6):
    ch = [Ch(n, b) for b in BET]
    kt = int(np.argmin([abs(b - target) for b in BET]))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npr = len(pairs)
    burn = nsweep // 3
    acc = []
    for it in range(nsweep + burn):
        for c in ch:
            for _ in range(npr):
                i, j = pairs[rs.integers(npr)]
                c.step(i, j, rs)
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            if rs.random() < math.exp(min(
                    0.0, (a.beta - b.beta) * (a.S - b.S))):
                a.R, b.R = b.R, a.R
                a.Iv, b.Iv = b.Iv, a.Iv
                a.c, b.c = b.c, a.c
                a.S, b.S = b.S, a.S
                a.nrel, b.nrel = b.nrel, a.nrel
        if it >= burn and (it - burn) % every == 0:
            acc.append(obs(ch[kt].R.copy()))
    rate = ch[kt].acc / max(ch[kt].tries, 1)
    return acc, rate, BET[kt]


def summarize(rows, key):
    v = np.array([r[key] for r in rows], float)
    return v.mean(), blockerr(v, nb=min(20, max(2, len(v) // 4)))


def main():
    print(__doc__)
    rs = np.random.default_rng(606)
    for n in (12, 15, 18):
        rows, rate, bt = sample_phase(n, 900, rs)
        print(f'\n=== N = {n}, beta = {bt}, выборок {len(rows)}, '
              f'доля принятых {rate:.3f} ===')
        print(f"{'':>16}{'доля связ.':>12}{'высота':>10}{'ширина':>10}"
              f"{'связи':>9}")
        line = {}
        for key in ('r', 'H', 'W', 'L'):
            m, e = summarize(rows, key)
            line[key] = (m, e)
        print(f'{"фаза перехода":>16}'
              f'{line["r"][0]:>9.4f}±{line["r"][1]:<.3f}'
              f'{line["H"][0]:>10.2f}{line["W"][0]:>10.2f}'
              f'{line["L"][0]:>9.1f}')
        for d in (2, 3, 4):
            vals = [obs(sprinkle_d(n, d, rs)) for _ in range(40)]
            rr = np.mean([v['r'] for v in vals])
            hh = np.mean([v['H'] for v in vals])
            ww = np.mean([v['W'] for v in vals])
            ll = np.mean([v['L'] for v in vals])
            dev = abs(line['r'][0] - rr) / max(line['r'][1], 1e-6)
            print(f'{f"засев d={d}":>16}{rr:>9.4f}{"":>6}'
                  f'{hh:>10.2f}{ww:>10.2f}{ll:>9.1f}'
                  f'   доля связ. на {dev:.1f}с')
        print(f'  (высота/ширина/связи — значения, не показатели: '
              f'при N={n} показателей нет)', flush=True)


if __name__ == '__main__':
    main()
