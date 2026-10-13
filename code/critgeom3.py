#!/usr/bin/env python3
"""СОВМЕСТНАЯ ПРОВЕРКА, вариант 3: честный бюджет ошибок модели.

critgeom2.py: контроль d=3 a=1.0 давал хи2 = 1.3, 8.5, 18.7, 18.5 при
N = 12,16,20,24 - РАСТУЩИЙ, и целиком из одной наблюдаемой L (-3.9σ).
Причина не физическая: L быстро растёт по a, глобальная кубика на
28-кратном диапазоне её не ловит, а остаток подгонки в бюджет ошибок не
входил - только ошибка выборки. То есть у модели была систематика,
которую я не считал. Это та же ошибка, что с весами в wscan: якорь
обязан идти тем же путём, что проверяемое.

Починка:
  * густая сетка по log a (31 точка вместо 13);
  * модель - линейная интерполяция между измеренными точками, то есть
    проходит через них точно;
  * погрешность интерполяции оценивается строго, по вторым разностям
    (|f''|h^2/8), и входит в бюджет вместе с ошибкой выборки.

КОНТРОЛЬ прежний и обязательный: известные засевы (d=2 a=2.0,
d=3 a=1.0, d=4 a=1.5) обязаны дать хи2 порядка единиц и восстановленные
d, a около истинных. Пока контроль не прошёл - фазу не смотрим.
"""
import json
import math
import sys

import numpy as np

from critphase import sample_phase, obs, summarize
from sectors import sprinkle_d

KEYS = ('r', 'H', 'W', 'L')
LA = np.linspace(math.log(0.25), math.log(7.0), 31)
DS = (2, 3, 4)
DENSE = np.linspace(LA[0], LA[-1], 600)


def box(n, d, a, rs):
    t = rs.random(n) * a
    x = rs.random((n, d - 1))
    dt = t[None, :] - t[:, None]
    dx2 = ((x[None, :, :] - x[:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2)


def stats(mk, nrep, rs):
    rows = [obs(mk()) for _ in range(nrep)]
    m = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    e = {k: float(np.std([r[k] for r in rows]) / math.sqrt(nrep))
         for k in KEYS}
    return m, e


def model(n, nrep, rs):
    """для каждой d: интерполяция по log a + полный бюджет ошибок."""
    out = {}
    h = LA[1] - LA[0]
    for d in DS:
        ms = {k: [] for k in KEYS}
        es = {k: [] for k in KEYS}
        for la in LA:
            m, e = stats(lambda: box(n, d, math.exp(la), rs), nrep, rs)
            for k in KEYS:
                ms[k].append(m[k])
                es[k].append(e[k])
        vals, errs = {}, {}
        for k in KEYS:
            y = np.array(ms[k])
            d2 = np.abs(np.diff(y, 2)).max() if len(y) > 2 else 0.0
            vals[k] = np.interp(DENSE, LA, y)
            errs[k] = math.hypot(float(np.mean(es[k])), d2 / 8.0)
        out[d] = (vals, errs)
    return out


def best(tm, te, mod, dias):
    out = None
    for d, (vals, errs) in mod.items():
        chi = sum(((vals[k] - tm[k]) / math.hypot(te[k], errs[k])) ** 2
                  for k in KEYS)
        i = int(np.argmin(chi))
        sig = {k: (vals[k][i] - tm[k]) / math.hypot(te[k], errs[k])
               for k in KEYS}
        cand = (float(chi[i]), f'd={d} a={math.exp(DENSE[i]):.2f}', sig)
        if out is None or cand[0] < out[0]:
            out = cand
    for d, (m, e) in dias.items():
        sig = {k: (m[k] - tm[k]) / math.hypot(te[k], e[k]) for k in KEYS}
        chi = sum(v * v for v in sig.values())
        if chi < out[0]:
            out = (chi, f'd={d} диамант', sig)
    return out


def report(lab, tm, te, mod, dias):
    chi, geo, sig = best(tm, te, mod, dias)
    print(f'  {lab}')
    print(f'    лучшая геометрия: {geo},  хи2={chi:.1f}')
    print('    отклонения: '
          + '  '.join(f'{k}={sig[k]:+.1f}σ' for k in KEYS), flush=True)
    return chi, geo


def main():
    args = sys.argv[1:]
    conly = '--controls' in args
    ns = [int(v) for v in args if v.isdigit()] or [20, 24]
    print(__doc__)
    rs = np.random.default_rng(242526)
    out = {}
    for n in ns:
        nrep = 150
        mod = model(n, nrep, rs)
        dias = {d: stats(lambda: sprinkle_d(n, d, rs), nrep, rs)
                for d in DS}
        print(f'\n=== N = {n} ===')
        for d0, a0 in ((2, 2.0), (3, 1.0), (4, 1.5)):
            m, e = stats(lambda: box(n, d0, a0, rs), nrep, rs)
            report(f'КОНТРОЛЬ d={d0} a={a0}', m, e, mod, dias)
        if conly:
            continue
        acc, rate, bet = sample_phase(n, 1200 if n <= 20 else 900, rs)
        tm, te = {}, {}
        for k in KEYS:
            mu, er = summarize(acc, k)
            tm[k], te[k] = float(mu), float(er)
        print(f'  фаза: beta={bet}, принятых {rate:.3f}, '
              f'образцов {len(acc)}')
        print('    ' + '  '.join(f'{k}={tm[k]:.3f}±{te[k]:.3f}'
                                 for k in KEYS))
        chi, geo = report('КРИТИЧЕСКАЯ ФАЗА', tm, te, mod, dias)
        out[n] = dict(chi=chi, geo=geo, bet=bet, rate=rate, tm=tm, te=te)
    if out:
        with open(f'/tmp/claude-0/critgeom3-{"-".join(map(str, ns))}.json',
                  'w') as f:
            json.dump(out, f)


if __name__ == '__main__':
    main()
