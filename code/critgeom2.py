#!/usr/bin/env python3
"""СОВМЕСТНАЯ ПРОВЕРКА, вариант с НЕПРЕРЫВНОЙ формой.

critgeom.py: контроль опознал известный засев по размерности верно
(d=2), но дал хи2=21 - раздуто ШАГОМ СЕТКИ (шаг по a 35%, а ошибка по r
0.005). На таком полу хи2=44 у фазы ничего не значит.

Починка: наблюдаемые гладко зависят от формы, значит сетка заменяется
непрерывной подгонкой каждой наблюдаемой кубикой по log a, и минимум
хи2 ищется по непрерывному a. Семейство: прямоугольник [0,a]x[0,1]^{d-1}
при d=2,3,4 плюс ДИАМАНТ при d=2,3,4 как отдельная форма.

Свободы по-прежнему две (размерность и форма) против четырёх
наблюдаемых (r, H, W, L).

КОНТРОЛЬ: два известных засева (d=2 a=2.0 и d=3 a=1.0) обязаны дать
хи2 порядка единиц и восстановленные d, a около истинных. Если контроль
не прошёл - шкала снова не та, и вердикта нет.
"""
import json
import math
import sys

import numpy as np

from critphase import sample_phase, obs, summarize
from sectors import sprinkle_d

KEYS = ('r', 'H', 'W', 'L')
LA = np.linspace(math.log(0.25), math.log(7.0), 13)
DS = (2, 3, 4)


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


def curves(n, nrep, rs):
    """для каждой d: кубики каждой наблюдаемой по log a + разброс."""
    cur = {}
    for d in DS:
        ms = {k: [] for k in KEYS}
        es = {k: [] for k in KEYS}
        for la in LA:
            m, e = stats(lambda: box(n, d, math.exp(la), rs), nrep, rs)
            for k in KEYS:
                ms[k].append(m[k])
                es[k].append(e[k])
        cur[d] = ({k: np.polyfit(LA, ms[k], 3) for k in KEYS},
                  {k: float(np.mean(es[k])) for k in KEYS})
    return cur


def best(tm, te, cur, dias):
    out = None
    las = np.linspace(LA[0], LA[-1], 400)
    for d, (pf, ge) in cur.items():
        vals = {k: np.polyval(pf[k], las) for k in KEYS}
        chi = sum(((vals[k] - tm[k]) / math.hypot(te[k], ge[k])) ** 2
                  for k in KEYS)
        i = int(np.argmin(chi))
        sig = {k: (vals[k][i] - tm[k]) / math.hypot(te[k], ge[k])
               for k in KEYS}
        cand = (float(chi[i]), f'd={d} a={math.exp(las[i]):.2f}', sig)
        if out is None or cand[0] < out[0]:
            out = cand
    for d, (m, e) in dias.items():
        sig = {k: (m[k] - tm[k]) / math.hypot(te[k], e[k]) for k in KEYS}
        chi = sum(v * v for v in sig.values())
        if chi < out[0]:
            out = (chi, f'd={d} диамант', sig)
    return out


def report(lab, tm, te, cur, dias):
    chi, geo, sig = best(tm, te, cur, dias)
    print(f'  {lab}')
    print('    ' + '  '.join(f'{k}={tm[k]:.3f}±{te[k]:.3f}' for k in KEYS))
    print(f'    лучшая геометрия: {geo},  хи2={chi:.1f}  (4 набл., 2 своб.)')
    print('    отклонения: '
          + '  '.join(f'{k}={sig[k]:+.1f}σ' for k in KEYS), flush=True)
    return chi, geo


def main():
    ns = [int(v) for v in sys.argv[1:]] or [12, 16]
    print(__doc__)
    rs = np.random.default_rng(555777)
    out = {}
    for n in ns:
        nrep = 120
        cur = curves(n, nrep, rs)
        dias = {d: stats(lambda: sprinkle_d(n, d, rs), nrep, rs)
                for d in DS}
        print(f'\n=== N = {n} ===')
        for d0, a0 in ((2, 2.0), (3, 1.0)):
            m, e = stats(lambda: box(n, d0, a0, rs), nrep, rs)
            report(f'КОНТРОЛЬ: засев d={d0} a={a0} '
                   f'(ждём d={d0}, a~{a0}, хи2 порядка единиц)',
                   m, e, cur, dias)
        acc, rate, bet = sample_phase(n, 1200 if n <= 20 else 900, rs)
        tm = {}
        te = {}
        for k in KEYS:
            mu, er = summarize(acc, k)
            tm[k], te[k] = float(mu), float(er)
        print(f'  фаза: beta={bet}, доля принятых {rate:.3f}, '
              f'образцов {len(acc)}')
        chi, geo = report('КРИТИЧЕСКАЯ ФАЗА', tm, te, cur, dias)
        out[n] = dict(chi=chi, geo=geo, bet=bet, rate=rate, tm=tm, te=te)
    with open(f'/tmp/claude-0/critgeom2-{"-".join(map(str, ns))}.json',
              'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
