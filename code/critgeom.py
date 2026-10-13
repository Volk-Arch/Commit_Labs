#!/usr/bin/env python3
"""СОВМЕСТНАЯ ПРОВЕРКА: ГЕОМЕТРИЧНА ЛИ КРИТИЧЕСКАЯ ФАЗА.

Прежний вопрос «расходится ли зазор по H/L» при N = 12..18 загрязнён
конечным размером у ОБОИХ ансамблей. Формулировка жёстче и в духе всей
сессии: одиночную статистику имитировать можно, СОВМЕСТНУЮ нельзя.

У засева размерности d в области формы a все четыре наблюдаемые
(r, H, W, L) заданы сразу, свободы всего две. Критическая фаза совпала
по r ~ 0.50 и по ширине, но не по высоте (на 22-36% площе) и не по
связям (на 25-33% связнее). Спрашиваем: существует ли ХОТЬ ОДНА
геометрия (d, a), подходящая по всем четырём ОДНОВРЕМЕННО?

Форма берётся настоящая. В 1+1 у диаманта свободы формы нет: сжатие
t->at, x->x/a есть буст, причинный порядок не меняется. Поэтому область
- прямоугольник [0,a] x [0,1]^{d-1}: при больших a почти цепь, при
малых почти антицепь, и r, H, W, L меняются по-настоящему.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ:
  * есть (d, a) с совместимыми отклонениями по всем четырём =>
    критическая фаза геометрична, и видно какая это геометрия;
  * ни одна геометрия не подходит => последний маршрут отбора закрыт.

КОНТРОЛЬ, без которого шкала хи-квадрат ничего не значит: известный
засев (d=2, a=2.0) прогоняется той же сеткой и обязан быть опознан -
малое хи-квадрат и восстановленные d, a около истинных.

ОШИБКИ: у фазы - блочные (наивные занижают втрое, урок crit-doc),
у сетки - обычные по выборке, складываются в квадратуре.
"""
import json
import math
import sys

import numpy as np

from critphase import sample_phase, obs, summarize

KEYS = ('r', 'H', 'W', 'L')
AS = np.exp(np.linspace(math.log(0.3), math.log(6.0), 11))
DS = (2, 3, 4)


def box(n, d, a, rs):
    t = rs.random(n) * a
    x = rs.random((n, d - 1))
    dt = t[None, :] - t[:, None]
    dx2 = ((x[None, :, :] - x[:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2)


def grid(n, rs, nrep):
    g = {}
    for d in DS:
        for a in AS:
            rows = [obs(box(n, d, a, rs)) for _ in range(nrep)]
            m = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
            e = {k: float(np.std([r[k] for r in rows]) / math.sqrt(nrep))
                 for k in KEYS}
            g[(d, float(a))] = (m, e)
    return g


def fit(tm, te, g):
    best = None
    for (d, a), (m, e) in g.items():
        sig = {k: (m[k] - tm[k]) / math.hypot(te[k], e[k]) for k in KEYS}
        chi = sum(v * v for v in sig.values())
        if best is None or chi < best[0]:
            best = (chi, d, a, sig)
    return best


def report(lab, tm, te, g):
    chi, d, a, sig = fit(tm, te, g)
    print(f'  {lab}')
    print('    ' + '  '.join(f'{k}={tm[k]:.3f}±{te[k]:.3f}' for k in KEYS))
    print(f'    лучшая геометрия d={d}, a={a:.2f}, хи2={chi:.1f} '
          f'(4 наблюдаемые)')
    print('    отклонения в сигмах: '
          + '  '.join(f'{k}={sig[k]:+.1f}' for k in KEYS), flush=True)
    return chi, d, a


def main():
    ns = [int(v) for v in sys.argv[1:]] or [12, 16, 20]
    print(__doc__)
    rs = np.random.default_rng(999331)
    out = {}
    for n in ns:
        nrep = 150
        g = grid(n, rs, nrep)
        print(f'\n=== N = {n} ===')
        rows = [obs(box(n, 2, 2.0, rs)) for _ in range(nrep)]
        cm = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
        ce = {k: float(np.std([r[k] for r in rows]) / math.sqrt(nrep))
              for k in KEYS}
        report('КОНТРОЛЬ, засев d=2 a=2.00 (ждём d=2, a~2, малое хи2)',
               cm, ce, g)
        sw = 1200 if n <= 20 else 900
        acc, rate, bet = sample_phase(n, sw, rs)
        tm, te = {}, {}
        for k in KEYS:
            mu, er = summarize(acc, k)
            tm[k], te[k] = float(mu), float(er)
        print(f'  фаза: beta={bet}, доля принятых {rate:.3f}, '
              f'образцов {len(acc)}')
        chi, d, a = report('критическая фаза', tm, te, g)
        out[n] = dict(chi=chi, d=d, a=a, bet=bet, rate=rate,
                      tm=tm, te=te)
    with open(f'/tmp/claude-0/critgeom-{"-".join(map(str, ns))}.json',
              'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    main()
