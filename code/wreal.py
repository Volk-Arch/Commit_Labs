#!/usr/bin/env python3
"""`w` ПО НАСТОЯЩИМ НАБРАННЫМ ДАННЫМ, против точного отклика.

Пункт 2 плана, в лучшем виде, чем задумывался. Закрытие пункта 1
(`exact-doc.md`) показало, что точный отклик воспроизводит набранные
данные везде (90 сравнений, медиана 0.75 сигмы). Значит `w` можно
подгонять не на синтетике, а на РЕАЛЬНО набранных причинностных
множествах.

Три плеча набора — это три значения `w`. При постоянном `w` масштабный
фактор равен `a ~ (eta - eta_s)^p`, `p = 2/(1+3w)`:

  `lin`:  a = 1 + h (eta - eta_0)      -> p = 1,   w = +1/3 (излучение)
  `root`: a = sqrt(1 + c (eta-eta_0))  -> p = 1/2, w = +1   (жёсткая материя)
  `exp`:  a = exp(k (eta - eta_0))     -> p -> oo, w = -1/3 (предел)

У `lin` и `root` одинаковое смещение начала `eta_s = eta_0 - 1/h`
(поскольку `h = c = 0.035`, это −18.57), поэтому оба — степенные законы
ОДНОГО семейства, и `p` подгоняется на общей сетке. `exp` — предельная
точка, на этой сетке недостижимая, поэтому идёт отдельно.

ПОЧИНКА ВЧЕРАШНЕЙ ОШИБКИ. Вчера опорная кривая строилась сприклингом, её
шум был сравним с шумом наблюдения, а складной нож брался только по
наблюдению — отсюда ложные 11 сигм смещения. Здесь опора считается
КВАДРАТУРОЙ (пуассоновского шума нет вовсе), и её погрешность
переносится в `chi²` в квадратурной сумме с погрешностью данных.

КОНТРОЛИ:
  1. два РАЗНЫХ плеча с разными истинными `w` обязаны дать разные
     подогнанные, каждое в пределах ошибки от своей истины;
  2. плоское плечо (`p = 0`, кривизны нет) обязано быть отвергнуто:
     ни одно `p` из сетки его не описывает.

  python3 wreal.py [--n 1200000]
"""
import argparse
import math

import numpy as np

from pencil import KFAC, c_k, IC, IX2, IX3, IV, IM
from refit import load_all
from x3prof import rest_diamond, sub_sample

ETA0 = 10.0
PGRID = np.array([0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.20, 1.45])
VBB = (0.0, 0.45, 0.65, 0.85001)
CBB = (200.0, 400.0, 650.0, 900.0001)


def exact_pow(p, eta_s, v, tau, mid, n, rs, d=4):
    """Точные <X_2>, <X_3> для a = ((eta-eta_s)/(mid-eta_s))^p."""
    gam = 1.0 / math.sqrt(1.0 - v * v)
    ref = mid - eta_s

    def wgt(pt):
        e = mid + ((pt[:, 0] - 0.5 * tau) + v * pt[:, 1]) * gam - eta_s
        return np.maximum(e / ref, 1e-9) ** (4.0 * p)

    s0, w0 = rest_diamond(rs, n, tau, d)
    x = np.column_stack([s0, w0])
    q = np.zeros((n, d)); q[:, 0] = tau
    wx = wgt(x)
    y, txq, ok1 = sub_sample(x, q, rs, d)
    wy = wgt(y)
    x2 = ((wx[ok1] * txq ** d * wy).mean()
          / (c_k(2) * tau ** d * wx.mean() ** 2) - 1.0)
    z, tyq, ok2 = sub_sample(y, q[ok1], rs, d)
    wz = wgt(z)
    x3 = ((wx[ok1][ok2] * txq[ok2] ** d * wy[ok2] * tyq ** d * wz).mean()
          / (c_k(3) * tau ** (2 * d) * wx.mean() ** 3) - 1.0)
    return x2, x3


def cells(A, sd):
    out = []
    for i in range(len(VBB) - 1):
        for j in range(len(CBB) - 1):
            m = ((A[:, IV] >= VBB[i]) & (A[:, IV] < VBB[i + 1])
                 & (A[:, IC] >= CBB[j]) & (A[:, IC] < CBB[j + 1]))
            if m.sum() < 150:
                continue
            us = np.unique(sd[m])
            vals, errs = [], []
            for col in (IX2, IX3):
                jk = np.array([A[m & (sd != u), col].mean() for u in us])
                errs.append(math.sqrt((len(us) - 1) / len(us)
                                      * ((jk - jk.mean()) ** 2).sum()))
                vals.append(A[m, col].mean())
            out.append({'v2': A[m, IV].mean(), 'c1': A[m, IC].mean(),
                        'mid': A[m, IM].mean(), 'obs': np.array(vals),
                        'err': np.array(errs), 'msk': m, 'n': int(m.sum())})
    return out


def ref_grid(cl, eta_s, n, rs, blocks=2):
    """Опора: точный отклик по сетке `p` для каждой ячейки + её ошибка."""
    R = np.zeros((len(PGRID), len(cl), 2))
    E = np.zeros_like(R)
    for a, p in enumerate(PGRID):
        for b, c in enumerate(cl):
            tau = math.sqrt(KFAC * math.sqrt(c['c1']))
            rr = np.array([exact_pow(p, eta_s, math.sqrt(c['v2']), tau,
                                     c['mid'], n, rs) for _ in range(blocks)])
            R[a, b] = rr.mean(axis=0)
            E[a, b] = rr.std(axis=0, ddof=1) / math.sqrt(blocks)
        print(f'    опора p = {p:.2f} готова', flush=True)
    return R, E


def fit_p(obs, err, R, E):
    """chi² по сетке с параболическим уточнением; ошибка опоры включена."""
    v = np.array([(((R[a] - obs) ** 2) / (err ** 2 + E[a] ** 2)).sum()
                  for a in range(len(PGRID))])
    k = int(np.argmin(v))
    if 0 < k < len(PGRID) - 1:
        y0, y1, y2 = v[k - 1], v[k], v[k + 1]
        den = y0 - 2 * y1 + y2
        sh = 0.5 * (y0 - y2) / den if abs(den) > 1e-30 else 0.0
        sh = max(-1.0, min(1.0, sh))
        pp = PGRID[k] + sh * (PGRID[k + 1] - PGRID[k])
    else:
        pp = PGRID[k]
    return pp, v[k], v


def w_of(p):
    return (2.0 / p - 1.0) / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1200000)
    g = ap.parse_args()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(314159)
    arms = load_all()
    eta_s = ETA0 - 1.0 / 0.035
    print(f'eta_s = {eta_s:.2f}, сетка p = {list(PGRID)}, '
          f'точек на узел {g.n}')
    print()
    for fam, ptrue in (('lin', 1.0), ('root', 0.5), ('flat', None)):
        A, sd, par = arms[fam]
        cl = cells(A, sd)
        print(f'--- {fam}: ячеек {len(cl)}, миров {len(np.unique(sd))} ---',
              flush=True)
        R, E = ref_grid(cl, eta_s, g.n, rs)
        obs = np.array([c['obs'] for c in cl])
        err = np.array([c['err'] for c in cl])
        pp, chi, curve = fit_p(obs, err, R, E)
        # складной нож по мирам
        us = np.unique(sd)
        jk = []
        for u in us:
            o2 = []
            for c in cl:
                mm = c['msk'] & (sd != u)
                o2.append([A[mm, IX2].mean(), A[mm, IX3].mean()])
            jk.append(fit_p(np.array(o2), err, R, E)[0])
        jk = np.array(jk); k = len(us)
        se = math.sqrt((k - 1) / k * ((jk - jk.mean()) ** 2).sum())
        dof = obs.size - 1
        print(f'  chi² = {chi:.1f} на {dof} степеней '
              f'({chi/dof:.2f} на степень)')
        if ptrue is None:
            print(f'  ближайшее p = {pp:.3f}, и это КОНТРОЛЬ: плоское '
                  f'не описывается ни одним p')
            print(f'  -> {"ОТВЕРГНУТО" if chi/dof > 3 else "НЕ отвергнуто"}')
        else:
            wh, wt = w_of(pp), w_of(ptrue)
            ew = abs(w_of(pp + se) - w_of(pp - se)) / 2
            print(f'  p = {pp:.4f} +- {se:.4f}   (истина {ptrue})')
            print(f'  w = {wh:+.4f} +- {ew:.4f}   (истина {wt:+.4f})   '
                  f'{abs(wh-wt)/max(ew,1e-9):.1f} сигм')
        print(flush=True)


if __name__ == '__main__':
    main()
