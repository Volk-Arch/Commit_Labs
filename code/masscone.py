#!/usr/bin/env python3
"""Спор наблюдаемых о массе — на настоящем причинном порядке.

Пересчёт masstest-doc.md. Там на покомпонентном порядке вышло:

  по оси КОЛИЧЕСТВА все наблюдаемые согласны (больше событий = тяжелее);
  по оси СЖАТИЯ раскол 2:1 — задержка и размерность за плотную массу,
  порванные связи за рыхлую;
  и ни одна наблюдаемая не внутренняя: показатели по объёму интервала
  -0.91, -0.81, -0.57 вместо нуля.

Вопрос: решается ли раскол на причинном порядке.

Схема та же. Отношения между фоновыми событиями от добавления массы не
меняются (координаты фона те же), меняются только прямые связи — это
проверяется утверждением на каждом прогоне.

  python3 masscone.py --axis both
"""
import argparse
import math
import time

import numpy as np

from coneworld import sprinkle, ball4, cone_rel, between
from core import mm_dim
from chaingrav import chain_len

CENTRE = np.zeros(4)
AXES = {
    'amount':  ('масса M', (25, 50, 100, 200, 400), 0.10),
    'squeeze': ('радиус', (0.04, 0.08, 0.15, 0.25, 0.40), 150),
}
KEYS = [('broken', 'порвано связей'), ('dtau', 'задержка Δτ'),
        ('ddim', 'размерность')]


def covering_pairs(X, r, n_pairs, rs, margin=0.05):
    t = X[:, 0]
    x = np.linalg.norm(X[:, 1:], axis=1)
    before = np.flatnonzero((-t - x > r + margin) & (t < 0))
    after = np.flatnonzero((t - x > r + margin) & (t > 0))
    if len(before) == 0 or len(after) == 0:
        return []
    n = min(n_pairs, len(before), len(after))
    return list(zip(rs.choice(before, n, replace=False),
                    rs.choice(after, n, replace=False)))


def obs(Xbg, ib, a, b, Xadd):
    """Наблюдаемые из одного произведения матриц."""
    parts = [Xbg[a][None], Xbg[ib]]
    nadd = 0
    if Xadd is not None and len(Xadd):
        parts.append(Xadd)
        nadd = len(Xadd)
    parts.append(Xbg[b][None])
    R = cone_rel(np.vstack(parts))
    n = len(R)
    nb = len(ib)
    A = np.asarray(R, dtype=np.float32)
    AA = A @ A
    L = R & (AA == 0)
    frac = R.sum() / (n * (n - 1) / 2)
    out = dict(tau=chain_len(R) - 1,
               links_bg=int(L[:1 + nb, :1 + nb].sum()),
               rel_bg=int(R[:1 + nb, :1 + nb].sum()),
               dim=mm_dim(frac) if 0 < frac < 1 else float('nan'))
    if nadd:
        sl = slice(1 + nb, 1 + nb + nadd)
        out['h_add'] = chain_len(R[sl, sl])
    else:
        out['h_add'] = 0
    return out


def run(N, k, axis, n_pairs, seed, vol_cap, reals):
    what, values, fixed = AXES[axis]
    rs = np.random.default_rng(seed)
    X = sprinkle(N, 4, rs)
    rb = np.random.default_rng(seed + 77)
    masses = {}
    for v in values:
        M, r = (v, fixed) if axis == 'amount' else (fixed, v)
        masses[v] = [ball4(M, 4, r, CENTRE, rb) for _ in range(reals)]
    r_max = fixed if axis == 'amount' else max(values)
    pairs = covering_pairs(X, r_max, n_pairs, np.random.default_rng(seed + 1))

    acc = {v: [] for v in values}
    vols = []
    for a, b in pairs:
        ib = np.flatnonzero(between(X, a, b))
        if len(ib) > vol_cap or len(ib) < 100:
            continue
        b0 = obs(X, ib, a, b, None)
        vols.append(len(ib))
        for v in values:
            for B in masses[v]:
                ins = np.flatnonzero(
                    (B[:, 0] - X[a, 0] >
                     np.linalg.norm(B[:, 1:] - X[a, 1:], axis=1)) &
                    (X[b, 0] - B[:, 0] >
                     np.linalg.norm(X[b, 1:] - B[:, 1:], axis=1)))
                o = obs(X, ib, a, b, B[ins])
                assert o['rel_bg'] == b0['rel_bg'], 'отношения фона изменились'
                acc[v].append(dict(
                    broken=(b0['links_bg'] - o['links_bg']) / b0['links_bg'],
                    dtau=o['tau'] - b0['tau'],
                    ddim=o['dim'] - b0['dim'],
                    h_add=o['h_add'], nin=len(ins)))
    return acc, np.array(vols, float), what, values, fixed


def spearman(x):
    r = np.argsort(np.argsort(x)).astype(float)
    i = np.arange(len(x), dtype=float)
    return float(np.corrcoef(r, i)[0, 1])


def sweep(a, axis):
    acc, vols, what, values, fixed = run(a.n, 4, axis, a.pairs, a.seed,
                                         a.volcap, a.reals)
    if len(vols) == 0:
        return None
    print(f'ОСЬ «{what}» (фиксировано '
          f'{"радиус" if axis == "amount" else "масса"} = {fixed}), '
          f'пар {len(vols)}, V_bg {vols.mean():.0f}')
    hdr = f'{what:>10}{"h_масса":>9}{"внутри":>8}'
    for _, nm in KEYS:
        hdr += f'{nm:>24}'
    print(hdr)
    table, errs = {k: [] for k, _ in KEYS}, {k: [] for k, _ in KEYS}
    for v in values:
        rows = acc[v]
        if not rows:
            continue
        line = (f'{v:>10}{np.mean([r["h_add"] for r in rows]):>9.2f}'
                f'{np.mean([r["nin"] for r in rows]):>8.0f}')
        for kk, _ in KEYS:
            arr = np.array([r[kk] for r in rows], float)
            table[kk].append(arr.mean())
            errs[kk].append(arr.std() / math.sqrt(len(arr)))
            line += f'{f"{arr.mean():+.5f} ± {arr.std()/math.sqrt(len(arr)):.5f}":>24}'
        print(line)
    print()
    return table, errs


def main():
    ap = argparse.ArgumentParser(description='Спор о массе на конусе')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--pairs', type=int, default=14)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--volcap', type=int, default=1500)
    ap.add_argument('--reals', type=int, default=2)
    a = ap.parse_args()

    t0 = time.time()
    res = {}
    for ax in ('amount', 'squeeze'):
        r = sweep(a, ax)
        if r:
            res[ax] = r
    if 'amount' not in res:
        print('без оси количества калибровать знак нечем')
        return

    print('КАЛИБРОВКА ЗНАКА по оси количества')
    conv = {}
    ta, ea = res['amount']
    for kk, nm in KEYS:
        if abs(ta[kk][-1] - ta[kk][0]) < 2 * math.hypot(ea[kk][0], ea[kk][-1]):
            print(f'{nm:>18}: не разрешена')
            continue
        conv[kk] = 1.0 if spearman(ta[kk]) > 0 else -1.0
        print(f'{nm:>18}: масса растёт при '
              f'{"росте" if conv[kk] > 0 else "падении"} величины')
    if 'squeeze' not in res or not conv:
        return
    print()
    print('ПРОВЕРКА НА ОСИ СЖАТИЯ в той же калибровке')
    ts, es = res['squeeze']
    verd = {}
    for kk, nm in KEYS:
        if kk not in conv:
            continue
        if abs(ts[kk][-1] - ts[kk][0]) < 2 * math.hypot(es[kk][0], es[kk][-1]):
            print(f'{nm:>18}: не разрешена')
            continue
        verd[kk] = conv[kk] * (1.0 if spearman(ts[kk]) > 0 else -1.0)
        print(f'{nm:>18}: '
              f'{"РЫХЛАЯ тяжелее" if verd[kk] > 0 else "ПЛОТНАЯ тяжелее"}')
    print()
    vs = set(verd.values())
    if len(vs) == 1:
        print('  ВСЕ СОГЛАСНЫ: масса упорядочена однозначно по обеим осям')
        print('  (на покомпонентном порядке был раскол 2:1)')
    elif len(vs) > 1:
        pro = sum(1 for v in verd.values() if v > 0)
        print(f'  РАСХОЖДЕНИЕ: {pro} за рыхлую, {len(verd)-pro} за плотную')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
