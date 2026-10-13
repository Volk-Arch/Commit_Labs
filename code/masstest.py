#!/usr/bin/env python3
"""Дешёвый тест на массу: все наблюдаемые против всех вариантов материи.

Тест (из source-doc.md): кандидат на «материю» годится, только если
независимые порядковые наблюдаемые упорядочивают его ОДИНАКОВО.

Здесь он прогоняется систематически. Два вида материи различаются по
двум осям:

  количество   фиксированный радиус, меняется M       «сколько массы»
  сжатие       фиксированное M, меняется радиус       «насколько сжата»

и пять наблюдаемых, все порядковые, все из ОДНОГО произведения матриц:

  порвано связей фона   прямые связи фона, разорванные массой
  задержка Δτ           прирост самой длинной цепи
  доля прямых           прямых связей на отношение, во всём интервале
  размерность           Майрхейма-Мейера по доле связанных пар
  действие BD           на событие

Если по оси «количество» все пять идут в одну сторону, а по оси
«сжатие» расходятся, то масса согласована как число событий и
рассогласована как плотность — это разные выводы, и прежний
(«скалярной массы нет») надо уточнить.

  python3 masstest.py --axis amount
  python3 masstest.py --axis squeeze
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import rel_of, mm_dim
from chaingrav import chain_len
from linkfield import ball, pick_pairs, CENTER

AXES = {
    # ось: (что меняется, значения, фиксированное)
    'amount': ('масса M', (20, 40, 80, 160, 320), 0.07),
    'squeeze': ('радиус', (0.02, 0.05, 0.09, 0.14, 0.22), 160),
}


def observables(Xsub, nb, n_mass):
    """Все наблюдаемые из одного произведения матриц."""
    R = rel_of(Xsub)
    n = len(R)
    A = np.asarray(R, dtype=np.float32)
    AA = A @ A
    L = R & (AA == 0)
    rel = int(R.sum())
    frac = rel / (n * (n - 1) / 2)
    nm = [int(((AA == m) & R).sum()) for m in range(4)]
    out = dict(
        tau=chain_len(R) - 1,
        links_bg=int(L[:nb, :nb].sum()),
        rel_bg=int(R[:nb, :nb].sum()),
        link_frac=L.sum() / max(rel, 1),
        dim=mm_dim(frac) if 0 < frac < 1 else float('nan'),
        bd=(n - nm[0] + 9 * nm[1] - 16 * nm[2] + 8 * nm[3]) / n,
    )
    if n_mass:
        sl = slice(nb, nb + n_mass)
        out['h_add'] = chain_len(R[sl, sl])
    return out


def run(N, k, axis, n_pairs, seed, vol_cap, reals):
    what, values, fixed = AXES[axis]
    rs = np.random.default_rng(seed)
    Xb = rs.random((N, k)).astype(np.float32)
    rb = np.random.default_rng(seed + 77)

    masses = {}
    for v in values:
        M, r = (v, fixed) if axis == 'amount' else (fixed, v)
        masses[v] = [ball(M, k, r, rb) for _ in range(reals)]
    r_max = fixed if axis == 'amount' else max(values)
    pairs = pick_pairs(Xb, r_max, n_pairs, np.random.default_rng(seed + 1))

    acc = {v: [] for v in values}
    base = []
    for a, b in pairs:
        xa, xb = Xb[a], Xb[b]
        m = (Xb > xa).all(1) & (Xb < xb).all(1)
        m[a] = m[b] = False
        ib = np.flatnonzero(m)
        if len(ib) > vol_cap or len(ib) < 200:
            continue
        Xbg, nb = Xb[ib], len(ib)
        b0 = observables(np.vstack([Xbg, xa[None], xb[None]]), nb, 0)
        base.append((nb, b0))
        for v in values:
            for B in masses[v]:
                o = observables(np.vstack([Xbg, B, xa[None], xb[None]]),
                                nb, len(B))
                assert o['rel_bg'] == b0['rel_bg'], 'отношения фона изменились'
                acc[v].append(dict(
                    broken=(b0['links_bg'] - o['links_bg']) / b0['links_bg'],
                    dtau=o['tau'] - b0['tau'],
                    dlink=o['link_frac'] - b0['link_frac'],
                    ddim=o['dim'] - b0['dim'],
                    dbd=o['bd'] - b0['bd'],
                    h_add=o['h_add'],
                ))
    return acc, base, what, values, fixed


# 'dlink' исключена из теста: доля прямых — та же величина, что
# «порвано связей», с обратным знаком (разрыв связи долю и уменьшает).
# Её -1 был бы двойным счётом, а не расхождением. Оставлена в таблице
# как сверка.
KEYS = [('broken', 'порвано связей'), ('dtau', 'задержка Δτ'),
        ('ddim', 'размерность'), ('dbd', 'действие BD')]
SHOW = KEYS + [('dlink', 'доля прямых (сверка)')]


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def sweep(a, axis):
    """Одна ось: таблица средних и ошибок по каждой наблюдаемой."""
    acc, base, what, values, fixed = run(a.n, a.k, axis, a.pairs,
                                         a.seed, a.volcap, a.reals)
    if not base:
        return None
    vols = np.array([b[0] for b in base], float)
    table, errs, used = {}, {}, []
    for kk, _ in SHOW:
        table[kk], errs[kk] = [], []
    hmass = []
    for v in values:
        rows = acc[v]
        if not rows:
            continue
        used.append(v)
        hmass.append(np.mean([r['h_add'] for r in rows]))
        for kk, _ in SHOW:
            arr = np.array([r[kk] for r in rows], float)
            table[kk].append(arr.mean())
            errs[kk].append(arr.std() / np.sqrt(len(arr)))
    return dict(what=what, values=used, fixed=fixed, table=table,
                errs=errs, hmass=hmass, npairs=len(base),
                vol=vols.mean())


def show(res, a, axis):
    print(f'ОСЬ «{res["what"]}»  (фиксировано '
          f'{"радиус" if axis == "amount" else "масса M"} = {res["fixed"]})')
    print(f'  k={a.k}, N={a.n}, пар {res["npairs"]}, V_bg {res["vol"]:.0f}')
    hdr = f'{res["what"]:>10}{"h_масса":>9}'
    for _, nm in SHOW:
        hdr += f'{nm:>24}'
    print(hdr)
    for i, v in enumerate(res['values']):
        line = f'{v:>10}{res["hmass"][i]:>9.2f}'
        for kk, _ in SHOW:
            line += f'{f"{res['table'][kk][i]:+.5f} ± {res['errs'][kk][i]:.5f}":>24}'
        print(line)
    print()


def resolved(res, kk):
    """Разрешена ли наблюдаемая: крайние точки расходятся больше 2σ."""
    t, e = res['table'][kk], res['errs'][kk]
    if len(t) < 2:
        return False
    return abs(t[-1] - t[0]) > 2 * math.hypot(e[0], e[-1])


def main():
    ap = argparse.ArgumentParser(description='Дешёвый тест на массу')
    ap.add_argument('--axis', choices=list(AXES) + ['both'], default='both')
    ap.add_argument('--n', type=int, default=8000)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--pairs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--volcap', type=int, default=1900)
    ap.add_argument('--reals', type=int, default=2)
    a = ap.parse_args()

    t0 = time.time()
    axes = ['amount', 'squeeze'] if a.axis == 'both' else [a.axis]
    res = {}
    for ax in axes:
        r = sweep(a, ax)
        if r is None:
            print(f'ось {ax}: пар не набралось')
            continue
        res[ax] = r
        show(r, a, ax)

    if 'amount' not in res:
        print('без оси количества калибровать знак нечем')
        return

    print('КАЛИБРОВКА ЗНАКА по оси количества')
    print('  («больше событий = больше массы» — это задаёт направление)')
    conv = {}
    for kk, nm in KEYS:
        if not resolved(res['amount'], kk):
            print(f'{nm:>18}: не разрешена (крайние точки в пределах 2σ)')
            continue
        conv[kk] = 1.0 if spearman(res['amount']['table'][kk],
                                   list(range(len(res['amount']['values'])))) > 0 else -1.0
        print(f'{nm:>18}: масса растёт при '
              f'{"росте" if conv[kk] > 0 else "падении"} величины')

    if 'squeeze' not in res or not conv:
        print()
        print('оси сжатия нет — сравнивать не с чем')
        return

    print()
    print('ПРОВЕРКА НА ОСИ СЖАТИЯ в той же калибровке')
    print('  вопрос: сжатая масса тяжелее или легче рыхлой?')
    verdicts = {}
    for kk, nm in KEYS:
        if kk not in conv:
            continue
        if not resolved(res['squeeze'], kk):
            print(f'{nm:>18}: не разрешена')
            continue
        sp = spearman(res['squeeze']['table'][kk],
                      list(range(len(res['squeeze']['values']))))
        d = conv[kk] * (1.0 if sp > 0 else -1.0)
        verdicts[kk] = d
        print(f'{nm:>18}: {"РЫХЛАЯ тяжелее" if d > 0 else "ПЛОТНАЯ тяжелее"}')

    print()
    vs = set(verdicts.values())
    if len(vs) <= 1 and vs:
        print('  ВСЕ СОГЛАСНЫ: масса упорядочена однозначно по обеим осям')
    elif len(vs) > 1:
        pro = sum(1 for v in verdicts.values() if v > 0)
        print(f'  РАСХОЖДЕНИЕ: {pro} наблюдаемых за рыхлую, '
              f'{len(verdicts)-pro} за плотную')
        print('  по количеству масса согласована, по сжатию — нет:')
        print('  сжатие оказывается второй, независимой степенью свободы')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
