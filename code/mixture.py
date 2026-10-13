#!/usr/bin/env python3
"""Распределение строгости в населении, а не одно число.

На семи признаках (три средних, два разброса ПО ЛЮДЯМ, связь между ними
и доля людей без петель) однородная пороговая модель провалилась: 0
подгонок из 288, и хуже всех шёл разброс безразличия — модель 0.101
против наблюдаемых 0.158.

Диагноз прямой: у людей РАЗНАЯ строгость, а однородная модель даёт всем
одну. Значит модель должна быть смесью, и выходом становится не одно
число, а распределение строгости — что полезнее.

Семь признаков против пяти параметров смеси: запас остаётся.

  python3 mixture.py
  python3 mixture.py --reps 25
"""
import argparse
import math
import time
import glob

import numpy as np

IND, WIN, LOS = 0, 1, 2
NOPT = 6
NAMES = ['безразл. средн', '3-циклы средн', '4-циклы средн',
         'безразл. разброс', '3-циклы разброс', 'связь между ними',
         'без петель, доля']
PAIRS = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4),
         (2, 4), (3, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]


def feats(R):
    n = len(R)
    iu = np.triu_indices(n, 1)
    ind = float((R[iu] == IND).mean())
    A = np.asarray(R == WIN, dtype=np.float64)
    A2 = A @ A
    c3 = float(np.trace(A2 @ A)) / 3.0 / math.comb(n, 3)
    c4 = float(np.trace(A2 @ A2)) / 4.0 / math.comb(n, 4)
    return ind, c3, c4


def profile(F):
    c = np.corrcoef(F[:, 0], F[:, 1])[0, 1] if F[:, 1].std() > 1e-12 else 0.0
    return np.array([F[:, 0].mean(), F[:, 1].mean(), F[:, 2].mean(),
                     F[:, 0].std(), F[:, 1].std(), c, (F[:, 1] == 0).mean()])


def noisify(R, eps, rs):
    if eps <= 0:
        return R
    iu = np.triu_indices(NOPT, 1)
    hit = rs.random(len(iu[0])) < eps
    pick = np.where(rs.random(len(iu[0])) < 0.5, WIN, LOS)
    v = R[iu].copy()
    v[hit] = pick[hit]
    R[iu] = v
    R[(iu[1], iu[0])] = np.where(v == WIN, LOS, np.where(v == LOS, WIN, IND))
    return R


def one_threshold(k, s, eps, sq, rs):
    """Порог со строгостью s у данного человека."""
    m = int(round(s * k))
    m = max(min(m, k), k // 2 + 1)
    q = rs.normal(0, sq, NOPT)
    U = q[:, None] + rs.normal(0, 1, (NOPT, k))
    w = np.zeros((NOPT, NOPT), dtype=np.int16)
    for t in range(k):
        w += (U[:, None, t] > U[None, :, t])
    R = np.full((NOPT, NOPT), IND, dtype=np.int8)
    R[w >= m] = WIN
    R[(k - w) >= m] = LOS
    np.fill_diagonal(R, IND)
    return noisify(R, eps, rs)


def one_utility(k, band, eps, sq, rs):
    """Взвешенная полезность с полосой безразличия band."""
    w = rs.random(k)
    w /= w.sum()
    q = rs.normal(0, sq, NOPT)
    U = q + (rs.normal(0, 1, (NOPT, k)) @ w)
    d = U[:, None] - U[None, :]
    R = np.full((NOPT, NOPT), IND, dtype=np.int8)
    R[d > band] = WIN
    R[d < -band] = LOS
    np.fill_diagonal(R, IND)
    return noisify(R, eps, rs)


def group_threshold(k, s0, ds, eps, sq, nsub, rs):
    ss = np.clip(rs.normal(s0, ds, nsub), 0.5, 1.0)
    return np.array([feats(one_threshold(k, s, eps, sq, rs)) for s in ss])


def group_utility(k, b0, db, eps, sq, nsub, rs):
    bb = np.clip(rs.normal(b0, db, nsub), 0.0, 2.0)
    return np.array([feats(one_utility(k, b, eps, sq, rs)) for b in bb])


def fit(name, group, params, po, nsub, reps, seed=11):
    out = []
    for pr in params:
        rs = np.random.default_rng(seed)
        P = np.array([profile(group(*pr, nsub, rs)) for _ in range(reps)])
        mu, sd = P.mean(0), P.std(0)
        z = (mu - po) / np.maximum(sd * math.sqrt(2), 1e-6)
        out.append((float(np.sqrt((z ** 2).sum())), float(np.abs(z).max()),
                    pr, mu, z))
    out.sort()
    tot, worst, pr, mu, z = out[0]
    print(f'  {name}: расхождение {tot:.1f} по семи, худшее одно '
          f'{worst:.1f}σ, параметры {pr}')
    w = int(np.argmax(np.abs(z)))
    print(f'    хуже всех: {NAMES[w]} — {mu[w]:.4f} против {po[w]:.4f} '
          f'({z[w]:+.1f}σ)')
    ok = [o for o in out if o[1] < 3]
    print(f'    подгонок, где ВСЕ семь в пределах 3σ: {len(ok)} из {len(out)}')
    return out, ok


def load_real():
    import pyreadr
    p = (glob.glob('/tmp/prefmod/data/cemspc.RData') +
         glob.glob('/tmp/claude-0/**/prefmod/data/cemspc.RData',
                   recursive=True))
    df = list(pyreadr.read_r(p[0]).values())[0]
    obs = []
    for _, r in df.iterrows():
        R = np.full((6, 6), IND, dtype=np.int8)
        ok = 0
        for t, (i, j) in enumerate(PAIRS):
            v = r[f'V{t+1}']
            if np.isnan(v):
                continue
            ok += 1
            if v == 0:
                R[i, j], R[j, i] = WIN, LOS
            elif v == 2:
                R[i, j], R[j, i] = LOS, WIN
        if ok >= 13:
            obs.append(feats(R))
    return np.array(obs)


def main():
    ap = argparse.ArgumentParser(description='Распределение строгости')
    ap.add_argument('--reps', type=int, default=15)
    a = ap.parse_args()

    t0 = time.time()
    O = load_real()
    po = profile(O)
    nsub = len(O)
    print(f'РЕАЛЬНЫЙ ПРОФИЛЬ ({nsub} человек)')
    for nm, v in zip(NAMES, po):
        print(f'  {nm:>18}  {v:.4f}')
    print()
    print('СМЕСЬ: у каждого человека своя строгость')
    tp = [(k, s0, ds, e, sq) for k in (4, 6, 8)
          for s0 in (0.6, 0.7, 0.8, 0.9) for ds in (0.0, 0.1, 0.2)
          for e in (0.1, 0.2) for sq in (1.0, 2.0, 4.0)]
    up = [(k, b0, db, e, sq) for k in (3, 5, 8)
          for b0 in (0.05, 0.1, 0.2, 0.4) for db in (0.0, 0.1, 0.2)
          for e in (0.1, 0.2) for sq in (1.0, 2.0, 4.0)]
    _, okt = fit('порог            ', group_threshold, tp, po, nsub, a.reps)
    _, oku = fit('взвеш. полезность', group_utility, up, po, nsub, a.reps)

    print()
    if okt and not oku:
        print('  РАЗЛИЧАЕТ: проходит только порог')
    elif oku and not okt:
        print('  РАЗЛИЧАЕТ: проходит только полезность')
    elif okt and oku:
        print('  обе проходят — семи признаков всё ещё не хватает')
    else:
        print('  НИ ОДНА не проходит: обе модели опровергнуты на этих данных')
    if okt:
        s = sorted({(pr[1], pr[2]) for _, _, pr, _, _ in okt})
        print(f'  строгость у выживших пороговых (среднее, разброс): {s}')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
