#!/usr/bin/env python3
"""Закон задержки без подгоняемых параметров.

Механизм, который виден из устройства измерения: цепь, идущая сквозь
массу, получает её внутреннюю высоту, но теряет то время, которое уже
было у фона в том же месте. Отсюда предсказание

    Δτ  =  h_масса  −  h_фон_внутри_массы

Все три величины порядковые. Свободных параметров нет.

«Фон внутри массы» определяется без координат: фоновые события
интервала, у которых есть событие массы и в прошлом, и в будущем.

Скан по радиусу шара проверяет закон целиком: при малом радиусе фона
внутри массы почти нет (h_фон ≈ 0, задержка максимальна), при большом
его много (h_фон велик, задержка должна исчезнуть). Масса M при этом
не меняется, и внутренняя высота массы от радиуса не зависит —
причинное множество масштабно-инвариантно.

Это же и контроль на связность, который убил три прежние попытки
(`matter-doc.md`): если эффект сводится к центральности добавленных
событий, он не может зависеть от радиуса при постоянных M и h_масса.

  python3 delaylaw.py --k 4
  python3 delaylaw.py --k 4 --n 100000 --pairs 40
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import rel_of
from chaingrav import chain_len

CENTER = 0.5
RADII = (0.02, 0.04, 0.07, 0.11, 0.16, 0.22)


def ball(M, k, r, rs):
    v = rs.normal(size=(M, k))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad = r * rs.random(M) ** (1.0 / k)
    return (CENTER + v * rad[:, None]).astype(np.float32)


def pick_pairs(Xb, r_max, n_pairs, rs):
    """Пары, накрывающие самый большой шар, но прилегающие к нему вплотную.

    Иначе интервал захватывает почти весь мир: событие с координатами
    около нуля даёт объём порядка N, и подматрица не влезает. Берём
    события, прижатые к границе шара — из них интервал минимален.
    """
    ia = np.flatnonzero((Xb < CENTER - r_max).all(1))
    ib = np.flatnonzero((Xb > CENTER + r_max).all(1))
    if len(ia) == 0 or len(ib) == 0:
        return []
    # чем больше максимальная координата у a, тем позже начинается интервал
    ia = ia[np.argsort(-Xb[ia].max(1))]
    ib = ib[np.argsort(Xb[ib].min(1))]
    pool = max(n_pairs * 3, n_pairs)
    ia, ib = ia[:pool], ib[:pool]
    na = min(n_pairs, len(ia), len(ib))
    return list(zip(rs.choice(ia, na, replace=False),
                    rs.choice(ib, na, replace=False)))


def one(Xb, ib, xa, xb, Xadd):
    """Все порядковые величины одной пары за один проход."""
    nb = len(ib)
    if Xadd is None:
        R = rel_of(np.vstack([xa[None], Xb[ib], xb[None]]))
        return dict(tau=chain_len(R) - 1)
    R = rel_of(np.vstack([xa[None], Xb[ib], Xadd, xb[None]]))
    tau = chain_len(R) - 1
    sl = slice(1 + nb, 1 + nb + len(Xadd))
    bg = slice(1, 1 + nb)
    h_add = chain_len(R[sl, sl])
    # фон внутри массы: есть событие массы и в прошлом, и в будущем
    # R[i,j] = j раньше i
    has_past = R[bg, sl].any(1)
    has_fut = R[sl, bg].any(0)
    inner = np.flatnonzero(has_past & has_fut)
    h_mix = chain_len(R[np.ix_(1 + inner, 1 + inner)]) if len(inner) else 0
    return dict(tau=tau, h_add=h_add, h_mix=h_mix, n_mix=len(inner))


def run(N, M, k, n_pairs, seed, vol_cap, reals):
    rs = np.random.default_rng(seed)
    Xb = rs.random((N, k)).astype(np.float32)
    # несколько реализаций массы на каждый радиус: иначе h_масса —
    # единичный случай, а не величина
    rb = np.random.default_rng(seed + 77)
    balls = {r: [ball(M, k, r, rb) for _ in range(reals)] for r in RADII}
    pairs = pick_pairs(Xb, max(RADII), n_pairs,
                       np.random.default_rng(seed + 1))
    acc = {r: [] for r in RADII}
    vols = []
    for a, b in pairs:
        xa, xb = Xb[a], Xb[b]
        m = (Xb > xa).all(1) & (Xb < xb).all(1)
        m[a] = m[b] = False
        ib = np.flatnonzero(m)
        if len(ib) > vol_cap or len(ib) < 100:
            continue
        t0 = one(Xb, ib, xa, xb, None)['tau']
        vols.append(len(ib))
        for r in RADII:
            for B in balls[r]:
                d = one(Xb, ib, xa, xb, B)
                acc[r].append((d['tau'] - t0, d['h_add'], d['h_mix'],
                               d['n_mix'], t0))
    return acc, np.array(vols, float)


def main():
    ap = argparse.ArgumentParser(description='Закон задержки без параметров')
    ap.add_argument('--n', type=int, default=60000)
    ap.add_argument('--m', type=int, default=200)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--volcap', type=int, default=3500)
    ap.add_argument('--reals', type=int, default=3,
                    help='реализаций массы на каждый радиус')
    a = ap.parse_args()

    t0 = time.time()
    acc, vols = run(a.n, a.m, a.k, a.pairs, a.seed, a.volcap, a.reals)
    if len(vols) == 0:
        print('пар не набралось: поднимите --volcap или --n')
        return
    print(f'N={a.n}, масса M={a.m}, k={a.k}, пар {len(vols)}, '
          f'V_bg {vols.mean():.0f}, M/V_bg {a.m/vols.mean():.3f}')
    print()
    print(f"{'радиус':>8}{'фона внутри':>13}{'h_масса':>10}{'h_фон':>8}"
          f"{'предсказ.':>11}{'измерено':>17}{'расхожд., σ':>13}")
    rows = []
    for r in RADII:
        A = np.array(acc[r], dtype=float)
        if len(A) == 0:
            continue
        dt, ha, hm, nm = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
        pred = np.maximum(ha - hm, 0)
        resid = dt - pred
        se = resid.std() / np.sqrt(len(resid))
        print(f'{r:>8.2f}{nm.mean():>13.1f}{ha.mean():>10.2f}{hm.mean():>8.2f}'
              f'{pred.mean():>11.2f}'
              f'{f"{dt.mean():.2f} ± {dt.std()/np.sqrt(len(dt)):.2f}":>17}'
              f'{abs(resid.mean())/se if se > 0 else float("nan"):>13.1f}')
        rows.append((pred, dt))
    if rows:
        P = np.concatenate([p for p, _ in rows])
        D = np.concatenate([d for _, d in rows])
        sl, ic = np.polyfit(P, D, 1)
        rr = np.corrcoef(P, D)[0, 1]
        print()
        print(f'  по всем радиусам сразу: измеренное = '
              f'{sl:.3f} * предсказанное + {ic:.3f},  r = {rr:.3f}')
        print(f'  закон без параметров требует наклон 1 и сдвиг 0')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
