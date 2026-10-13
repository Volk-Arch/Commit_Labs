#!/usr/bin/env python3
"""Связанный ансамбль: exp(-beta*S) при закреплённом числе связанных пар.

ПОВОД. weight-doc.md: свободный exp(-beta*S) окна не имеет. При beta<=0.3
цепь инертна, при beta=1 действие съезжает без предела (-252 -> -1446) и
мир вырождается (доля пар 0.18 -> 0.31, размерность 3.3 -> 2.6). Причина
измерена: понизить действие глобальным уплотнением порядка можно на
тысячи, а локальным стягиванием к объекту — на единицы. Гравитационный
отклик утоплен в глобальной неустойчивости.

ИДЕЯ ШАГА. Отобрать у цепи глобальную ручку: закрепить полное число
связанных пар. Вырождение тогда запрещено по построению, и локальный
отклик остаётся единственным, что может произойти.

ЛОВУШКА, КОТОРУЮ НАДО ОБОЙТИ. Доля связанных пар и размерность
Мирхейма–Мейера — одна и та же величина (d восстанавливается из f).
Закрепив f, я сделаю глобальную размерность тривиально верной, и она
перестанет быть диагностом. Поэтому мерится РАЗМЕРНОСТЬ ПО
ПОДИНТЕРВАЛАМ: внутри интервала Александрова между случайной связанной
парой. Глобальное закрепление её не фиксирует, значит локальное
вырождение будет видно.

Дополнительные независимые диагносты: высота и спектральная мера
(положительная = конусная подпись).

ВОПРОСЫ.
  1. Остаётся ли мир манифолдоподобным при больших beta?
  2. Притягивает ли вес фон к локальному объекту — против рассыпанного
     контроля той же массы?

  python3 mccon.py --pilot
  python3 mccon.py --sweeps 80
"""
import argparse
import math
import time

import numpy as np

from core import mm_dim, height, spectral
from mc import World
from mcdiam import diamond, axis_tube, in_diamond, t_for


def local_dim(R, rs, nsamp=40, nmin=25):
    """Размерность по подинтервалам — независима от глобальной доли пар."""
    N = len(R)
    ii, jj = np.nonzero(R)
    if len(ii) == 0:
        return float('nan'), 0
    pick = rs.choice(len(ii), min(len(ii), 25 * nsamp), replace=False)
    out = []
    for p in pick:
        i, j = int(ii[p]), int(jj[p])
        idx = np.flatnonzero(R[i, :] & R[:, j])      # события между j и i
        m = len(idx)
        if m < nmin:
            continue
        sub = R[np.ix_(idx, idx)]
        f = float(sub.sum() / (m * (m - 1) / 2))
        d = mm_dim(f)
        if np.isfinite(d):
            out.append(d)
        if len(out) >= nsamp:
            break
    return (float(np.mean(out)) if out else float('nan')), len(out)


def diag(w, rs):
    n = len(w.R)
    f = w.cnt / (n * (n - 1) / 2)
    ld, k = local_dim(w.R, rs)
    return f, mm_dim(f), ld, k, height(w.R)[0], spectral(w.R), float(w.S)


def chain(obj, N, T, beta, eps, sweeps, rs, step, tol, edges, log=None):
    M = len(obj)
    w = World(np.vstack([obj, diamond(N, T, rs)]), eps)
    target = w.cnt
    lim = max(1, int(round(tol * target)))
    prof, pc, acc, rej_c = np.zeros(len(edges) - 1), 0, 0, 0
    if log is not None:
        log(0, diag(w, rs), 1.0, 0.0)
    for sw in range(1, sweeps + 1):
        for _ in range(N):
            i = M + int(rs.integers(N))
            xi = w.X[i] + step * rs.normal(size=4)
            if not in_diamond(xi, T):
                continue
            st = w.move(i, xi)
            if abs(w.cnt - target) > lim:          # связь нарушена
                w.undo(st)
                rej_c += 1
                continue
            dS = w.S - st[8]
            if dS <= 0 or rs.random() < math.exp(-beta * dS):
                acc += 1
            else:
                w.undo(st)
        if sw >= sweeps // 2:
            r = np.linalg.norm(w.X[M:, 1:], axis=1)
            prof += np.bincount(np.digitize(r, edges) - 1,
                                minlength=len(edges) - 1)[:len(edges) - 1]
            pc += 1
        if log is not None and (sw % max(1, sweeps // 6) == 0
                                or sw == sweeps):
            log(sw, diag(w, rs), acc / (sw * N), rej_c / (sw * N))
    return prof / max(pc, 1), acc / (sweeps * N), target, w.cnt


def main():
    ap = argparse.ArgumentParser(description='Связанный ансамбль')
    ap.add_argument('--n', type=int, default=560)
    ap.add_argument('--m', type=int, default=40)
    ap.add_argument('--rad', type=float, default=0.03)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--sweeps', type=int, default=80)
    ap.add_argument('--step', type=float, default=0.15)
    ap.add_argument('--tol', type=float, default=0.004)
    ap.add_argument('--betas', type=float, nargs='+', default=[1.0, 3.0, 10.0])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--pilot', action='store_true')
    a = ap.parse_args()

    t0 = time.time()
    T = t_for(a.n + a.m)
    edges = np.array([0.0, 0.10, 0.18, 0.27, 0.38, 0.5, 99.0])
    print(f'ромб T={T:.3f}, N={a.n+a.m} ({a.m} закреплено), eps={a.eps}')
    print(f'связь: число связанных пар держится в ±{a.tol:.1%}')
    print(f'развёрток {a.sweeps}, шаг {a.step}')
    print()
    print('ХОД ЦЕПИ. Доля пар закреплена, поэтому глобальная размерность '
          'тривиальна;')
    print('смотреть надо на РАЗМЕРНОСТЬ ПО ПОДИНТЕРВАЛАМ, высоту и спектр.')

    betas = a.betas[:1] if a.pilot else a.betas
    sweeps = 6 if a.pilot else a.sweeps
    for beta in [0.0] + list(betas):
        print(f'\n  beta = {beta}')
        print(f"{'развёртка':>11}{'доля пар':>10}{'разм.глоб':>11}"
              f"{'разм.подинт':>13}{'(выборок)':>11}{'высота':>8}"
              f"{'спектр':>9}{'S':>10}{'принят':>8}{'откл.связью':>13}")

        def log(sw, d, ac, rc):
            ld = f'{d[2]:.3f}' if np.isfinite(d[2]) else '  —  '
            print(f'{sw:>11}{d[0]:>10.4f}{d[1]:>11.3f}{ld:>13}'
                  f'{d[3]:>11}{d[4]:>8.0f}{d[5]:>9.4f}{d[6]:>10.1f}'
                  f'{ac:>8.2f}{rc:>13.2f}', flush=True)

        rs = np.random.default_rng(1234)
        _, _, tg, fin = chain(axis_tube(a.m, a.rad, T, rs), a.n, T, beta,
                              a.eps, sweeps, rs, a.step, a.tol, edges,
                              log=log)
        print(f'    связанных пар: цель {tg}, в конце {fin} '
              f'({(fin-tg)/tg:+.4%})')
    if a.pilot:
        print(f'\n  время {time.time()-t0:.0f} с')
        return

    print()
    print('ПРИТЯГИВАЕТ ЛИ — плотность фона по |x|, отнесённая к beta=0')
    print(f"{'мир':>10}{'beta':>6}{'принят':>8}"
          + ''.join(f'{f"<{e}":>8}' for e in edges[1:-1]))
    for lbl, mk in (('трубка', True), ('контроль', False)):
        ref = None
        for beta in [0.0] + list(a.betas):
            ps, ac = [], []
            for t in range(a.seeds):
                rs = np.random.default_rng(1234 + 31 * t)
                obj = (axis_tube(a.m, a.rad, T, rs) if mk
                       else diamond(a.m, T, rs))
                p, r, _, _ = chain(obj, a.n, T, beta, a.eps, a.sweeps,
                                   rs, a.step, a.tol, edges)
                ps.append(p)
                ac.append(r)
            A = np.array(ps)
            P, E = A.mean(0), A.std(0) / math.sqrt(len(A))
            if ref is None:
                ref = P.copy()
            Q = P / np.maximum(ref, 1e-9)
            QE = E / np.maximum(ref, 1e-9)
            k = len(edges) - 2
            print(f'{lbl:>10}{beta:>6.1f}{np.mean(ac):>8.2f}'
                  + ''.join(f'{v:>8.3f}' for v in Q[:k]), flush=True)
            print(f"{'± ошибка':>24}"
                  + ''.join(f'{v:>8.3f}' for v in QE[:k]), flush=True)
        print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
