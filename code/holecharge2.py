#!/usr/bin/env python3
"""Заряд разрежения, попытка вторая: трубка вдоль опорного направления.

ПОЧЕМУ ПЕРВАЯ ПОПЫТКА НЕ СОСТОЯЛАСЬ (holecharge.py). Разрежение шло
вдоль оси времени, значит охватывать его надо петлёй в ПРОСТРАНСТВЕННОЙ
плоскости. А директорное поле axis_of(i0, j) гладко только в плоскости,
ПОПЕРЕЧНОЙ направлению i0->j. В пространственной плоскости при
фиксированном времени неоднозначность рёбер вышла 0.85..0.95 против
0.038 в хорошей плоскости — при такой выборке индекс есть шум, и все три
мира дали одинаковые числа.

Это само по себе содержательный вывод: поле НЕ ЛОКАЛЬНОЕ, оно
организовано вокруг опорного события.

ПОЧИНКА. Вести разрежение вдоль самого отрезка i0 -> j0, который
времениподобен. Тогда петля в поперечной плоскости у j0 охватывает
трубку естественно, и поле в этой плоскости гладко.

ПАРНОЕ СРАВНЕНИЕ обязательно: charge-doc.md уже показал, что у заплаты
есть базовая намотка около +0.30, так что мерить надо РАЗНОСТЬ на одном
и том же засеве.

Три мира на каждый засев, одна и та же опора, один и тот же центр:
    без дырки            — базовая намотка
    трубка вдоль i0->j0  — проверяемое
    столько же удалено случайно — контроль на «просто меньше событий»

УСЛОВИЯ ПОЛОЖИТЕЛЬНОГО ИСХОДА, названные заранее:
    1. разность индекса трубка минус без дырки значимо ненулевая;
    2. случайное удаление такой разности НЕ даёт;
    3. неоднозначность у трубки НЕ выше, чем без дырки;
    4. индекс выживает при огрублении петли вдвое.

  python3 holecharge2.py
  python3 holecharge2.py --rad 0.18 --rhos 0.26 0.34 --seeds 20
"""
import argparse
import math
import time

import numpy as np

from topo import diamond, cone_rel, axis_of, plane_of
from holecharge import unwrap_pi


def tube_mask(X, p0, p1, rad):
    """Маска событий ВНЕ трубки радиуса rad вокруг отрезка p0->p1.

    Расстояние берётся поперечное: проекция на отрезок снимается.
    """
    a = p1 - p0
    n2 = float(a @ a)
    if n2 < 1e-12:
        return np.ones(len(X), bool)
    s = ((X - p0) @ a) / n2
    perp = (X - p0) - np.outer(s, a)
    d = np.linalg.norm(perp, axis=1)
    inside = (d < rad) & (s > 0.03) & (s < 0.97)
    return ~inside


def loop_index(X, R, i0, cand, ctr4, rho, nsteps, minsize, projmin, u, v):
    """Индекс петли радиуса rho вокруг ctr4 в плоскости (u,v)."""
    vals = []
    P = X[cand]
    for s in range(nsteps):
        th = 2 * math.pi * s / nsteps
        tgt = ctr4 + rho * (math.cos(th) * u + math.sin(th) * v)
        d = np.linalg.norm(P - tgt, axis=1)
        got = None
        for idx in np.argsort(d)[:18]:
            j = int(cand[idx])
            ax = axis_of(X, R, i0, j, minsize)
            if ax is None:
                continue
            p = np.array([ax @ u, ax @ v])
            if np.linalg.norm(p) < projmin:
                continue
            got = math.atan2(p[1], p[0])
            break
        if got is None:
            return None, None
        vals.append(got)
    tot, amb = 0.0, 0.0
    for t in range(nsteps):
        dd = unwrap_pi(vals[(t + 1) % nsteps] - vals[t])
        tot += dd
        amb = max(amb, abs(dd) / (math.pi / 2))
    return tot / (2 * math.pi), amb


def main():
    ap = argparse.ArgumentParser(description='Заряд разрежения, попытка 2')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.18)
    ap.add_argument('--rhos', type=float, nargs='+', default=[0.26, 0.34])
    ap.add_argument('--nsteps', type=int, default=24)
    ap.add_argument('--minsize', type=int, default=20)
    ap.add_argument('--projmin', type=float, default=0.25)
    ap.add_argument('--seeds', type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    print(f'ромб T={a.t}, N={a.n}; разрежение радиуса {a.rad} '
          f'вдоль отрезка i0->j0')
    print(f'петли в ПОПЕРЕЧНОЙ плоскости у j0, шагов {a.nsteps}, '
          f'радиусы {a.rhos}')
    print('сравнение ПАРНОЕ: один засев, одна опора, один центр')
    print(f'засевов {a.seeds}')
    print()

    for rho in a.rhos:
        acc = {k: [] for k in ('без', 'трубка', 'случ')}
        amb = {k: [] for k in ('без', 'трубка', 'случ')}
        rob = {k: [] for k in ('без', 'трубка', 'случ')}
        for t in range(a.seeds):
            rs = np.random.default_rng(3301000 + 7919 * t)
            X0 = diamond(a.n, a.t, rs)
            R0 = cone_rel(X0)
            early = X0[:, 0] < 0.22 * a.t
            if not early.any():
                del R0, X0; continue
            i0 = int(np.argmax(R0.sum(0) * early))
            fut = np.flatnonzero(R0[:, i0] & (X0[:, 0] > 0.62 * a.t))
            if len(fut) < 5:
                del R0, X0; continue
            j0 = int(fut[np.argmax(R0[fut, i0] * 1.0
                                   + (-np.linalg.norm(
                                       X0[fut][:, 1:] - X0[i0, 1:],
                                       axis=1)))])
            try:
                u, v = plane_of(X0, i0, j0)
            except Exception:
                del R0, X0; continue
            p0, p1 = X0[i0].copy(), X0[j0].copy()
            ctr4 = X0[j0].copy()
            keep = tube_mask(X0, p0, p1, a.rad)
            k = int((~keep).sum())
            if k < 12:
                del R0, X0; continue
            worlds = {}
            worlds['без'] = X0
            worlds['трубка'] = X0[keep]
            ii = rs.choice(len(X0), len(X0) - k, replace=False)
            worlds['случ'] = X0[np.sort(ii)]
            ok = True
            res = {}
            for nm, X in worlds.items():
                # опора и центр ищутся заново по БЛИЖАЙШЕМУ событию,
                # чтобы геометрия сравнения совпадала
                R = R0 if nm == 'без' else cone_rel(X)
                d0 = np.linalg.norm(X - p0, axis=1)
                ii0 = int(np.argmin(d0))
                cand = np.flatnonzero(R[:, ii0])
                if len(cand) < 150:
                    ok = False
                    if nm != 'без':
                        del R
                    break
                vv, am = loop_index(X, R, ii0, cand, ctr4, rho, a.nsteps,
                                    a.minsize, a.projmin, u, v)
                v2, _ = loop_index(X, R, ii0, cand, ctr4, rho,
                                   a.nsteps // 2, a.minsize, a.projmin,
                                   u, v)
                if nm != 'без':
                    del R
                if vv is None:
                    ok = False
                    break
                res[nm] = (vv, am, (abs(vv - v2) < 1e-6)
                           if v2 is not None else None)
            del R0, X0
            if not ok or len(res) < 3:
                continue
            for nm in acc:
                acc[nm].append(res[nm][0])
                amb[nm].append(res[nm][1])
                if res[nm][2] is not None:
                    rob[nm].append(res[nm][2])
        n = len(acc['без'])
        print(f'rho = {rho}:  годных засевов {n} (удалено ~{k} событий)')
        if n < 3:
            print('   мало данных\n')
            continue
        print(f"{'мир':>10}{'<индекс>':>11}{'±':>8}{'|инд|>=1/2':>12}"
              f"{'неоднозн':>10}{'огрубл.совп':>13}")
        for nm in ('без', 'трубка', 'случ'):
            A = np.array(acc[nm])
            print(f'{nm:>10}{A.mean():>+11.4f}'
                  f'{A.std()/math.sqrt(len(A)):>8.4f}'
                  f'{(np.abs(np.round(A*2)/2) >= 0.5).mean():>12.3f}'
                  f'{np.mean(amb[nm]):>10.3f}'
                  f'{(np.mean(rob[nm]) if rob[nm] else float("nan")):>13.3f}')
        for nm in ('трубка', 'случ'):
            d = np.array(acc[nm]) - np.array(acc['без'])
            se = d.std() / math.sqrt(len(d))
            print(f'   ПАРНО {nm} минус без: {d.mean():+.4f} ± {se:.4f}'
                  f'   ({abs(d.mean())/max(se,1e-12):.1f}σ),'
                  f'  различий {int((np.abs(d) > 1e-6).sum())}/{len(d)}')
        print(flush=True)

    print('  положительно ТОЛЬКО если: трубка даёт значимую разность,')
    print('  случайное удаление не даёт, неоднозначность не выше,')
    print('  и огрубление совпадает')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
