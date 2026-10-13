#!/usr/bin/env python3
"""Ломает ли ТОПОЛОГИЧЕСКИЙ заряд антикорреляцию сохранения и влияния.

АРХИВНЫЙ ФАКТ (`ORDER-final.md:363`). Для десяти величин измерены
сохранение (передача вдоль прямой связи за вычетом случайного уровня) и
влияние (связь с долей прямых в окрестности):

    заряд        0.300 / 0.269
    размах фаз   0.256 / 0.206
    прошлое      0.662 / 0.147
    время        0.815 / 0.096
    глубина      0.687 / 0.031
    корреляция (сохранение, влияние) = -0.657

> Чем лучше величина сохраняется, тем меньше она влияет. А для
> уравнения поля нужна величина, которая сохраняется И влияет.

ПОЧЕМУ ЕСТЬ ЛАЗЕЙКА. Все десять архивных величин — ГЛАДКИЕ счётные
функции события, и механизм антикорреляции тот же, что измерен сегодня
на фазах: хорошо сохраняющаяся вдоль цепи величина есть функция
положения на цепи, то есть гладкая; гладкое локально не меняется, значит
не влияет.

    сохраняющееся = телескопирующееся = гладкое = не влияет

Три независимых появления: архивная антикорреляция -0.657; архивная
нулевая холономия у «разности прошлого» против 8.7 сигмы у
калибровочной; сегодняшний показатель когерентности 2.000 против 1.004.

Выход виден прямо из формулировки: нужна величина, сохраняющаяся ТОЧНО,
но НЕ ГЛАДКО. Это определение топологического заряда — он сохраняется
абсолютно, почти всюду нуль, сосредоточен в точках. Под антикорреляцию
не попадает по построению. И такой у нас есть (`charge-doc.md`): индекс
дефекта квантован точно, масштабно-инвариантен (0.888), дисперсия суммы
0.452 от пуассоновской (5.1 сигмы) — рождается парами.

ЗДЕСЬ: индекс приписывается событию как индекс малой петли вокруг него в
плоскости, поперечной направлению на опору (как в topo.py), и мерится
теми же двумя величинами. Архивные величины пересчитываются тем же
кодом, чтобы линия антикорреляции была из ОДНОГО замера.

  python3 topoinfl.py
  python3 topoinfl.py --n 2500 --samp 300 --seeds 3
"""
import argparse
import math
import time

import numpy as np

from topo import diamond, cone_rel, axis_of, plane_of
from holecharge import unwrap_pi


def links_of(R):
    A = np.asarray(R, np.float32)
    return np.asarray(R & ((A @ A) == 0), bool)


def loop_index(X, R, i0, cand, j0, rho, nsteps, minsize, projmin):
    """Индекс директорного поля по петле радиуса rho вокруг события j0."""
    try:
        u, v = plane_of(X, i0, j0)
    except Exception:
        return None
    vals = []
    P = X[cand]
    for s in range(nsteps):
        th = 2 * math.pi * s / nsteps
        tgt = X[j0] + rho * (math.cos(th) * u + math.sin(th) * v)
        d = np.linalg.norm(P - tgt, axis=1)
        got = None
        for idx in np.argsort(d)[:14]:
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
            return None
        vals.append(got)
    tot = 0.0
    for t in range(nsteps):
        tot += unwrap_pi(vals[(t + 1) % nsteps] - vals[t])
    return tot / (2 * math.pi)


def main():
    ap = argparse.ArgumentParser(description='Влияние топологического заряда')
    ap.add_argument('--n', type=int, default=2500)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--samp', type=int, default=260)
    ap.add_argument('--pairs', type=int, default=220)
    ap.add_argument('--rho', type=float, default=0.10)
    ap.add_argument('--nsteps', type=int, default=24)
    ap.add_argument('--minsize', type=int, default=25)
    ap.add_argument('--projmin', type=float, default=0.25)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, шкал {a.k}; петля радиуса {a.rho}, шагов {a.nsteps}')
    print('сохранение = корр. по прямым связям минус по случайным парам')
    print('влияние    = |корр. с долей прямых связей у события|')
    print(f'засевов {a.seeds}, выборка {a.samp} событий и {a.pairs} связей')
    print()

    acc = {}
    for t in range(a.seeds):
        rs = np.random.default_rng(5401000 + 977 * t)
        X = diamond(a.n, a.t, rs)
        R = cone_rel(X)
        L = links_of(R)
        N = len(X)
        # шкалы конуса: время = сумма, заряд/размах = разброс
        u = rs.normal(size=(a.k, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        S = X[:, 0:1] - X[:, 1:] @ u.T
        rel_n = R.sum(1) + R.sum(0)
        lnk_n = L.sum(1) + L.sum(0)
        infl = lnk_n / np.maximum(rel_n, 1)          # доля прямых у события
        # высота
        h = np.zeros(N, np.int64)
        for i in np.argsort(R.sum(1), kind='stable'):
            i = int(i)
            pre = np.flatnonzero(R[i])
            if len(pre):
                h[i] = h[pre].max() + 1
        Q = {
            'время (сумма шкал)': S.sum(1),
            'размах шкал (заряд)': S.std(1),
            'прошлое': R.sum(1).astype(float),
            'глубина (высота)': h.astype(float),
        }
        # опора для директорного поля
        fut = R.sum(0)
        i0 = int(np.argmax(fut * (X[:, 0] < 0.25 * a.t)))
        cand = np.flatnonzero(R[:, i0])
        if len(cand) < 300:
            del R, L, X
            continue
        # индекс на выборке событий
        pool = rs.choice(cand, min(a.samp, len(cand)), replace=False)
        idx_val, idx_ev = [], []
        for j0 in pool:
            v = loop_index(X, R, i0, cand, int(j0), a.rho, a.nsteps,
                           a.minsize, a.projmin)
            if v is not None:
                idx_val.append(v)
                idx_ev.append(int(j0))
        idx_val = np.array(idx_val)
        idx_ev = np.array(idx_ev, int)
        # влияние
        for nm, q in Q.items():
            m = np.isin(np.arange(N), idx_ev)
            c = abs(np.corrcoef(q[m], infl[m])[0, 1])
            acc.setdefault(nm, {'infl': [], 'cons': []})['infl'].append(c)
        if len(idx_val) > 30:
            c = abs(np.corrcoef(idx_val, infl[idx_ev])[0, 1])
            acc.setdefault('ИНДЕКС ДЕФЕКТА',
                           {'infl': [], 'cons': []})['infl'].append(c)
        # сохранение: по прямым связям против случайных пар
        li, lj = np.where(L)
        pick = rs.choice(len(li), min(a.pairs * 8, len(li)), replace=False)
        for nm, q in Q.items():
            cl = np.corrcoef(q[li[pick]], q[lj[pick]])[0, 1]
            r1 = rs.integers(0, N, len(pick)); r2 = rs.integers(0, N, len(pick))
            cr = np.corrcoef(q[r1], q[r2])[0, 1]
            acc[nm]['cons'].append(abs(cl) - abs(cr))
        # сохранение индекса: пары связанных событий внутри выборки
        inset = {int(e): i for i, e in enumerate(idx_ev)}
        pa, pb = [], []
        for s in range(len(li)):
            i_, j_ = int(li[s]), int(lj[s])
            if i_ in inset and j_ in inset:
                pa.append(inset[i_]); pb.append(inset[j_])
            if len(pa) >= a.pairs:
                break
        if len(pa) > 20:
            cl = np.corrcoef(idx_val[pa], idx_val[pb])[0, 1]
            r1 = rs.integers(0, len(idx_val), len(pa))
            r2 = rs.integers(0, len(idx_val), len(pa))
            cr = np.corrcoef(idx_val[r1], idx_val[r2])[0, 1]
            acc['ИНДЕКС ДЕФЕКТА']['cons'].append(abs(cl) - abs(cr))
        print(f'   засев {t}: индексов {len(idx_val)}, '
              f'ненулевых {int((np.abs(np.round(idx_val*2)/2) >= 0.5).sum())}, '
              f'связанных пар в выборке {len(pa)}', flush=True)
        del R, L, X

    print()
    print(f"{'величина':>22}{'сохранение':>13}{'влияние':>11}{'замеров':>10}")
    pts = []
    for nm, d in acc.items():
        if not d['cons'] or not d['infl']:
            print(f'{nm:>22}{"мало данных":>13}')
            continue
        cs, inf = float(np.mean(d['cons'])), float(np.mean(d['infl']))
        print(f'{nm:>22}{cs:>13.3f}{inf:>11.3f}{len(d["infl"]):>10}')
        pts.append((nm, cs, inf))
    sm = [p for p in pts if p[0] != 'ИНДЕКС ДЕФЕКТА']
    if len(sm) >= 3:
        cc = np.corrcoef([p[1] for p in sm], [p[2] for p in sm])[0, 1]
        print(f'\n  корреляция (сохранение, влияние) по ГЛАДКИМ: {cc:+.3f}'
              f'   (архив −0.657)')
    ix = [p for p in pts if p[0] == 'ИНДЕКС ДЕФЕКТА']
    if ix and sm:
        _, cs, inf = ix[0]
        pred = np.polyfit([p[1] for p in sm], [p[2] for p in sm], 1)
        exp = np.polyval(pred, cs)
        print(f'  индекс: сохранение {cs:.3f}, влияние {inf:.3f}; '
              f'линия гладких предсказывает {exp:.3f}')
        print(f'  превышение над линией: {inf - exp:+.3f}')
        print('  заметно выше линии => антикорреляция СЛОМАНА')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
