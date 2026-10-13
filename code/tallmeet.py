#!/usr/bin/env python3
"""ВЫСОКИЙ МИР: граф встреч, когда истории действительно длинные.

ПОЧЕМУ ПЕРВЫЙ ЗАМЕР НЕ ОТВЕТИЛ. `meetings.py` строил граф встреч в ромбе
и получил плотный тесный мир: степень 74.5 в графе из 350 узлов, `d = 2`
достаёт 331 из 349, рост шара +0.055 — места для роста просто нет.

Причина не в идее, а в геометрии. Истории вышли длиной 3.1 события при
максимуме 9: это не траектория, а крупинка. Триста пятьдесят крупинок в
ромбе высотой десять длин дискретности видят друг друга все.

Но граф встреч ПРОСТРАНСТВО ЗНАЕТ, и это он показал:

    графовое d=1 -> истинное 0.266 ± 0.099
             d=2 -> 0.460 ± 0.132
             d=3 -> 0.662 ± 0.135      корреляция +0.63 и +0.58

ЧТО ИСПРАВЛЕНО. Мир высокий и узкий: цилиндр `|x| <= R`, `t` в `[0, T]`
при `T >> R`. Тогда истории тянутся во всю высоту, их число равно ширине
в единицах дискретности, и у графа встреч появляется диаметр. Это не
подгонка параметра, а геометрия, делающая вопрос отвечаемым — та же
стена, что в §7 `PICTURE.md`: всё местное живёт в 1–2 длинах
дискретности, а нужно десять.

Длина дискретности `ell = (V/N)^(1/4)`, `V = T·(4π/3)R³`. Печатаются
`T/ell` (высота в длинах) и `R/ell` (ширина), чтобы видеть, высокий ли
мир на самом деле, а не по замыслу.

ЧТО МЕРИМ:
  рост шара в графе встреч — ожидание `d³`, если пространство пришло;
  рост шара в графе СВЯЗЕЙ ПО СОБЫТИЯМ — заведомо тесный мир
    (`locality-doc.md`: связи светоподобны и длинны), контроль;
  сверка с координатами — графовое расстояние против истинного
    пространственного; координаты только для проверки, в построении
    графа их нет.

РАЗЛОЖЕНИЕ на истории жадное, первым подходящим по времени: в высоком
узком мире оно само даёт длинные цепи, и это печатается.

  python3 tallmeet.py
  python3 tallmeet.py --n 4000 --t 6 --r 0.45
"""
import argparse
import math
import time

import numpy as np

from meetings import ball_growth, fit_growth


def tall(N, T, R, rs):
    """Цилиндр: t в [0,T], |x| <= R. Равномерно."""
    X = np.empty((N, 4))
    X[:, 0] = rs.uniform(0, T, N)
    v = rs.normal(size=(N, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    X[:, 1:] = v * R * rs.random((N, 1)) ** (1.0 / 3.0)
    return X[np.argsort(X[:, 0])]


def greedy_long(R):
    """Истории: первой подходящей по времени. В высоком мире даёт длинные."""
    N = len(R)
    ends, seqs = [], []
    owner = -np.ones(N, np.int64)
    for i in range(N):
        put = -1
        for c in range(len(ends)):
            if R[ends[c], i]:
                put = c
                break
        if put < 0:
            ends.append(i)
            seqs.append([i])
            owner[i] = len(seqs) - 1
        else:
            ends[put] = i
            seqs[put].append(i)
            owner[i] = put
    return seqs, owner


def main():
    ap = argparse.ArgumentParser(description='Граф встреч в высоком мире')
    ap.add_argument('--n', type=int, default=3500)
    ap.add_argument('--t', type=float, default=6.0)
    ap.add_argument('--r', type=float, default=0.45)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--nseed', type=int, default=40)
    a = ap.parse_args()

    t0 = time.time()
    vol = a.t * (4.0 * math.pi / 3.0) * a.r ** 3
    ell = (vol / a.n) ** 0.25
    print(f'N = {a.n}, цилиндр T = {a.t}, R = {a.r}')
    print(f'длина дискретности ell = {ell:.4f}  =>  высота {a.t/ell:.1f} '
          f'длин, ширина {a.r/ell:.1f} длин')
    print('ОЖИДАНИЕ, если пространство приходит: шар в графе встреч ~ d³')
    print('контроль: граф связей по событиям — заведомо тесный мир')
    print()

    for s in range(a.seeds):
        rs = np.random.default_rng(9500 + 53 * s)
        X = tall(a.n, a.t, a.r, rs)
        dt = X[:, 0][None, :] - X[:, 0][:, None]
        dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=2)
        Rl = (dt > 0) & (dt > dx)
        del dt, dx
        A = Rl.astype(np.float32)
        LK = Rl & ~((A @ A) > 0.5)
        del A
        seqs, owner = greedy_long(Rl)
        lens = np.array([len(q) for q in seqs])
        nh = len(seqs)

        li, lj = np.nonzero(LK)
        ok = owner[li] != owner[lj]
        hi_, hj = owner[li[ok]], owner[lj[ok]]
        adjH = [set() for _ in range(nh)]
        for u, v in zip(hi_, hj):
            adjH[int(u)].add(int(v))
            adjH[int(v)].add(int(u))
        degH = np.array([len(q) for q in adjH], float)

        U = LK | LK.T
        adjE = [set(np.flatnonzero(U[i]).tolist()) for i in range(a.n)]
        del U, LK, Rl

        rsg = np.random.default_rng(11 + s)
        gH = ball_growth(adjH, a.nseed, rsg, dmax=8)
        gE = ball_growth(adjE, a.nseed, rsg, dmax=8)

        print(f'--- засев {s}: историй {nh}, длины {lens.min()}..'
              f'{lens.max()} (средн {lens.mean():.1f}), '
              f'степень встреч {degH.mean():.1f}')
        print(f"    {'d':>6}" + ''.join(f'{d:>9}' for d in range(9)))
        print(f"    {'истор':>6}" + ''.join(f'{v:>9.1f}' for v in gH))
        print(f"    {'событ':>6}" + ''.join(f'{v:>9.1f}' for v in gE))
        print(f'    показатель роста: ИСТОРИИ {fit_growth(gH, 1, 5):+.3f}, '
              f'события {fit_growth(gE, 1, 5):+.3f}')

        cen = np.array([X[q][:, 1:].mean(0) for q in seqs])
        rs2 = np.random.default_rng(29 + s)
        pick = rs2.choice(nh, min(40, nh), replace=False)
        dd, rr = [], []
        for u in pick:
            seen, front = {int(u)}, {int(u)}
            for d in range(1, 6):
                nxt = set()
                for v in front:
                    nxt |= adjH[v]
                nxt -= seen
                seen |= nxt
                front = nxt
                for v in nxt:
                    dd.append(d)
                    rr.append(float(np.linalg.norm(cen[v] - cen[u])))
        if dd:
            dd, rr = np.array(dd), np.array(rr)
            print('    графовое -> истинное пространственное '
                  f'(ширина мира {2*a.r:.2f}):')
            for d in sorted(set(dd.tolist())):
                m = dd == d
                print(f'      d={d}: {rr[m].mean():.4f} ± {rr[m].std():.4f}'
                      f'  ({int(m.sum())} пар)')
            print(f'    корреляция {np.corrcoef(dd, rr)[0,1]:+.4f}')
        del X
        print()

    print('  истории ~3, события взрывной рост => пространство приходит')
    print('  НА УРОВНЕ ИСТОРИЙ. Иначе — граф встреч тоже тесный мир.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
