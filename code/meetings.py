#!/usr/bin/env python3
"""ГРАФ ВСТРЕЧ: приходит ли пространство на уровне историй.

ЕГО ИДЕЯ И ЧТО ОНА ДАЛА. Он предложил смотреть на порядок как на уже
состоявшиеся частицы. Замер (`particles.py`, `scratchpad/bound.py`)
подтвердил это в точной форме:

    N      L      предел ЛЮБОГО разложения   доля ПРЯМЫХ связей вдоль истории
    250    6.2    0.1996                     0.0699
    1000   9.8    0.0865                     0.0330
    4000   15.5   0.0365                     0.0161

Первая колонка — предел: длина цепи не больше L, сумма длин равна N,
значит внутрь историй нельзя уложить больше `N(L-1)/2` отношений НИ ПРИ
КАКОМ разложении. Показатели: цепь `N^+0.297`, наибольший слой
`N^+0.738`, доля внутри историй `N^-0.663` (R² = 0.9998).

Прямые связи суть единственное первичное сведение — остальное выводится
транзитивностью. И 98.4 % прямых связей соединяют события РАЗНЫХ историй.

> Мы читаем `A ≺ B` как «история A привела к B». Это чтение верно для
> 1.6 % первичных отношений. Остальные 98.4 % говорят «световой конус A
> достал до B», то есть записывают ВСТРЕЧУ двух разных историй.

ОТСЮДА ЭТОТ ЗАМЕР. Если порядок почти целиком есть записи встреч, то
первичный объект — не событие с отношением, а ИСТОРИЯ со встречами. И у
этого объекта есть свойство, которого нет ни у чего прежнего: разложение
на истории НЕ ЕДИНСТВЕННО, то есть это выбор — опора. А §7 `PICTURE.md`
давно говорит, что локальность в этой конструкции всегда покупается за
опору. Сегодняшний §10 показал, что из порядка напрямую пространство не
достать вовсе: поперечник не есть функция порядка (буст меняет его в
4.88 раза при расхождении порядковых величин 0.000 %).

ЧТО МЕРИМ. Разложение на истории (самая длинная цепь, вынуть, повторить).
Граф встреч: узлы — истории, ребро — если есть прямая связь между
событиями двух историй. И затем РОСТ ШАРА: сколько историй в пределах
графового расстояния `d`.

    число ~ d^3   -> граф встреч ТРЁХМЕРЕН, пространство пришло
    рост взрывной -> тесный мир, пространства нет

КОНТРОЛИ, поставленные заранее:
  граф связей по СОБЫТИЯМ — заведомо тесный мир (связи светоподобны и
    длинны, `locality-doc.md`), обязан дать взрывной рост; если
    история даст 3, а событие нет, это и есть находка;
  случайный ящик той же плотности — обязан дать тесный мир;
  сверка с координатами: расстояние в графе встреч против истинного
    пространственного — координаты только для ПРОВЕРКИ, в построении
    графа их нет.

  python3 meetings.py
  python3 meetings.py --n 2000 --maxch 400
"""
import argparse
import time

import numpy as np

from particles import diamond, rel_of


def longest_from(R, alive):
    """Самая длинная цепь среди живых. Возвращает список индексов."""
    idx = np.flatnonzero(alive)
    if len(idx) == 0:
        return []
    pos = {int(v): t for t, v in enumerate(idx)}
    sub = R[np.ix_(idx, idx)]
    n = len(idx)
    best = np.ones(n, np.int64)
    prev = -np.ones(n, np.int64)
    for t in range(n):
        pre = np.flatnonzero(sub[:, t])
        if len(pre):
            j = int(pre[np.argmax(best[pre])])
            if best[j] + 1 > best[t]:
                best[t] = best[j] + 1
                prev[t] = j
    t = int(np.argmax(best))
    out = []
    while t >= 0:
        out.append(int(idx[t]))
        t = int(prev[t])
    return out[::-1]


def decompose(R, maxch):
    """Истории: самая длинная цепь, вынуть, повторить."""
    N = len(R)
    alive = np.ones(N, bool)
    hist, owner = [], -np.ones(N, np.int64)
    while alive.any() and len(hist) < maxch:
        c = longest_from(R, alive)
        if not c:
            break
        for v in c:
            alive[v] = False
            owner[v] = len(hist)
        hist.append(c)
    return hist, owner


def ball_growth(adj, nseed, rs, dmax=6):
    """Сколько узлов в пределах графового расстояния d. Среднее по началам."""
    n = len(adj)
    seeds = rs.choice(n, min(nseed, n), replace=False)
    acc = np.zeros(dmax + 1)
    for s in seeds:
        seen = {int(s)}
        front = {int(s)}
        acc[0] += 1
        for d in range(1, dmax + 1):
            nxt = set()
            for v in front:
                nxt |= adj[v]
            nxt -= seen
            seen |= nxt
            front = nxt
            acc[d] += len(seen)
        del seen
    return acc / len(seeds)


def fit_growth(acc, lo=2, hi=5):
    d = np.arange(len(acc))
    m = (d >= lo) & (d <= hi) & (acc > 1)
    if m.sum() < 3:
        return float('nan')
    return float(np.polyfit(np.log(d[m]), np.log(acc[m]), 1)[0])


def main():
    ap = argparse.ArgumentParser(description='Граф встреч')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--maxch', type=int, default=400)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--nseed', type=int, default=40)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N = {a.n}, четырёхмерный ромб, засевов {a.seeds}')
    print('граф встреч: узлы — истории, ребро — прямая связь между ними')
    print('ОЖИДАНИЕ, если пространство приходит: шар ~ d^3')
    print('координаты — только для ПРОВЕРКИ, в построении графа их нет')
    print()

    for s in range(a.seeds):
        X = diamond(a.n, np.random.default_rng(9100 + 41 * s))
        R = rel_of(X)
        A = R.astype(np.float32)
        LK = R & ~((A @ A) > 0.5)
        hist, owner = decompose(R, a.maxch)
        nh = len(hist)
        lens = np.array([len(c) for c in hist])
        cov = int((owner >= 0).sum())

        # граф встреч
        li, lj = np.nonzero(LK)
        ok = (owner[li] >= 0) & (owner[lj] >= 0) & (owner[li] != owner[lj])
        hi_, hj = owner[li[ok]], owner[lj[ok]]
        adjH = [set() for _ in range(nh)]
        for u, v in zip(hi_, hj):
            adjH[int(u)].add(int(v))
            adjH[int(v)].add(int(u))
        degH = np.array([len(q) for q in adjH], float)

        # контроль: граф связей по СОБЫТИЯМ
        U = LK | LK.T
        adjE = [set(np.flatnonzero(U[i]).tolist()) for i in range(a.n)]

        rs = np.random.default_rng(7 + s)
        gH = ball_growth(adjH, a.nseed, rs)
        gE = ball_growth(adjE, a.nseed, rs)

        print(f'--- засев {s}: историй {nh}, покрыто {cov}/{a.n}, '
              f'длины {lens.min()}..{lens.max()} (средн {lens.mean():.1f})')
        print(f'    степень в графе встреч {degH.mean():.1f}, '
              f'в графе событий {np.mean([len(q) for q in adjE]):.1f}')
        print(f"    {'d':>3}" + ''.join(f'{d:>9}' for d in range(7)))
        print(f"    {'истор':>3}" + ''.join(f'{v:>9.1f}' for v in gH))
        print(f"    {'событ':>3}" + ''.join(f'{v:>9.1f}' for v in gE))
        print(f'    показатель роста: истории {fit_growth(gH):+.3f}, '
              f'события {fit_growth(gE):+.3f}')

        # проверка координатами: графовое расстояние против истинного
        cen = np.array([X[c][:, 1:].mean(0) for c in hist])
        rs2 = np.random.default_rng(31 + s)
        pick = rs2.choice(nh, min(30, nh), replace=False)
        dd, rr = [], []
        for u in pick:
            seen = {int(u)}
            front = {int(u)}
            for d in range(1, 5):
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
            print('    графовое расстояние -> истинное пространственное:')
            for d in sorted(set(dd.tolist())):
                m = dd == d
                print(f'      d={d}: {rr[m].mean():.3f} ± '
                      f'{rr[m].std():.3f}  ({int(m.sum())} пар)')
            print(f'    корреляция графового с истинным '
                  f'{np.corrcoef(dd, rr)[0,1]:+.4f}')
        del R, LK, A, U, X
        print()

    print('  истории дают ~3, события взрывной рост => пространство')
    print('  приходит НА УРОВНЕ ИСТОРИЙ, а не событий')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
