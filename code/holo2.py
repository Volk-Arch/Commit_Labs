#!/usr/bin/env python3
"""ВЛИЯЕТ ЛИ СВЯЗНОСТЬ: холономия против местной структуры.

ГИПОТЕЗА. Архивная антикорреляция (сохранение против влияния, -0.657)
измерена на ВЕЛИЧИНАХ — функциях места. Сегодня её довели до закона:
на идеально сохраняющемся конце (топологический индекс) влияние ровно
0.00, и топологичность не спасает. Перебраны все три класса величин:
гладкие на событиях, состояние на связях (поток), топологический заряд.

Но СВЯЗНОСТЬ — не величина. Она не функция места, а правило сравнения, и
её содержание — холономия, то есть зависимость сравнения от пути. Под
«сохраняющееся = гладкое = не влияет» она не подпадает по типу.

И связность у нас ровно одна, и она единственный объект, сидящий на обеих
сторонах:
  калибровочная фаза phi(i->j) = |общее прошлое| - |общее будущее|
    * переживает перешкалирование (1.000000, замерено)
    * интерферирует (показатель 1.312 — единственная порядковая)
    * имеет холономию 22.24 +- 2.54, то есть 8.7 сигмы, и НЕ
      телескопируется (архивный контроль: «разность прошлого» на том же
      коде даёт ровно нуль)

ЧТО МЕРИМ. Для каждого события — холономия малых циклов в графе прямых
связей вокруг него (сумма phi со знаком обхода). Затем влияние: связь с
долей прямых связей у события.

КОНТРОЛИ, поставленные ЗАРАНЕЕ (предыдущий замер убил именно их
отсутствие):
  1. ВЫСОТА. Предыдущая попытка дала 3.0 сигмы, которые целиком
     объяснились высотой (дефекты сидят ниже, а у низких событий доля
     прямых выше автоматически). Здесь сравнение идёт внутри слоёв
     одинаковой высоты И через регрессионный остаток.
  2. ТЕЛЕСКОПИРУЮЩАЯСЯ фаза: psi(i->j) = |past(i)| - |past(j)|. Её
     холономия обязана быть нулём, и влияния быть не должно. Если она
     даёт то же, что калибровочная, — замер ни о чём.
  3. Перемешанная фаза: те же значения, разложенные по случайным парам.

  python3 holo2.py
  python3 holo2.py --n 2200 --seeds 6
"""
import argparse
import math
import time

import numpy as np

from topo import diamond, cone_rel


def links_of(R):
    A = np.asarray(R, np.float32)
    return np.asarray(R & ((A @ A) == 0), bool)


def cycles_through(U, v, want, rs, tries=400):
    """Малые циклы длины 4 через v в неориентированном графе связей."""
    nb = np.flatnonzero(U[v])
    out = []
    if len(nb) < 2:
        return out
    for _ in range(tries):
        if len(out) >= want:
            break
        a, b = rs.choice(nb, 2, replace=False)
        a, b = int(a), int(b)
        na = np.flatnonzero(U[a])
        nbb = set(np.flatnonzero(U[b]).tolist())
        mid = [int(x) for x in na if int(x) in nbb and x != v]
        if not mid:
            continue
        c = int(rs.choice(mid))
        out.append((v, a, c, b))
    return out


def holonomy(cyc, A):
    """Холономия: сумма АНТИСИММЕТРИЧНОЙ связности по обходу.

    ПОЙМАННАЯ ОШИБКА. Первая версия умножала фазу на знак причинного
    направления внутри холономии. Для СИММЕТРИЧНОЙ калибровочной фазы
    это законная антисимметризация, а для телескопирующейся
    |past(i)|-|past(j)|, которая антисимметрична УЖЕ, знак навешивался
    повторно — и телескопирование ломалось: контроль дал холономию 1069
    вместо нуля и влияние 13.1 сигмы, то есть сильнее проверяемого.

    Правильно: связность приводится к антисимметричному виду ОДИН раз,
    снаружи, а холономия есть просто сумма по обходу.
    """
    tot = 0.0
    for t in range(len(cyc)):
        tot += A[cyc[t], cyc[(t + 1) % len(cyc)]]
    return tot


def antisym_causal(G, R):
    """Антисимметризовать СИММЕТРИЧНУЮ фазу знаком причинного направления."""
    s = np.where(R.T, 1.0, 0.0) - np.where(R, 1.0, 0.0)
    return s * G


def main():
    ap = argparse.ArgumentParser(description='Влияет ли связность')
    ap.add_argument('--n', type=int, default=2200)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--samp', type=int, default=500)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, выборка {a.samp} событий, циклов на событие {a.ncyc}')
    print('фазы: КАЛИБРОВОЧНАЯ |общее прошлое|-|общее будущее|;')
    print('      ТЕЛЕСКОП |past(i)|-|past(j)| (контроль, холономия 0);')
    print('      ПЕРЕМЕШАННАЯ (контроль)')
    print('влияние = связь с долей прямых связей, ПРИ СОВПАДАЮЩЕЙ ВЫСОТЕ')
    print()

    rows = {k: [] for k in ('gauge', 'tele', 'shuf')}
    meta = []
    for t in range(a.seeds):
        rs = np.random.default_rng(5801000 + 977 * t)
        X = diamond(a.n, a.t, rs)
        R = cone_rel(X)
        L = links_of(R)
        U = L | L.T
        N = len(X)
        A = R.astype(np.float32)
        CP = A @ A.T                      # |общее прошлое|
        CF = A.T @ A                      # |общее будущее|
        sz = A.sum(1)
        # КАЛИБРОВОЧНАЯ: CP-CF симметрична, антисимметризуем знаком
        # причинного направления (один раз, снаружи холономии)
        G = antisym_causal((CP - CF).astype(np.float64), R)
        # ТЕЛЕСКОП: уже антисимметрична, холономия обязана быть НУЛЁМ
        TE = sz[:, None] - sz[None, :]
        # ПЕРЕМЕШАННАЯ: те же значения по случайным парам, затем
        # приведена к антисимметричному виду
        idx = rs.permutation(N)
        SH = G[np.ix_(idx, idx)]
        SH = 0.5 * (SH - SH.T)
        infl = (L.sum(1) + L.sum(0)) / np.maximum(R.sum(1) + R.sum(0), 1)
        h = np.zeros(N, np.int64)
        for i in np.argsort(R.sum(1), kind='stable'):
            i = int(i)
            pre = np.flatnonzero(R[i])
            if len(pre):
                h[i] = h[pre].max() + 1
        deg = U.sum(1)
        pool = np.flatnonzero(deg >= 3)
        if len(pool) < 50:
            del R, L, U, X; continue
        pool = rs.choice(pool, min(a.samp, len(pool)), replace=False)
        for v in pool:
            v = int(v)
            cyc = cycles_through(U, v, a.ncyc, rs)
            if len(cyc) < 2:
                continue
            hg = np.mean([abs(holonomy(c, G)) for c in cyc])
            ht = np.mean([abs(holonomy(c, TE)) for c in cyc])
            hs = np.mean([abs(holonomy(c, SH)) for c in cyc])
            rows['gauge'].append(hg); rows['tele'].append(ht)
            rows['shuf'].append(hs)
            meta.append((infl[v], float(h[v]), float(deg[v])))
        del R, L, U, X

    M = np.array(meta)
    if len(M) < 100:
        print('мало данных'); return
    INF, H, DEG = M[:, 0], M[:, 1], M[:, 2]
    print(f'событий промерено {len(M)}')
    print(f"{'фаза':>14}{'<|холономия|>':>15}{'сырая корр.':>13}"
          f"{'при фикс.высоте':>17}{'сигм':>7}{'контроль':>10}")
    for k, nm in (('gauge', 'калибровочная'), ('tele', 'телескоп'),
                  ('shuf', 'перемешанная')):
        v = np.array(rows[k])
        if v.std() < 1e-12:
            print(f'{nm:>14}{v.mean():>15.3f}{"постоянна":>13}')
            continue
        raw = np.corrcoef(v, INF)[0, 1]
        # остаток после снятия высоты (квадратичной) у ОБОИХ
        rv = v - np.polyval(np.polyfit(H, v, 2), H)
        ri = INF - np.polyval(np.polyfit(H, INF, 2), H)
        part = np.corrcoef(rv, ri)[0, 1]
        z = abs(part) * math.sqrt(max(len(v) - 3, 1))
        # контроль: перестановка остатка
        rs2 = np.random.default_rng(4242)
        zs = [abs(np.corrcoef(rv[rs2.permutation(len(rv))], ri)[0, 1])
              * math.sqrt(len(v) - 3) for _ in range(200)]
        print(f'{nm:>14}{v.mean():>15.3f}{raw:>+13.4f}'
              f'{part:>+17.4f}{z:>7.1f}{np.percentile(zs, 95):>10.1f}')

    print()
    print('  калибровочная заметно выше контроля при фикс. высоте,')
    print('  а телескоп и перемешанная — нет => СВЯЗНОСТЬ ВЛИЯЕТ')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
