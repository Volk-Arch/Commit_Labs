#!/usr/bin/env python3
"""СОХРАНЯЕТСЯ ЛИ СУММАРНЫЙ ИНДЕКС: заряд это или шум.

ОТКРЫТЫЙ ВОПРОС АРХИВА (layer2-doc.md:191), никогда не измеренный.
Архив нашёл в поле поперечных осей дефекты с индексом -1/2 (0.033 из 390
обходов) и проверил двулистность — не прошла, «дефект локален, его можно
обойти». Но главное не измерялось: СОХРАНЯЕТСЯ ЛИ СУММА. Именно это
делает дефекты частицами — в физике суммарный топологический заряд
сохраняется, поэтому дефекты рождаются и гибнут только ПАРАМИ, и защита
не требует никакой динамики. А отбор ограничением — ровно то, чего нам
не хватает: вес отбирать геометрию не может (preclusion-doc.md), а
запрет сгладить дефект по одному — может.

ПОЧЕМУ НУЖНА СЕТКА, А НЕ СЛУЧАЙНЫЕ ПЕТЛИ. Сумма индексов малых петель
равна индексу большой ТОЖДЕСТВЕННО, если петли считаны по ОДНИМ И ТЕМ ЖЕ
значениям поля: внутренние ребра проходятся дважды в противоположных
направлениях и сокращаются, потому что unwrap(-x) = -unwrap(x). Значит
аддитивность сама по себе ничего не доказывает — она проверяет лишь, не
сломалось ли разворачивание угла. А вот ЧТО отличает заряд от шума:

  1. РАЗРЕЖЕННОСТЬ: доля клеток с ненулевым индексом мала.
  2. КВАНТОВАННОСТЬ: значения ±1/2, а |индекс| >= 1 редок.
  3. БАЛАНС ЗНАКОВ: плюсов столько же, сколько минусов (рождение
     парами). Архивные «одни минусы» были следствием ошибки знака,
     исправленной в topo.py.
  4. УСТОЙЧИВОСТЬ К МАСШТАБУ: дефект, найденный клеткой, обязан
     находиться и петлёй вокруг блока клеток. Шум — нет.
  5. ОДНОЗНАЧНОСТЬ РАЗВОРАЧИВАНИЯ: доля рёбер, где |dtheta| близко к
     pi/2, то есть поле недосэмплировано.

КОНТРОЛЬ: то же самое на СЛУЧАЙНОМ поле директоров на той же сетке. Там
обязаны провалиться пункты 1, 2, 4, 5 — аддитивность же выполнится и
там, что и доказывает её непригодность в качестве теста.

  python3 charge.py
  python3 charge.py --n 4000 --grid 9 --step 0.07
"""
import argparse
import math
import time

import numpy as np

from topo import diamond, cone_rel, axis_of, plane_of


def field_grid(X, R, i0, j0, u, v, g, step, minsize, cand, projmin):
    """Углы директора (mod pi) на сетке (2g+1)x(2g+1) в плоскости (u,v).

    Возвращает (углы, маска годности). Для каждой точки сетки берётся
    ближайшее подходящее событие.
    """
    n = 2 * g + 1
    th = np.full((n, n), np.nan)
    P = X[cand]
    for ia in range(n):
        for ib in range(n):
            tgt = (X[j0] + (ia - g) * step * u + (ib - g) * step * v)
            d = np.linalg.norm(P - tgt, axis=1)
            for idx in np.argsort(d)[:14]:
                j = int(cand[idx])
                ax = axis_of(X, R, i0, j, minsize)
                if ax is None:
                    continue
                p = np.array([ax @ u, ax @ v])
                if np.linalg.norm(p) < projmin:
                    continue
                th[ia, ib] = math.atan2(p[1], p[0])
                break
    return th


def unwrap_pi(d):
    """Приращение директорного угла в (-pi/2, pi/2]."""
    return (d + math.pi / 2) % math.pi - math.pi / 2


def loop_on_grid(th, ia, ib, w):
    """Индекс петли — границы квадрата со стороной w клеток, угол (ia,ib).

    Возвращает (индекс, максимальная |неоднозначность| по рёбрам) или
    (None, None), если на петле есть пропуск.
    """
    pts = []
    for s in range(w):
        pts.append((ia + s, ib))
    for s in range(w):
        pts.append((ia + w, ib + s))
    for s in range(w):
        pts.append((ia + w - s, ib + w))
    for s in range(w):
        pts.append((ia, ib + w - s))
    vals = []
    for p, q in pts:
        if not (0 <= p < th.shape[0] and 0 <= q < th.shape[1]):
            return None, None
        if not np.isfinite(th[p, q]):
            return None, None
        vals.append(th[p, q])
    tot, amb = 0.0, 0.0
    for t in range(len(vals)):
        d = unwrap_pi(vals[(t + 1) % len(vals)] - vals[t])
        tot += d
        amb = max(amb, abs(d) / (math.pi / 2))
    return tot / (2 * math.pi), amb


def census(th, g):
    """Перепись: клетки, аддитивность на блоках 2x2, устойчивость."""
    n = 2 * g + 1
    cell = np.full((n - 1, n - 1), np.nan)
    ambs = []
    for ia in range(n - 1):
        for ib in range(n - 1):
            v, am = loop_on_grid(th, ia, ib, 1)
            if v is not None:
                cell[ia, ib] = v
                ambs.append(am)
    # аддитивность 2x2
    ok, bad = 0, 0
    # устойчивость: индекс блока 2x2 против суммы его клеток
    rob_hit, rob_tot = 0, 0
    for ia in range(0, n - 2, 2):
        for ib in range(0, n - 2, 2):
            big, _ = loop_on_grid(th, ia, ib, 2)
            sub = cell[ia:ia + 2, ib:ib + 2]
            if big is None or not np.isfinite(sub).all():
                continue
            s = float(sub.sum())
            if abs(big - s) < 1e-6:
                ok += 1
            else:
                bad += 1
            rob_tot += 1
            if abs(round(2 * s) / 2) > 0.25 and abs(big - s) < 1e-6:
                rob_hit += 1
    return cell, np.array(ambs), ok, bad, rob_hit, rob_tot


def main():
    ap = argparse.ArgumentParser(description='Сохранение суммарного индекса')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--grid', type=int, default=7)
    ap.add_argument('--step', type=float, default=0.07)
    ap.add_argument('--minsize', type=int, default=25)
    ap.add_argument('--projmin', type=float, default=0.25)
    ap.add_argument('--patches', type=int, default=14)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    print(f'конусный порядок, ромб T={a.t}, N={a.n}, интервал >= {a.minsize}')
    print(f'сетка {2*a.grid+1}x{2*a.grid+1}, шаг {a.step}, '
          f'заплат на засев {a.patches}')
    print('индекс директора кратен 1/2; знак из плоскости обхода')
    print()

    for mode in ('поле', 'случайное'):
        cells, ambs = [], []
        OK = BAD = RH = RT = 0
        totals = []
        for t in range(a.seeds):
            rs = np.random.default_rng(2801000 + 977 * t)
            X = diamond(a.n, a.t, rs)
            R = cone_rel(X)
            fut = R.sum(0)
            i0 = int(np.argmax(fut * (X[:, 0] < 0.25 * a.t)))
            cand = np.flatnonzero(R[:, i0])
            if len(cand) < 300:
                del R, X
                continue
            rs2 = np.random.default_rng(900 + t)
            cent = rs2.choice(cand, min(a.patches, len(cand)), replace=False)
            for j0 in cent:
                j0 = int(j0)
                try:
                    u, v = plane_of(X, i0, j0)
                except Exception:
                    continue
                if mode == 'поле':
                    th = field_grid(X, R, i0, j0, u, v, a.grid, a.step,
                                    a.minsize, cand, a.projmin)
                else:
                    th = rs2.uniform(-math.pi, math.pi,
                                     (2 * a.grid + 1, 2 * a.grid + 1))
                c, am, ok, bad, rh, rt = census(th, a.grid)
                cells.append(c[np.isfinite(c)])
                ambs.append(am)
                OK += ok; BAD += bad; RH += rh; RT += rt
                # суммарный индекс заплаты против внешней петли
                big, _ = loop_on_grid(th, 0, 0, 2 * a.grid)
                s = float(np.nansum(c))
                if big is not None:
                    totals.append((big, s))
            del R, X
        if not cells:
            print(f'{mode}: поле не построилось')
            continue
        A = np.concatenate(cells)
        AM = np.concatenate(ambs)
        q = np.round(A * 2) / 2
        pos = int((q >= 0.5).sum()); neg = int((q <= -0.5).sum())
        nz = pos + neg
        big1 = int((np.abs(q) >= 1.0).sum())
        print(f'--- {mode}:  клеток {len(A)}')
        print(f'  1. разреженность: ненулевых {nz/len(A):.4f} '
              f'({nz} из {len(A)})')
        print(f'  2. квантованность: |индекс| >= 1 у {big1/len(A):.4f}; '
              f'с.к.о. индекса {A.std():.3f}')
        if nz:
            p = pos / nz
            e = math.sqrt(max(p * (1 - p), 1e-9) / nz)
            print(f'  3. баланс знаков: {pos} плюс / {neg} минус, '
                  f'доля {p:.3f} ± {e:.3f}, '
                  f'{abs(p-0.5)/max(e,1e-9):.1f}σ от симметрии')
        else:
            print('  3. баланс знаков: ненулевых нет')
        print(f'  4. устойчивость: блок 2x2 воспроизводит ненулевую сумму '
              f'в {RH}/{RT}')
        print(f'  5. однозначность: доля рёбер с |dtheta| > 0.9*pi/2 = '
              f'{(AM > 0.9).mean():.4f}, среднее {AM.mean():.3f}')
        print(f'  аддитивность 2x2: точно {OK}, нарушено {BAD}')
        if totals:
            T2 = np.array(totals)
            print(f'  внешняя петля против суммы клеток: совпало '
                  f'{int((np.abs(T2[:,0]-T2[:,1])<1e-6).sum())}/{len(T2)};'
                  f'  суммарный индекс заплаты '
                  f'{T2[:,1].mean():+.3f} ± '
                  f'{T2[:,1].std()/math.sqrt(len(T2)):.3f}')
        print()

    print('  заряд: разреженность мала, значения ±1/2, знаки сбалансированы,')
    print('  блок воспроизводит сумму, неоднозначных рёбер мало.')
    print('  случайное поле обязано провалить всё, кроме аддитивности —')
    print('  это и показывает, что аддитивность тестом не является.')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
