#!/usr/bin/env python3
"""КЛЮЧЕВОЙ КАМЕНЬ: поперечник не есть порядковая величина.

ЗАЧЕМ. Сегодняшняя цепочка сошлась к одному вопросу. Калибровочная
теория на решётке берёт кривизну как холономию МАЛОЙ петли: кривизна на
площадь, площадь -> 0. Я весь день мерил холономию по четвёрным циклам
графа связей и получал, что перемешанный контроль ей равен. Причина
нашлась структурно:

  `locality.py` связи светоподобны: собственное время ~ N^-0.2095 при
                ell ~ N^-0.25, а |Δx| ~ N^-0.0160 стоит на месте;
                поперечник цикла 0.62 в ромбе радиуса 1.0 при любом N;
  `interval.py` ограничение интервалом помогает лишь частично:
                поперечник N^-0.0758, площадь N^-0.2413;
  `fatint.py`   отбор по длине цепи не помогает вовсе (−0.1392 против
                −0.1350), зато показывает причину: при фиксированном
                числе событий интервала пары СПОЛЗАЮТ к конусу,
                Δt/|Δx| = 2.62 -> 1.75 при N 400 -> 3200.

ПОДОЗРЕНИЕ. Дело не в неудачном признаке, а в том, что отбирать нечем.
Россыпь лоренц-инвариантна. Буст растягивает интервал в длинную тонкую
щепку и при этом сохраняет ПОРЯДОК точно — он же причинное отношение не
меняет. Тогда пространственный поперечник не является функцией порядка, и
НИКАКОЙ порядковый отбор не может выделить компактные интервалы.

ЧТО МЕРИМ. Берём россыпь, бустим с разной быстротой, и сравниваем:
    порядок     — число отношений, связей, высота, доля прямых, доля
                  сравнимых пар (размерность Мирхейма–Мейера);
    координаты  — пространственный поперечник интервалов и циклов.
Буст применяется к ТЕМ ЖЕ событиям, поэтому порядок обязан совпасть
ТОЖДЕСТВЕННО (ноль расхождений), а поперечник — разъехаться.

ЕСЛИ ТАК, то вывод теоремного уровня: холономия по малой петле в
причинном множестве невозможна В ПРИНЦИПЕ, а не трудна. И тогда понятно,
почему теория причинных множеств идёт другим путём: нелокальный
д'Аламбертиан с промежуточным масштабом и действие Бенинказы–Доукера,
где знакопеременная сумма по числам малых интервалов (1, −9, +16, −8)
делает ту работу, которую на решётке делают малые петли.

  python3 boost.py
  python3 boost.py --rap 0 0.5 1.0 1.5 2.0 --seeds 6
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of
from response import warped_diamond


def boosted(X, rap):
    """Буст по оси 1 с быстротой rap. Причинность сохраняется точно."""
    ch, sh = math.cosh(rap), math.sinh(rap)
    Y = X.copy()
    Y[:, 0] = ch * X[:, 0] + sh * X[:, 1]
    Y[:, 1] = sh * X[:, 0] + ch * X[:, 1]
    return Y


def order_facts(R):
    """Порядковые величины: отношения, связи, высота, доля прямых."""
    L = links_of(R)
    n = len(R)
    h = np.ones(n, np.int64)
    for i in np.argsort(R.sum(1), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    nr, nl = int(R.sum()), int(L.sum())
    frac = nr / (n * (n - 1) / 2)
    return nr, nl, int(h.max()), nl / max(nr, 1), frac


def spatial(X, R, clo, chi, npair, rs):
    """Пространственный поперечник интервалов — величина НЕ порядковая."""
    A = R.astype(np.float32)
    INT = A @ A
    rel = np.argwhere(R)
    card = INT[rel[:, 0], rel[:, 1]]
    ok = np.flatnonzero((card >= clo) & (card <= chi))
    if len(ok) < 8:
        return float('nan'), float('nan')
    ok = rs.choice(ok, min(npair, len(ok)), replace=False)
    dia, ratio = [], []
    for p in ok:
        i, j = int(rel[p, 0]), int(rel[p, 1])
        mem = np.flatnonzero(R[:, j] & R[i, :])
        sub = np.concatenate([[j], mem, [i]])
        P = X[sub]
        d = 0.0
        for q in range(len(P)):
            dd = np.linalg.norm(P[q + 1:, 1:] - P[q, 1:], axis=1)
            if len(dd):
                d = max(d, float(dd.max()))
        dia.append(d)
        dt = abs(X[i, 0] - X[j, 0])
        dx = float(np.linalg.norm(X[i, 1:] - X[j, 1:]))
        ratio.append(dt / max(dx, 1e-9))
    return float(np.mean(dia)), float(np.mean(ratio))


def main():
    ap = argparse.ArgumentParser(description='Поперечник против порядка')
    ap.add_argument('--n', type=int, default=1800)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rap', type=float, nargs='+',
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument('--card', type=int, nargs=2, default=[12, 20])
    ap.add_argument('--npair', type=int, default=400)
    ap.add_argument('--seeds', type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    clo, chi = a.card
    print(f'N={a.n}, ромб T={a.t}, интервалы {clo}..{chi} событий, '
          f'засевов {a.seeds}')
    print('буст применяется к ТЕМ ЖЕ событиям: порядок обязан совпасть')
    print('тождественно, пространственный поперечник — разъехаться')
    print()
    print(f"{'быстрота':>9}{'отношений':>11}{'связей':>9}{'высота':>8}"
          f"{'доля прям.':>12}{'доля сравн.':>13}"
          f"{'ПОПЕРЕЧНИК':>13}{'Δt/|Δx|':>10}")

    base = None
    rows = []
    for rp in a.rap:
        of, di, ra = [], [], []
        for t in range(a.seeds):
            rs = np.random.default_rng(8001000 + 977 * t)
            X = warped_diamond(a.n, a.t, 0.0, 0.45, rs)
            Y = boosted(X, rp)
            o = np.argsort(Y[:, 0])
            Y = Y[o]
            R = cone_rel(Y)
            of.append(order_facts(R))
            d, r = spatial(Y, R, clo, chi, a.npair,
                           np.random.default_rng(23 + t))
            di.append(d); ra.append(r)
            del R, X, Y
        O = np.array(of, float)
        m = O.mean(0)
        if base is None:
            base = m.copy()
        print(f'{rp:>9.2f}{m[0]:>11.0f}{m[1]:>9.0f}{m[2]:>8.1f}'
              f'{m[3]:>12.5f}{m[4]:>13.5f}'
              f'{np.nanmean(di):>13.5f}{np.nanmean(ra):>10.2f}', flush=True)
        rows.append((rp, m, float(np.nanmean(di))))

    print()
    print('  расхождение порядковых величин с быстротой 0:')
    nms = ['отношений', 'связей', 'высота', 'доля прямых', 'доля сравнимых']
    for k, nm in enumerate(nms):
        dev = max(abs(r[1][k] - base[k]) /
                  max(abs(base[k]), 1e-12) for r in rows)
        print(f'    {nm:<16} не более {100*dev:.3f} %')
    d0 = rows[0][2]
    dmax = max(r[2] for r in rows)
    print(f'  поперечник: {d0:.5f} при быстроте 0, до {dmax:.5f} — '
          f'в {dmax/d0:.2f} раза')
    print()
    print('  порядок не шевельнулся, поперечник вырос =>')
    print('  ПОПЕРЕЧНИК НЕ ЕСТЬ ФУНКЦИЯ ПОРЯДКА, и порядковым отбором')
    print('  компактные интервалы выделить НЕЛЬЗЯ. Малой петли нет.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
