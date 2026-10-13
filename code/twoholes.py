#!/usr/bin/env python3
"""ПРИТЯГИВАЮТСЯ ЛИ ДВЕ МАССЫ: действие против расстояния между дырками.

ПОЧЕМУ ЭТО ВОЗМОЖНО ТОЛЬКО ТЕПЕРЬ. delay-doc.md: масса есть РАЗРЕЖЕНИЕ,
а не сгусток. Разрежение замедляет часы (4.5 сигмы) и выгибает цепи
наружу (5.0 сигмы) — обе половины тяготения с верными знаками, и плавно
по глубине, то есть поле, а не запрет пройти. Сгусток даёт обе
половины наоборот, потому что он лестница (внутренняя высота 57..105
против фоновых 11.8).

Все прежние попытки получить притяжение ставили массой ТРУБКУ
ДОБАВЛЕННЫХ событий. Если отождествление перевёрнуто, то перевёрнут и
знак всех тех измерений, и опыт надо ставить заново — на дырках.

ЧТО МЕРИМ. Две трубки-разрежения на расстоянии sep друг от друга,
действие мира против sep.

    S падает при сближении  =>  ПРИТЯЖЕНИЕ МАСС
    S растёт при сближении  =>  отталкивание
    S не меняется           =>  взаимодействия нет

ЧТО СДЕЛАНО, ЧТОБЫ ЭТО НЕ БЫЛО АРТЕФАКТОМ.

  1. Число событий ТОЧНО совпадает при всех sep: удаляется ровно K
     событий, выбранных случайно из попавших в объединение трубок.
     Иначе S менялось бы просто от N.
  2. Сравнение ПАРНОЕ: один и тот же засев при всех sep, разности
     берутся внутри засева. Это убирает почти всю дисперсию.
  3. Контроль: те же K событий удалены случайно по всему ромбу —
     даёт уровень S при том же N без всякой формы.
  4. Трубки не пересекаются: sep > 2*rad при всех значениях.
  5. Засевов 16: на шести проект дважды получал ложные «13 сигм».

  python3 twoholes.py
  python3 twoholes.py --seps 0.4 0.6 0.8 1.0 --seeds 16
"""
import argparse
import math
import time

import numpy as np

from weight import diamond
from mc import World


def two_tubes(X, rad, sep, axis=1):
    """Маска событий, попавших в объединение двух трубок вдоль времени."""
    c = sep / 2.0
    d1 = X[:, 1:].copy(); d1[:, axis - 1] -= c
    d2 = X[:, 1:].copy(); d2[:, axis - 1] += c
    r1 = np.linalg.norm(d1, axis=1)
    r2 = np.linalg.norm(d2, axis=1)
    return (r1 < rad) | (r2 < rad)


def main():
    ap = argparse.ArgumentParser(description='Притяжение двух дырок')
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.16)
    ap.add_argument('--seps', type=float, nargs='+',
                    default=[0.40, 0.55, 0.70, 0.85, 1.00])
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--seeds', type=int, default=16)
    a = ap.parse_args()

    t0 = time.time()
    print(f'ромб T={a.t}; две трубки-разрежения радиуса {a.rad}, '
          f'eps={a.eps}')
    print('число удалённых событий ТОЧНО одинаково при всех расстояниях')
    print('сравнение ПАРНОЕ: один засев при всех расстояниях')
    print('S падает при сближении => ПРИТЯЖЕНИЕ МАСС')
    print(f'засевов {a.seeds}')

    # сколько событий удалять: минимум по всем sep и засевам, с запасом
    probe = []
    for t in range(4):
        rs = np.random.default_rng(2701000 + 977 * t)
        X = diamond(a.t, rs)
        for sep in a.seps:
            probe.append(int(two_tubes(X, a.rad, sep).sum()))
    K = int(min(probe) * 0.92)
    print(f'удаляется K = {K} событий (минимум по пробе {min(probe)})')
    print()

    rows = {sep: [] for sep in a.seps}
    ctrl = []
    for t in range(a.seeds):
        rs = np.random.default_rng(2701000 + 977 * t)
        X0 = diamond(a.t, rs)
        N0 = len(X0)
        for sep in a.seps:
            cand = np.flatnonzero(two_tubes(X0, a.rad, sep))
            if len(cand) < K:
                rows[sep].append(np.nan)
                continue
            drop = rs.choice(cand, K, replace=False)
            keep = np.ones(N0, bool); keep[drop] = False
            w = World(X0[keep], a.eps)
            rows[sep].append(w.S / len(w.X))
            del w
        drop = rs.choice(N0, K, replace=False)
        keep = np.ones(N0, bool); keep[drop] = False
        w = World(X0[keep], a.eps)
        ctrl.append(w.S / len(w.X))
        del w
        if (t + 1) % 4 == 0:
            print(f'   засевов готово {t+1}/{a.seeds}', flush=True)

    print()
    print(f"{'расстояние':>12}{'S/N':>14}{'± ошибка':>12}"
          f"{'минус дальнее':>16}{'сигм (парно)':>14}")
    far = np.array(rows[a.seps[-1]], float)
    for sep in a.seps:
        v = np.array(rows[sep], float)
        ok = np.isfinite(v) & np.isfinite(far)
        d = v[ok] - far[ok]                     # парная разность
        se = d.std() / math.sqrt(max(len(d), 1)) if len(d) > 1 else np.nan
        print(f'{sep:>12.2f}{np.nanmean(v):>14.5f}'
              f'{np.nanstd(v)/math.sqrt(ok.sum()):>12.5f}'
              f'{d.mean():>+16.5f}'
              f'{(abs(d.mean())/se if se > 1e-12 else float("nan")):>14.1f}')
    c = np.array(ctrl, float)
    print(f"{'контроль':>12}{c.mean():>14.5f}"
          f"{c.std()/math.sqrt(len(c)):>12.5f}")

    # монотонность и наклон
    xs = np.array(a.seps)
    ys = np.array([np.nanmean(rows[s]) for s in a.seps])
    dif = np.diff(ys)
    print(f'\n  наклон S/N по расстоянию: '
          f'{np.polyfit(xs, ys, 1)[0]:+.5f} на единицу')
    print(f'  монотонность: {int((dif > 0).sum())} шагов вверх из '
          f'{len(dif)}')
    print('  всё вверх => S растёт с расстоянием => сближение выгодно '
          '=> ПРИТЯЖЕНИЕ')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
