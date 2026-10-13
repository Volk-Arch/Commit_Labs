#!/usr/bin/env python3
"""ПРОТЕЧКА: зависит ли вклад наблюдателя в действие от его скорости.

ВОПРОС, И ПОЧЕМУ ИМЕННО ТАКОЙ. `radar-doc.md` установил трёхслойную
картину: порядок даёт время (часы 4–6 %) и не даёт пространства (буст:
поперечник в 4.88 раза при расхождении порядковых величин 0.000 %);
порядок плюс ОДНА цепь даёт полную систему координат (0.9999 по времени,
0.9997 по пространству); порядок плюс ДВЕ цепи даёт группу Лоренца
(`A = γ²` с промахом 0.07…0.49 %).

Его вторая мысль: без пространства — квантовая структура, с ним — наш
мир. Проверять это в лоб нельзя: «безопорная величина не зависит от
опоры» есть тавтология того же рода, что убила вчерашнюю холономию.

Содержательный вопрос другой. Цепь-наблюдатель — не метка на бумаге, а
ФИЗИЧЕСКИЙ объект внутри множества (плотная мировая линия). Амплитуда
квантовой меры Соркина есть `exp(iS)`, а действие обязано быть
ЛОРЕНЦЕВЫМ СКАЛЯРОМ. Значит если вставить ОДИН И ТОТ ЖЕ объект — та же
мощность, то же собственное время — но в разном состоянии движения, его
вклад в действие обязан быть ОДИНАКОВ.

    ΔS не зависит от быстроты -> слои чисты, наблюдатель даёт координаты
                                 и не трогает амплитуду
    ΔS зависит от быстроты    -> ПРОТЕЧКА: система отсчёта вошла в то, из
                                 чего делается амплитуда

УСТРОЙСТВО. Наблюдатель строится по СОБСТВЕННОМУ времени: от середины
ромба вдоль направления `(cosh ρ, sinh ρ)` на проперное время `tau`, с
`dens` событиями через равные промежутки собственного времени. Тогда при
любой быстроте это один и тот же объект — та же мощность, та же
собственная длительность, — только движущийся.

КОНТРОЛИ:
  быстрота 0 рядом — точка отсчёта;
  СДВИГ при той же быстроте: `ΔS` обязано совпасть (однородность);
  число добавленных отношений печатается рядом: оно тоже обязано не
    зависеть от быстроты, и это более грубый, но независимый признак;
  проверка попадания: наблюдатель обязан целиком лежать внутри ромба,
    иначе сравниваются разные объекты — печатается отдельно.

  python3 leak.py
  python3 leak.py --n 900 --raps 0 0.3 0.6 0.9 --seeds 30
"""
import argparse
import math
import time

import numpy as np

from ens2 import action_and_pairs
from particles import diamond


def obs_proper(dens, tau, rap, t0, x0):
    """Наблюдатель по СОБСТВЕННОМУ времени: один объект при любой скорости."""
    ch, sh = math.cosh(rap), math.sinh(rap)
    s = np.linspace(-tau / 2.0, tau / 2.0, dens)
    Z = np.zeros((dens, 4))
    Z[:, 0] = t0 + ch * s
    Z[:, 1] = x0 + sh * s
    return Z


def fits(Z, T=1.0):
    r = np.linalg.norm(Z[:, 1:], axis=1)
    return bool(np.all((Z[:, 0] >= r) & (T - Z[:, 0] >= r)))


def main():
    ap = argparse.ArgumentParser(description='Протечка отсчёта в действие')
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--dens', type=int, default=24)
    ap.add_argument('--tau', type=float, default=0.30)
    ap.add_argument('--raps', type=float, nargs='+',
                    default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument('--seeds', type=int, default=26)
    ap.add_argument('--off', type=float, default=0.07)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    print(f'N фона = {a.n}, ell = {ell:.4f}')
    print(f'наблюдатель: {a.dens} событий, собственное время {a.tau} = '
          f'{a.tau/ell:.1f} ell — ОДИН объект при любой быстроте')
    print(f'засевов {a.seeds}')
    print()
    print('ΔS = S(с наблюдателем) − S(без). Действие обязано быть')
    print('лоренцевым скаляром, значит ΔS НЕ ДОЛЖНО зависеть от быстроты.')
    print()

    # проверка попадания до всякого замера
    print('попадание наблюдателя в ромб:')
    for rap in a.raps:
        Z = obs_proper(a.dens, a.tau, rap, 0.5, 0.0)
        Zo = obs_proper(a.dens, a.tau, rap, 0.5, a.off)
        print(f'  быстрота {rap:<4.1f} центр: {"да" if fits(Z) else "НЕТ"}'
              f'   сдвиг {a.off}: {"да" if fits(Zo) else "НЕТ"}')
    if not all(fits(obs_proper(a.dens, a.tau, r, 0.5, 0.0))
               for r in a.raps):
        print('  наблюдатель вылезает — объекты разные, замер негоден')
        return
    print()

    S0 = []
    worlds = []
    for s in range(a.seeds):
        X = diamond(a.n, np.random.default_rng(25500 + 107 * s))
        worlds.append(X)
        S0.append(action_and_pairs(X)[0])
    S0 = np.array(S0)
    print(f'действие пустого мира: {S0.mean():.1f} ± '
          f'{S0.std(ddof=1)/math.sqrt(len(S0)):.1f}  '
          f'(разброс {S0.std(ddof=1):.1f})')
    print()
    print(f"{'быстрота':>9}{'ΔS':>11}{'ст.ошибка':>11}{'сигм от 0':>11}"
          f"{'Δотношений':>13}{'ст.ош':>8}{'промах ΔS к ρ=0':>18}")

    ref = None
    for rap in a.raps:
        d, dp = [], []
        for s in range(a.seeds):
            Z = obs_proper(a.dens, a.tau, rap, 0.5, 0.0)
            Sw, Pw = action_and_pairs(np.vstack([worlds[s], Z]))
            Sb, Pb = action_and_pairs(worlds[s])
            d.append(Sw - Sb)
            dp.append(Pw - Pb)
        d = np.array(d); dp = np.array(dp)
        se = d.std(ddof=1) / math.sqrt(len(d))
        sp = dp.std(ddof=1) / math.sqrt(len(dp))
        if ref is None:
            ref = (d.mean(), se)
            pm = '— точка отсчёта'
        else:
            dd = d.mean() - ref[0]
            sd = math.hypot(se, ref[1])
            pm = (f'{dd:+.1f} ({abs(dd)/max(sd,1e-9):.1f}σ)')
        print(f'{rap:>9.1f}{d.mean():>11.1f}{se:>11.1f}'
              f'{abs(d.mean())/max(se,1e-9):>11.1f}'
              f'{dp.mean():>13.1f}{sp:>8.1f}{pm:>18}', flush=True)

    print()
    print(f'КОНТРОЛЬ ОДНОРОДНОСТИ: та же быстрота, сдвиг на {a.off}')
    print(f"{'быстрота':>9}{'ΔS центр':>11}{'ΔS сдвиг':>11}"
          f"{'разница':>10}{'сигм':>7}")
    for rap in a.raps:
        dc, do = [], []
        for s in range(a.seeds):
            Sb = action_and_pairs(worlds[s])[0]
            Zc = obs_proper(a.dens, a.tau, rap, 0.5, 0.0)
            Zo = obs_proper(a.dens, a.tau, rap, 0.5, a.off)
            dc.append(action_and_pairs(np.vstack([worlds[s], Zc]))[0] - Sb)
            do.append(action_and_pairs(np.vstack([worlds[s], Zo]))[0] - Sb)
        dc = np.array(dc); do = np.array(do)
        dd = do.mean() - dc.mean()
        sd = math.hypot(dc.std(ddof=1) / math.sqrt(len(dc)),
                        do.std(ddof=1) / math.sqrt(len(do)))
        print(f'{rap:>9.1f}{dc.mean():>11.1f}{do.mean():>11.1f}'
              f'{dd:>+10.1f}{abs(dd)/max(sd,1e-9):>7.1f}', flush=True)

    print()
    print('  ΔS одинаково при всех быстротах => слои чисты;')
    print('  зависит => ПРОТЕЧКА отсчёта в амплитуду')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
