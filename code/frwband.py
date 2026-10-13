#!/usr/bin/env python3
"""ПАРНОЕ СРАВНЕНИЕ ПО ПОЛОСАМ: предсказание БЕЗ свободных чисел.

ЗАЧЕМ ОТДЕЛЬНО ОТ `frwfit.py`. Сводная подгонка описывает помеху
гладкой четырёхпараметрической формой, и это оправдано, когда помеха
гладкая. Но плоское плечо показывает в нижней полосе выброс −0.0071,
который гладкая форма не ловит, и он протекает в амплитуду: сводное
`R_uu` вышло на 26 % ниже истины, тогда как верхние полосы, где сигнал
втрое выше помехи, ложатся точно.

Поэтому здесь помеха не моделируется вовсе, а ВЫЧИТАЕТСЯ парно: те же
полосы по `v²`, те же размеры, разность «излучение минус плоское».
Остаётся сравнение с предсказанием

    X_2(полоса) = v_2 * sqrt(24 C_1/pi) * (3+v²)/(eta0⁴(1-v²))

где СВОБОДНЫХ ЧИСЕЛ НЕТ ВОВСЕ: `v_2 = -0.0201` и `v_3 = -0.0452` взяты
из другого прогона (`invert.py`), `eta0` задано построением, остальное —
измеренные `C_1` и `v²`. Поэтому хи² считается на ПОЛНОМ числе полос, без
вычета степеней свободы.

Внутриполосное усреднение плана берётся по самим интервалам, а не по
средним `C_1` и `v²`: модель нелинейна по обоим, и подстановка средних
дала бы смещение.

Ошибки — складным ножом по засевам, парно: из каждой выборки
выбрасывается один и тот же номер засева.

  python3 frwband.py
"""
import math
import pickle

import numpy as np

from frw import SCRATCH, KFAC, V2CAL, V3CAL, IC, IX2, IX3, IV


def model(B, eta0, vcal):
    """Предсказание X для каждого интервала, затем среднее по полосе."""
    Ruu = (3.0 + B[:, IV]) / (eta0 ** 4 * (1.0 - B[:, IV]))
    return float(np.mean(vcal * KFAC * np.sqrt(B[:, IC]) * Ruu))


def jack(vals):
    n = len(vals)
    m = np.mean(vals)
    return m, math.sqrt((n - 1) / n * ((vals - m) ** 2).sum())


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/frw.pkl', 'rb') as fh:
        res, cfg, VB = pickle.load(fh)
    fk = [k for k in res if k[0] == 'flat'][0]
    eps = sorted([k for k in res if k[0] != 'flat'], key=lambda k: k[1])
    Af, sdf = res[fk]
    seeds = np.unique(sdf)

    for col, cname, vcal in ((IX2, 'X_2', V2CAL), (IX3, 'X_3', V3CAL)):
        print('=' * 78)
        print(f'{cname},  v = {vcal:+.4f} из другого прогона')
        for key in eps:
            A, sd = res[key]
            eta0 = key[1]
            t_tru = eta0 ** 2 / 2.0
            print(f'  --- излучение eta0 = {eta0}, '
                  f't = {t_tru:.1f} планк. времён ---')
            print(f"{'v²':>8}{'C_1':>7}{'излуч.':>10}{'плоское':>10}"
                  f"{'разность':>11}{'ст.ош':>9}{'предск.':>10}"
                  f"{'сигм':>7}")
            chi = 0.0
            nb = 0
            Rs = []
            for i in range(len(VB) - 1):
                ma = (A[:, IV] >= VB[i]) & (A[:, IV] < VB[i + 1])
                mf = (Af[:, IV] >= VB[i]) & (Af[:, IV] < VB[i + 1])
                if ma.sum() < 50 or mf.sum() < 50:
                    continue
                # складной нож парно по номеру засева
                jk = []
                for s in seeds:
                    a_ = A[ma & (sd != s)]
                    f_ = Af[mf & (sdf != s)]
                    jk.append(a_[:, col].mean() - f_[:, col].mean())
                jk = np.array(jk)
                d = A[ma][:, col].mean() - Af[mf][:, col].mean()
                _, e = jack(jk)
                pred = model(A[ma], eta0, vcal)
                z = (d - pred) / max(e, 1e-30)
                chi += z * z
                nb += 1
                # восстановленное R_uu(сопутствующий) = 3/eta0⁴ по полосе:
                # d = vcal * g / eta0⁴, где g — форма без 1/eta0⁴
                g = float(np.mean(KFAC * np.sqrt(A[ma][:, IC])
                                  * (3.0 + A[ma][:, IV])
                                  / (1.0 - A[ma][:, IV])))
                Rs.append((3.0 * d / (vcal * g), 3.0 * e / abs(vcal * g)))
                print(f'{A[ma][:,IV].mean():>8.3f}{A[ma][:,IC].mean():>7.0f}'
                      f'{A[ma][:,col].mean():>+10.4f}'
                      f'{Af[mf][:,col].mean():>+10.4f}{d:>+11.4f}{e:>9.4f}'
                      f'{pred:>+10.4f}{z:>+7.1f}')
            print(f'    хи² = {chi:.2f} на {nb} полос, свободных чисел НЕТ'
                  f'   (хи²/полоса = {chi/max(nb,1):.2f})')
            # взвешенное восстановление R_uu по всем полосам
            Rs = np.array(Rs)
            w = 1.0 / Rs[:, 1] ** 2
            Rm = float((w * Rs[:, 0]).sum() / w.sum())
            Re = float(1.0 / math.sqrt(w.sum()))
            tru = 3.0 / eta0 ** 4
            print(f'    R_uu l² (сопутств.) = {Rm:.3e} +- {Re:.3e}'
                  f'   истина {tru:.3e}'
                  f'   {abs(Rm-tru)/max(Re,1e-30):.1f} сигм')
            if Rm > 0:
                age = math.sqrt(3.0 / (4.0 * Rm))
                age_e = age * 0.5 * Re / Rm
                print(f'    ВОЗРАСТ ВСЕЛЕННОЙ = {age:.1f} +- {age_e:.1f}'
                      f' планк. времён   истина {t_tru:.1f}'
                      f'   {abs(age-t_tru)/max(age_e,1e-30):.1f} сигм')
                print(f'    плотность rho = {Rm/(8*math.pi):.3e} планк.'
                      f'   истина {3/(32*math.pi*t_tru**2):.3e}')
            print()


if __name__ == '__main__':
    main()
