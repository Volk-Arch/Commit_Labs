#!/usr/bin/env python3
"""ИЗМЕРИТЕЛЬ `w_dot`: согласие подгонок при разных `tau`.

`w` измерено по настоящим данным (`wreal.py`): lin даёт
+0.3498 +- 0.0160 против +1/3 (1.0 сигмы), root +0.9696 +- 0.1404 против
+1 (0.2 сигмы), плоское отвергнуто (chi²/степень = 3.64).

Отсюда новый инструмент. При ПОСТОЯННОМ `w` подгонки по интервалам
разного размера обязаны согласиться — потому что отклик считается точно
при любом `tau`, и никакого разложения, которое ломалось бы с ростом
`tau`, здесь нет. Значит РАСХОЖДЕНИЕ подгонок при разных `tau` есть
прямая мера того, что `w` меняется, то есть `w_dot`.

Идея подсказана отзывом другой ИИ, но в её форме («минимум отклика
ползёт с `tau`, значит `dw_min/dtau` несёт информацию о `w_dot`») она
неверна: минимум ползёт из-за нелинейности отклика, а семейство имеет
постоянное `w` по построению. Верная форма — обратная: несогласованность
подгонок, а не положение минимума.

ЗДЕСЬ инструмент проверяется на плече с ПОСТОЯННЫМ `w` (lin): ячейки
делятся по `C_1` на малые и большие, `w` подгоняется в каждой половине
отдельно, и разность обязана выйти НУЛЁМ. Это одновременно контроль
подгонки и калибровка чувствительности к `w_dot`.

  python3 wdot.py
"""
import math

import numpy as np

from pencil import KFAC, IX2, IX3, IC
from refit import load_all
from wreal import (ETA0, PGRID, cells, ref_grid, fit_p, w_of)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(161803399)
    arms = load_all()
    eta_s = ETA0 - 1.0 / 0.035
    A, sd, par = arms['lin']
    cl = cells(A, sd)
    tau = np.array([math.sqrt(KFAC * math.sqrt(c['c1'])) for c in cl])
    print(f'плечо lin, ячеек {len(cl)}, tau от {tau.min():.2f} до '
          f'{tau.max():.2f}')
    print('опора считается один раз на все ячейки...', flush=True)
    R, E = ref_grid(cl, eta_s, 900000, rs)
    med = np.median(tau)
    lo = tau <= med
    print()
    print(f'раздел по tau = {med:.2f}: малых ячеек {lo.sum()}, '
          f'больших {(~lo).sum()}')
    print()
    print(f"{'половина':>12}{'<tau>':>8}{'p':>18}{'w':>20}{'chi²/степ':>12}")
    out = []
    us = np.unique(sd)
    for nm, m in (('малые tau', lo), ('большие tau', ~lo)):
        idx = np.flatnonzero(m)
        obs = np.array([cl[i]['obs'] for i in idx])
        err = np.array([cl[i]['err'] for i in idx])
        Rm, Em = R[:, idx], E[:, idx]
        pp, chi, _ = fit_p(obs, err, Rm, Em)
        jk = []
        for u in us:
            o2 = [[A[cl[i]['msk'] & (sd != u), IX2].mean(),
                   A[cl[i]['msk'] & (sd != u), IX3].mean()] for i in idx]
            jk.append(fit_p(np.array(o2), err, Rm, Em)[0])
        jk = np.array(jk); k = len(us)
        se = math.sqrt((k - 1) / k * ((jk - jk.mean()) ** 2).sum())
        ew = abs(w_of(pp + se) - w_of(pp - se)) / 2
        dof = obs.size - 1
        out.append((w_of(pp), ew, jk))
        print(f'{nm:>12}{tau[m].mean():>8.2f}'
              f'{f"{pp:.4f}+-{se:.4f}":>18}'
              f'{f"{w_of(pp):+.4f}+-{ew:.4f}":>20}'
              f'{f"{chi:.1f}/{dof}":>12}', flush=True)
    print()
    print('=' * 72)
    (w1, e1, j1), (w2, e2, j2) = out
    # разность на ОДНИХ мирах -> складной нож по разности, не по сумме
    dj = np.array([w_of(a) - w_of(b) for a, b in zip(j1, j2)])
    k = len(dj)
    sed = math.sqrt((k - 1) / k * ((dj - dj.mean()) ** 2).sum())
    d = w1 - w2
    print(f'разность w(малые) - w(большие) = {d:+.4f} +- {sed:.4f}'
          f'   ({abs(d)/max(sed,1e-9):.1f} сигм от нуля)')
    print(f'  (складной нож взят по РАЗНОСТИ на одних мирах — половины '
          f'коррелированы,\n   и складывать их ошибки в квадратуре было бы '
          f'неверно)')
    print()
    print('КОНТРОЛЬ: у плеча с ПОСТОЯННЫМ `w` разность обязана быть нулём.')
    print('->', 'ЧИСТО, инструмент калиброван' if abs(d) < 2.5 * sed
          else 'РАСХОЖДЕНИЕ: либо подгонка смещена, либо отклик неполон')
    print()
    print(f'''ЧУВСТВИТЕЛЬНОСТЬ. Ошибка разности {sed:.4f} и есть предел, с
которым этот набор различает постоянное `w` от меняющегося. Интервалы
здесь охватывают tau от {tau.min():.1f} до {tau.max():.1f} при
eta_0 - eta_s = {ETA0-eta_s:.1f}, то есть долю
{(tau.max()-tau.min())/(ETA0-eta_s):.2f} космологического времени.''')


if __name__ == '__main__':
    main()
