#!/usr/bin/env python3
"""Суммарный индекс: равен ли нулю, и выживает ли при огрублении.

ЧТО УЖЕ ЕСТЬ (charge.py). Поле поперечных осей — настоящее директорное
поле с разреженными полуцелыми дефектами:

    ненулевых клеток        0.0212  против 0.3406 у случайного поля
    |индекс| >= 1           0.0000
    знаки                   65 плюс / 55 минус, 0.9σ от симметрии
    неоднозначных рёбер     0.0382  против 0.3513
    аддитивность 2x2        точно 1353, нарушено 0

Плюсы ПОЯВИЛИСЬ — архив находил одни минусы, причиной была ошибка знака
(поворот читался с оси времени), исправленная в topo.py. Дефекты бывают
обоих знаков и в равном числе: это рождение парами.

Аддитивность выполнилась и у случайного поля, то есть тестом не
является — она тождество разворачивания угла.

ОСТАЛИСЬ ДВА ЧИСЛА, И ОНИ РЕШАЮТ.

  1. СУММАРНЫЙ ИНДЕКС заплаты. У сохраняющегося заряда в вакууме он
     обязан быть нулём: сколько родилось плюсов, столько минусов.
     На 11 заплатах вышло +0.273 ± 0.118 — 2.3σ, слишком тонко.
     Нужна статистика.

  2. УСТОЙЧИВОСТЬ К ОГРУБЛЕНИЮ — настоящая проверка на масштаб, какой в
     charge.py не было. Одна и та же физическая область считается
     дважды: мелкой сеткой (2g при шаге h) и грубой (g при шаге 2h).
     У настоящего дефекта суммарный индекс области от масштаба не
     зависит. У шума — зависит.

  python3 charge2.py
  python3 charge2.py --seeds 8 --patches 24
"""
import argparse
import math
import time

import numpy as np

from topo import diamond, cone_rel, plane_of
from charge import field_grid, loop_on_grid, census


def main():
    ap = argparse.ArgumentParser(description='Суммарный индекс и масштаб')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--grid', type=int, default=5)
    ap.add_argument('--step', type=float, default=0.05)
    ap.add_argument('--minsize', type=int, default=25)
    ap.add_argument('--projmin', type=float, default=0.25)
    ap.add_argument('--patches', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    print(f'ромб T={a.t}, N={a.n}; мелкая сетка {4*a.grid+1} при шаге '
          f'{a.step}, грубая {2*a.grid+1} при шаге {2*a.step}')
    print('одна и та же физическая область, два масштаба')
    print(f'заплат на засев {a.patches}, засевов {a.seeds}')
    print()

    for mode in ('поле', 'случайное'):
        FIN, COA, NF, NC = [], [], [], []
        for t in range(a.seeds):
            rs = np.random.default_rng(2901000 + 977 * t)
            X = diamond(a.n, a.t, rs)
            R = cone_rel(X)
            fut = R.sum(0)
            i0 = int(np.argmax(fut * (X[:, 0] < 0.25 * a.t)))
            cand = np.flatnonzero(R[:, i0])
            if len(cand) < 300:
                del R, X
                continue
            rs2 = np.random.default_rng(1300 + t)
            cent = rs2.choice(cand, min(a.patches, len(cand)), replace=False)
            for j0 in cent:
                j0 = int(j0)
                try:
                    u, v = plane_of(X, i0, j0)
                except Exception:
                    continue
                if mode == 'поле':
                    thf = field_grid(X, R, i0, j0, u, v, 2 * a.grid,
                                     a.step, a.minsize, cand, a.projmin)
                    thc = field_grid(X, R, i0, j0, u, v, a.grid,
                                     2 * a.step, a.minsize, cand, a.projmin)
                else:
                    thf = rs2.uniform(-math.pi, math.pi,
                                      (4 * a.grid + 1, 4 * a.grid + 1))
                    thc = rs2.uniform(-math.pi, math.pi,
                                      (2 * a.grid + 1, 2 * a.grid + 1))
                bf, _ = loop_on_grid(thf, 0, 0, 4 * a.grid)
                bc, _ = loop_on_grid(thc, 0, 0, 2 * a.grid)
                if bf is None or bc is None:
                    continue
                cf = census(thf, 2 * a.grid)[0]
                cc = census(thc, a.grid)[0]
                FIN.append(bf); COA.append(bc)
                NF.append(int((np.abs(np.round(cf * 2) / 2) >= 0.5).sum()))
                NC.append(int((np.abs(np.round(cc * 2) / 2) >= 0.5).sum()))
            del R, X
        if not FIN:
            print(f'{mode}: не построилось')
            continue
        F = np.array(FIN); C = np.array(COA)
        se_f = F.std() / math.sqrt(len(F))
        se_c = C.std() / math.sqrt(len(C))
        d = F - C
        se_d = d.std() / math.sqrt(len(d))
        print(f'--- {mode}:  заплат {len(F)}')
        print(f'  суммарный индекс, мелкая сетка  {F.mean():+.4f} ± '
              f'{se_f:.4f}   ({abs(F.mean())/max(se_f,1e-12):.1f}σ от нуля)')
        print(f'  суммарный индекс, грубая сетка  {C.mean():+.4f} ± '
              f'{se_c:.4f}   ({abs(C.mean())/max(se_c,1e-12):.1f}σ от нуля)')
        print(f'  ПАРНАЯ разность мелкая минус грубая {d.mean():+.4f} ± '
              f'{se_d:.4f}   ({abs(d.mean())/max(se_d,1e-12):.1f}σ)')
        print(f'  совпало точно: {int((np.abs(d) < 1e-6).sum())}/{len(d)};'
              f'  корреляция {np.corrcoef(F, C)[0,1]:+.3f}')
        print(f'  дефектов в заплате: мелкая {np.mean(NF):.2f}, '
              f'грубая {np.mean(NC):.2f}')
        print()

    print('  сохранение: суммарный индекс НУЛЬ и не зависит от масштаба')
    print('  шум: суммарный индекс тоже около нуля, но корреляция между')
    print('  масштабами отсутствует — вот это и различает')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
