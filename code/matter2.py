#!/usr/bin/env python3
"""ЛОКАЛЕН ЛИ ОБРАТНЫЙ ПРОПАГАТОР.

matter.py спросил неправильно. Мера ||диаг||/||внедиаг|| в d=2 наполовину
мерила долю несравнимых пар: там B K равно нулю по причинности, у засева и
у врага одинаково. Разделения не было (0.76-1.27), и это ничего не значило.

Правильная постановка нашлась при попытке починить меру. Пропагатор
Джонстона обращается в точности:
    K = I + Phi (I - b Phi)^{-1}   =>   K^{-1} = I - Phi (I + (1-b) Phi)^{-1}
(проверка ниже численно). Значит обратный оператор СУЩЕСТВУЕТ у любого
частичного порядка, и спрашивать надо не про существование, а про форму:

    ЛОКАЛЕН ли K^{-1}, то есть лежит ли он в линейной оболочке
    «слоёв» L_m = A .* [|I(x,y)| = m] при малых m?

У засева локальность = локальность в пространстве-времени: волновой
оператор есть по построению, значит невязка мала. У слоистого врага
собственного времени нет, |I| ничего не означает геометрически, значит
K^{-1} обязан размазаться по всем слоям.

ЭТО УСЛОВИЕ ИНОГО РОДА. Враг подгоняет значения статистик тремя
параметрами. Здесь подгонять нечего: невязка - это решение обратной
задачи на N^2 числах при M+1 свободных коэффициентах.

ЧЕСТНОСТЬ: обоим даётся ОДИНАКОВОЕ число базисных матриц. Невязка
нормирована на ||K^{-1} - I||, то есть на ту часть, которую и надо
объяснить слоями (I в базисе есть у всех даром).

КОНТРОЛИ:
  (a) формула для K^{-1} против прямого обращения;
  (b) невязка засева должна ПАДАТЬ с ростом числа слоёв M;
  (c) третий ансамбль (перколяция) как ориентир - не засев и не слоистый.
"""
import math

import numpy as np

from waveop import causal, sprinkle_11
from lkdeg import layered


def perc(n, p, rs):
    U = np.triu(rs.random((n, n)) < p, 1)
    Rc = U.copy()
    for k in range(n):                      # транзитивное замыкание
        Rc |= np.outer(Rc[:, k], Rc[k, :]) & np.triu(np.ones((n, n), bool), 1)
    return Rc


def kinv(Rc, rho, mass):
    n = Rc.shape[0]
    Phi = 0.5 * Rc.astype(float)
    b = -mass * mass / rho
    return np.eye(n) - Phi @ np.linalg.inv(np.eye(n) + (1.0 - b) * Phi)


def layers(Rc, M):
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    return [(Rc & (Iv == m)).astype(float) for m in range(M)]


def resid(Rc, rho, mass, M):
    """доля ||K^{-1}-I||, НЕ объяснённая слоями 0..M-1 (мнк)."""
    T = kinv(Rc, rho, mass) - np.eye(Rc.shape[0])
    Ls = layers(Rc, M)
    G = np.array([[float((a * b).sum()) for b in Ls] for a in Ls])
    r = np.array([float((a * T).sum()) for a in Ls])
    c = np.linalg.lstsq(G, r, rcond=None)[0]
    R = T - sum(ci * Li for ci, Li in zip(c, Ls))
    return float(np.sqrt((R ** 2).sum() / max((T ** 2).sum(), 1e-300)))


def main():
    print(__doc__)
    rs = np.random.default_rng(271828)

    n0 = 200
    Rc0 = causal(sprinkle_11(n0, rs))
    K = np.eye(n0) + 0.5 * Rc0 @ np.linalg.inv(
        np.eye(n0) + (4.0 / (n0 / 0.5)) * 0.5 * Rc0)
    err = float(np.abs(K @ kinv(Rc0, n0 / 0.5, 2.0) - np.eye(n0)).max())
    print(f'КОНТРОЛЬ (a): K * K^{{-1}} - I, наибольшее отклонение {err:.2e}'
          f' => {"ОК" if err < 1e-8 else "СБОЙ"}')

    ens = (('засев 1+1', lambda n: causal(sprinkle_11(n, rs))),
           ('слоистый k=3 q=0.2', lambda n: layered(n, 3, 0.2, rs)),
           ('слоистый k=5 q=0.1', lambda n: layered(n, 5, 0.1, rs)),
           ('слоистый k=12 q=0.4', lambda n: layered(n, 12, 0.4, rs)),
           ('перколяция p=0.08', lambda n: perc(n, 0.08, rs)))

    for n, rep in ((300, 5), (600, 3)):
        print(f'\nНЕВЯЗКА локальной аппроксимации K^{{-1}}, m=2, N={n}, '
              f'{rep} засевов')
        print(f"{'ансамбль':>22}" + ''.join(f'{f"M={M}":>10}'
                                            for M in (1, 2, 4, 8)))
        for lab, mk in ens:
            vs = []
            for M in (1, 2, 4, 8):
                xs = [resid(mk(n), n / 0.5, 2.0, M) for _ in range(rep)]
                vs.append(float(np.mean(xs)))
            print(f'{lab:>22}' + ''.join(f'{v:>10.4f}' for v in vs),
                  flush=True)

    print('\nЧТЕНИЕ: контроль (b) - невязка засева обязана падать с M.')
    print('Если у засева она заметно НИЖЕ, чем у врага, при равном M -')
    print('у засева локальный волновой оператор есть, у врага нет.')


if __name__ == '__main__':
    main()
