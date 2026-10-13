#!/usr/bin/env python3
"""ПЕРЕБИВАЕТ ЛИ ВЕС ИЗ ПРОФИЛЯ ЭНТРОПИЮ: сравнение показателей.

Профиль обилий N_m/N_0 оказался единственной неимитируемой величиной
(`prostoe-doc.md`): ни подгонкой (одиночный враг 187.5 против 6.4 у
контроля), ни смешиванием (смесь состоит из тех же конфигураций, а их
доля ноль из сорока). Значит из него и надо строить вес - не из действия
БД, которое геометрию ПОДАВЛЯЕТ (`crit-geom-doc.md`, доля 0.973 ->
0.007 по beta).

Вес: S_prof(C) = хи2 профиля C до центра засева, в единицах
ПОКОНФИГУРАЦИОННОГО разброса засева. Локален (обилия интервалов) и
безразмерен (отношения).

Вопрос «перебивает ли энтропию» есть сравнение ДВУХ ПОКАЗАТЕЛЕЙ по N:
    подавление  ~ exp(-beta * хи2),       хи2 ~ N^p
    конкуренты  ~ 2^(N^2/4)  (Клейтман-Ротшильд), то есть N^2
Если p < 2, при любом ФИКСИРОВАННОМ beta энтропия побеждает при больших
N, и вес не может быть правилом отбора. Нужное beta тогда растёт как
N^(2-p) - а это уже не константа связи, а подгонка под размер.

Поэтому мерим p, а не рассуждаем о нём.

КОНТРОЛЬ: у второй партии засевов хи2 обязано СТОЯТЬ на ~8 при всех N
(это 8 условий), не расти. Если растёт - область построена неверно.
"""
import math

import numpy as np

from sectors import sprinkle_d
from lkdeg import layered
from ratios import nm

MS = (1, 2, 3, 4, 5, 6, 7, 8)


def prof(Rc):
    v, _ = nm(Rc)
    return v[1:] / max(v[0], 1.0)


def batch(mk, rep):
    return np.array([prof(mk()) for _ in range(rep)])


def main():
    print(__doc__)
    rs = np.random.default_rng(7071067)
    NS = (100, 200, 400, 800)
    rows = {}
    print(f"{'N':>6}{'контроль':>12}{'КР k=3 q=0.5':>15}"
          f"{'k=6 d=2':>12}{'k=13 d=1':>12}")
    for n in NS:
        rep = 40 if n <= 200 else (20 if n == 400 else 10)
        A = batch(lambda: sprinkle_d(n, 4, rs), rep)
        mu, sd = A.mean(axis=0), A.std(axis=0, ddof=1)

        def chi(X):
            return (((X - mu) / sd) ** 2).sum(axis=1)

        out = []
        B = batch(lambda: sprinkle_d(n, 4, rs), rep)
        out.append(float(np.median(chi(B))))
        for k, q in ((3, 0.5), (6, 2.0 * 6 / n), (13, 1.0 * 13 / n)):
            E = batch(lambda: layered(n, k, min(q, 0.95), rs),
                      max(6, rep // 3))
            out.append(float(np.median(chi(E))))
        rows[n] = out
        print(f'{n:>6}' + ''.join(f'{v:>15.1f}' if i == 1 else f'{v:>12.1f}'
                                  for i, v in enumerate(out)), flush=True)

    ns = np.array(NS, float)
    labs = ('контроль (засев)', 'КР k=3 q=0.5', 'k=6 delta=2',
            'k=13 delta=1')
    print('\nпоказатели роста хи2 ~ N^p (мнк по log-log):')
    for i, lab in enumerate(labs):
        y = np.array([rows[n][i] for n in NS], float)
        p = np.polyfit(np.log(ns), np.log(np.maximum(y, 1e-9)), 1)[0]
        print(f'  {lab:>18}: p = {p:+.2f}')
    print('\nэнтропия соответствует p = 2.00 (число порядков ~ 2^(N^2/4))')
    print('ЧТЕНИЕ: p < 2 => при фиксированном beta энтропия побеждает,')
    print('и нужное beta растёт как N^(2-p), то есть не константа связи.')


if __name__ == '__main__':
    main()
