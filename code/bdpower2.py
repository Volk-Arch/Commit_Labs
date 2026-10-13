#!/usr/bin/env python3
"""СИЛА ЗАМЕРА, версия вторая: одна область при разных warp.

ПОЙМАННАЯ ОШИБКА (`bdpower.py`). Я сравнивал ⟨b⟩ ВНУТРИ сгущения против
СНАРУЖИ и получил нулевую точку, которая не нуль: при warp = 0 сдвиг доли
прямых уже −0.2598, и он почти не менялся с warp (−0.2598 → −0.2622).
Причина: `warped_diamond` строит ромб с временной протяжённостью `T` и
пространственным радиусом `T/2`. Я передал `T = 1.0` (чтобы `ℓ` совпадало
с `bdfield.py`), но оставил `sig = 0.35` из вчерашнего прогона, где было
`T = 2.0`. При `T = 1` радиус равен 0.5, так что «внутри 0.35» — это 70 %
мира, и деление внутри/снаружи стало делением «ядро против светового
края». Весь сдвиг был краевым.

ИСПРАВЛЕНО ДВУМЯ СПОСОБАМИ СРАЗУ:
  1. радиус пропорционально: `sig = 0.175` при `T = 1`;
  2. главное — сравнивать ОДНУ И ТУ ЖЕ область при разных warp, а не
     внутри против снаружи. Тогда краевого артефакта нет по построению:
     нулевая точка есть сам warp = 0, а сигнал — изменение.

ЗАЧЕМ ВСЁ ЭТО. `bdfield.py` прошёл калибровку (в плоском
`⟨b⟩ = −0.201 ± 0.439`, то есть нуль при `R = 0`, разброс 70.6 на
событие) и не нашёл отклика на разрежение (0.0…1.9σ, знак скачет).
Объявлять это нулём нельзя, пока не известна сила замера: `b` имеет
масштаб `1/ℓ²` = 135, значит кривизна порядка единицы дала бы ~55 на
событие, а кривизна 0.01 — 0.55, то есть на границе чувствительности.

ЧТО МЕРИМ. ⟨b⟩ в центральной области против силы warp, при совпадающей
высоте и без него (высота сама меняется с плотностью, поэтому печатаются
оба). Рядом — наведённая геометрия: связи на событие и доля прямых В ТОЙ
ЖЕ области, чтобы видеть, что warp действительно работает.

КОНТРОЛИ:
  warp = 0 — сама точка отсчёта;
  геометрия в той же области — обязана монотонно расти с warp, иначе
    наводить нечем;
  ⟨b⟩ по ВСЕМУ миру — должно двигаться слабее, чем в центре, иначе
    отклик не местный.

  python3 bdpower2.py
  python3 bdpower2.py --warps 0 2 4 8 16 --seeds 16
"""
import argparse
import math
import time

import numpy as np

from bdfield import bd_density, heights
from particles import rel_of
from response import warped_diamond


def main():
    ap = argparse.ArgumentParser(description='Сила замера, версия 2')
    ap.add_argument('--n', type=int, default=2400)
    ap.add_argument('--sig', type=float, default=0.175)
    ap.add_argument('--core', type=float, default=0.12)
    ap.add_argument('--warps', type=float, nargs='+', default=[0, 2, 4, 8])
    ap.add_argument('--seeds', type=int, default=14)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    print(f'N = {a.n} ТОЧНО при всех warp, T = 1 (радиус ромба 0.5)')
    print(f'сгущение цилиндра радиуса {a.sig} = {a.sig/ell:.1f} ell, '
          f'замер в ядре r < {a.core} = {a.core/ell:.1f} ell')
    print(f'ell = {ell:.4f}, масштаб поля 1/ell² = {1/ell**2:.0f}')
    print('СРАВНИВАЕТСЯ ОДНА ОБЛАСТЬ при разных warp — краевого '
          'артефакта нет')
    print()
    print(f"{'warp':>6}{'связей/соб.':>13}{'доля прям.':>12}{'событий':>9}"
          f"{'⟨b⟩ ядро':>11}{'ст.ош':>8}{'⟨b⟩ весь мир':>14}{'ст.ош':>8}")

    res = {}
    for w in a.warps:
        core, whole, ln, fr, nc = [], [], [], [], []
        hb = {}
        for s in range(a.seeds):
            rs = np.random.default_rng(21500 + 89 * s)
            X = warped_diamond(a.n, 1.0, w, a.sig, rs)
            R = rel_of(X)
            b = bd_density(R)
            h = heights(R)
            A = R.astype(np.float32)
            L = R & ~((A @ A) > 0.5)
            lk = (L.sum(0) + L.sum(1)).astype(float)
            fo = (L.sum(0) + L.sum(1)) / np.maximum(R.sum(0) + R.sum(1), 1)
            del A, L, R
            m = np.linalg.norm(X[:, 1:], axis=1) < a.core
            hmax = int(h.max())
            deep = (h >= 3) & (h <= hmax - 2)
            if (m & deep).sum() < 20:
                continue
            core.append(b[m & deep].mean())
            whole.append(b[deep].mean())
            ln.append(lk[m].mean()); fr.append(fo[m].mean())
            nc.append(int((m & deep).sum()))
            for hv in np.unique(h[m & deep]):
                hb.setdefault(int(hv), []).append(
                    b[m & deep & (h == hv)].mean())
            del X
        if not core:
            print(f'{w:>6.0f}  мало данных')
            continue
        c = np.array(core); wh = np.array(whole)
        sc = c.std(ddof=1) / math.sqrt(len(c))
        sw = wh.std(ddof=1) / math.sqrt(len(wh))
        res[w] = (c.mean(), sc, wh.mean(), sw, np.mean(ln), np.mean(fr),
                  hb)
        print(f'{w:>6.0f}{np.mean(ln):>13.1f}{np.mean(fr):>12.4f}'
              f'{np.mean(nc):>9.0f}{c.mean():>+11.2f}{sc:>8.2f}'
              f'{wh.mean():>+14.2f}{sw:>8.2f}', flush=True)

    if 0 in res and len(res) > 1:
        b0, s0 = res[0][0], res[0][1]
        l0, f0 = res[0][4], res[0][5]
        print()
        print('ОТКЛИК относительно warp = 0 (точка отсчёта)')
        print(f"{'warp':>6}{'Δ связей':>11}{'Δ доли прям.':>14}"
              f"{'Δ⟨b⟩ ядро':>12}{'ст.ош':>8}{'сигм':>7}"
              f"{'Δ⟨b⟩ на Δсвязь':>17}")
        for w in a.warps:
            if w == 0 or w not in res:
                continue
            m, se, _, _, ln, fr, _ = res[w]
            d = m - b0
            sd = math.hypot(se, s0)
            dl = ln - l0
            print(f'{w:>6.0f}{dl:>+11.1f}{fr-f0:>+14.4f}{d:>+12.2f}'
                  f'{sd:>8.2f}{abs(d)/max(sd,1e-9):>7.1f}'
                  f'{(d/dl if abs(dl) > 1e-9 else float("nan")):>+17.3f}')
        print()
        print('  геометрия растёт, а Δ⟨b⟩ не выходит за ошибку =>')
        print('  замер СЛЕП к кривизне, и вчерашний нуль ничего не значит')
        print('  Δ⟨b⟩ растёт с warp => замер видит, и нуль осмыслен')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
