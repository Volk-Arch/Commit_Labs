#!/usr/bin/env python3
"""СИЛА ЗАМЕРА: сколько отклика даёт ИЗВЕСТНАЯ кривизна.

ЗАЧЕМ. `bdfield.py` прошёл калибровку — в плоском `⟨b⟩ = −0.201 ± 0.439`,
то есть нуль при `R = 0`, как обязано, при разбросе 70.6 на событие. И не
нашёл отклика на разрежение: 0.0…1.9σ, знак скачет (+0.6, +0.8, −1.7).

Объявлять это нулём нельзя, пока не измерена СИЛА замера. Прикидка: `b`
имеет масштаб `1/ℓ²`, при `ℓ = 0.086` это 135, значит кривизна порядка
единицы дала бы ~55 на событие — много выше чувствительности 0.44; а
кривизна 0.01 дала бы 0.55, то есть ровно на границе. Без этого числа
нуль ничего не означает.

ЧЕМ САДИТСЯ ИЗВЕСТНАЯ КРИВИЗНА. Архивным способом: переменная плотность
засева («warp», `gravity-doc.md`, `action-doc.md`). Это конформный
множитель, у него `R ≠ 0`. Архивная калибровка для сверки: при warp 0→8
связей на событие 15.06→23.76, а доля прямых падает 0.396→0.215.

ЧТО МЕРИМ.
  1. ⟨b⟩ внутри сгущения минус снаружи, при СОВПАДАЮЩЕЙ высоте, по
     ансамблю — против силы warp. Это и есть крутизна замера.
  2. Та же величина при warp = 0 — нулевая точка, обязана быть нулём.
  3. Перевод: какой warp отвечает разрежению из `bdfield.py` по сдвигу
     доли прямых. Тогда видно, был ли вчерашний нуль осмысленным.

КОНТРОЛИ:
  warp = 0 — нулевая точка;
  доля прямых и связи на событие печатаются рядом, чтобы видеть, что
    геометрия действительно наведена, а не только задумана;
  совпадающая высота — обязательна, краевые члены действия велики
    (в `bdfield.py` ⟨b⟩ на высоте 1-2 равно −0.611 при 24.5σ).

  python3 bdpower.py
  python3 bdpower.py --warps 0 2 4 8 --seeds 16
"""
import argparse
import math
import time

import numpy as np

from bdfield import bd_density, heights
from particles import rel_of
from response import warped_diamond


def matched(b, h, inside):
    """Внутри минус снаружи при совпадающей высоте, взвешенно."""
    num, den = 0.0, 0.0
    for hv in np.unique(h):
        m = h == hv
        x, y = b[m & inside], b[m & ~inside]
        if len(x) < 3 or len(y) < 8:
            continue
        num += (x.mean() - y.mean()) * len(x)
        den += len(x)
    return num / den if den > 0 else float('nan')


def geom(X):
    """Доля прямых и связей на событие внутри/снаружи — сверка с архивом."""
    R = rel_of(X)
    A = R.astype(np.float32)
    L = R & ~((A @ A) > 0.5)
    fr = (L.sum(0) + L.sum(1)) / np.maximum(R.sum(0) + R.sum(1), 1)
    ln = (L.sum(0) + L.sum(1)).astype(float)
    del R, A, L
    return fr, ln


def main():
    ap = argparse.ArgumentParser(description='Сила замера на поле БД')
    ap.add_argument('--n', type=int, default=2400)
    ap.add_argument('--sig', type=float, default=0.35)
    ap.add_argument('--warps', type=float, nargs='+', default=[0, 2, 4, 8])
    ap.add_argument('--seeds', type=int, default=14)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    print(f'N = {a.n} ТОЧНО при всех warp, сгущение цилиндра радиуса '
          f'{a.sig}')
    print(f'ell = {ell:.4f}, масштаб поля 1/ell² = {1/ell**2:.0f}')
    print(f'чувствительность из bdfield.py: ±0.44 по ансамблю '
          f'(разброс 70.6 на событие)')
    print('всё при СОВПАДАЮЩЕЙ высоте и по ансамблю')
    print()
    print(f"{'warp':>6}{'доля прям. вн':>15}{'снаруж':>9}{'сдвиг':>9}"
          f"{'связей вн':>11}{'снаруж':>9}"
          f"{'⟨b⟩ вн−снар':>14}{'ст.ошибка':>11}{'сигм':>7}")

    tab = []
    for w in a.warps:
        db, fi, fo, li, lo = [], [], [], [], []
        for s in range(a.seeds):
            rs = np.random.default_rng(21100 + 83 * s)
            X = warped_diamond(a.n, 1.0, w, a.sig, rs)
            R = rel_of(X)
            b = bd_density(R)
            h = heights(R)
            ins = np.linalg.norm(X[:, 1:], axis=1) < a.sig
            db.append(matched(b, h, ins))
            del R
            fr, ln = geom(X)
            fi.append(fr[ins].mean()); fo.append(fr[~ins].mean())
            li.append(ln[ins].mean()); lo.append(ln[~ins].mean())
            del X
        v = np.array([q for q in db if np.isfinite(q)])
        se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float('nan')
        sg = abs(v.mean()) / se if se and se > 1e-12 else float('nan')
        sh = np.mean(fi) - np.mean(fo)
        tab.append((w, sh, v.mean(), se, sg))
        print(f'{w:>6.0f}{np.mean(fi):>15.4f}{np.mean(fo):>9.4f}'
              f'{sh:>+9.4f}{np.mean(li):>11.1f}{np.mean(lo):>9.1f}'
              f'{v.mean():>+14.2f}{se:>11.2f}{sg:>7.1f}', flush=True)

    print('  архивный эталон: доля прямых 0.2973 внутри против 0.4462 '
          'снаружи при warp 8')
    print()
    base = [t for t in tab if t[0] == 0]
    if base and len(tab) > 1:
        print('КРУТИЗНА: отклик поля на единицу наведённой геометрии')
        print(f"{'warp':>6}{'сдвиг доли прям.':>18}{'⟨b⟩':>10}"
              f"{'⟨b⟩ на единицу сдвига':>24}")
        b0 = base[0][2]
        for w, sh, m, se, sg in tab:
            if abs(sh) < 1e-6:
                print(f'{w:>6.0f}{sh:>+18.4f}{m:>+10.2f}'
                      f'{"нулевая точка":>24}')
                continue
            print(f'{w:>6.0f}{sh:>+18.4f}{m:>+10.2f}'
                  f'{(m-b0)/sh:>+24.1f}')
        print()
        print('  Перевод для вчерашнего нуля: в bdfield.py дырка радиуса')
        print('  1.6 ell убрала 182 из 2400 событий. Если крутизна известна,')
        print('  видно, какой сдвиг геометрии это даёт и ждался ли отклик.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
