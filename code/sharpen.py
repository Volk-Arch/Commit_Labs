#!/usr/bin/env python3
"""УТОЧНЕНИЕ ПОКАЗАТЕЛЯ: откуда едет наклон и как его измерить честно.

ОТКУДА ВОПРОС. В `newton-doc.md` показатель спадания зависел от внешней
отсечки: −0.906 (5 полос), −0.953 (8 полос), −1.041 (6 полос), −1.238
(полосы за пределами мира). Систематика ±0.1. Его замечание: «скачущий
показатель вероятно зависит от чего-то у нас, и можно его сделать более
точным». Верно, и подозреваемый уже измерен.

ЧТО ПОРТИТ НАКЛОН. Контроль рассыпанного удаления не нуль, а почти
ПОСТОЯННАЯ добавка около −2·10⁻⁴: это чистый эффект «событий стало
меньше», без локализации. Дефицит дырки есть сумма настоящего поля и этой
добавки, а постоянная добавка в подгонке логарифмов искажает наклон.
Грубая проверка по краям таблицы N = 60000: с вычитанием контроля наклон
по крайним точкам выходит −1.014 вместо −0.953.

ТРИ УТОЧНЕНИЯ, все принципиальные, не косметические:

 1. ВЫЧИТАНИЕ БАЗЫ. Из дефицита дырки вычитается дефицит рассыпанного
    удаления ТОГО ЖЕ числа событий, засев в засев. Это «та же масса без
    локализации», то есть правильный нуль.
 2. СВОБОДНАЯ ДОБАВКА. Подгонка `дефицит = A / r^b + c` с тремя
    параметрами вместо подгонки логарифмов с неявным `c = 0`. Если `c`
    выходит нулём в пределах ошибки, добавки нет и логарифмическая
    подгонка была законна.
 3. МЕСТНЫЙ ПОКАЗАТЕЛЬ. Наклон в скользящем окне из трёх соседних полос.
    Чистую область находят САМИ ДАННЫЕ: где местный показатель выходит на
    полку, там и степенная область, а края (поверхность дырки изнутри,
    граница мира снаружи) отсекаются сами.

Полосы вдвое уже обычных (по половине длины дискретности), чтобы на
скользящее окно хватило точек.

КОНТРОЛИ сохранены: нулевая масса обязана дать ровно нуль; защита от
границы из `bigfall.py` действует; высота считается только по фону и
печатается как защита от возврата пойманной ошибки.

  python3 sharpen.py --n 8000 --seeds 40
  python3 sharpen.py --n 60000 --seeds 8
"""
import argparse
import math
import time

import numpy as np

# Мир строит ТОТ ЖЕ `one` из `bigfall.py` — не копия, а импорт, чтобы
# пойманная ошибка «высота по цепи» не могла вернуться через копипаст.
from bigfall import one


def profile(A, B, bands, dens):
    """Дефицит хода часов по полосам.

    ПОЙМАННАЯ ОШИБКА, из-за которой показатель и ехал. Возвращалась только
    РАЗНОСТЬ ходов. Но гравитационное замедление КРАТНО: ход умножается на
    (1 − GM/r). Значит

        разность = основа(r) x кратный дефицит(r),
        показатель разности = показатель основы + показатель закона,

    а основа `<h/tau>` у ромба сама зависит от r (у далёких событий
    прошлое обрезано световым конусом). То есть мерилась СУММА двух
    показателей, и всё, что меняло основу — размер мира, N, радиус
    дырки, — ехало прямо в «закон». Та же болезнь, что в ошибках №3 и
    №10: загрязнение числителем.

    Теперь возвращается и основа, и ОТНОСИТЕЛЬНЫЙ дефицит — безразмерный,
    ровно та величина, которую ОТО предсказывает равной −GM/r.
    """
    dif, base = [], []
    for lo, hi in bands:
        mA = A['ok'] & (A['rho'] >= lo * dens) & (A['rho'] < hi * dens)
        mB = B['ok'] & (B['rho'] >= lo * dens) & (B['rho'] < hi * dens)
        if mA.sum() < 15 or mB.sum() < 15:
            dif.append(np.nan); base.append(np.nan)
            continue
        rA = A['h'][mA] / np.maximum(A['tau'][mA], 1e-9)
        rB = B['h'][mB] / np.maximum(B['tau'][mB], 1e-9)
        dif.append(float(rB.mean() - rA.mean()))
        base.append(float(rA.mean()))
    return np.array(dif), np.array(base)


def fit_power(r, y, free_c=False):
    """Подгонка y = A/r^b (+ c). Возвращает b."""
    m = np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    if not free_c:
        b, a = np.polyfit(np.log(r[m]), np.log(y[m]), 1)
        return -b, 0.0
    # сетка по c, внутри — линейная подгонка логарифмов
    best = (np.nan, np.nan, np.inf)
    ymin = y[m].min()
    for c in np.linspace(-0.9 * ymin, 0.9 * ymin, 61):
        z = y[m] - c
        if (z <= 0).any():
            continue
        b, a = np.polyfit(np.log(r[m]), np.log(z), 1)
        res = ((np.log(z) - (b * np.log(r[m]) + a)) ** 2).sum()
        if res < best[2]:
            best = (-b, c, res)
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser(description='Уточнение показателя')
    ap.add_argument('--n', type=int, default=8000)
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--rad', type=float, default=0.077)
    ap.add_argument('--radell', type=float, default=0.0,
                    help='радиус дырки в ell (перебивает --rad): нужен, '
                         'чтобы при разных N объект был ОДИН И ТОТ ЖЕ')
    ap.add_argument('--dens', type=int, default=300)
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--step', type=float, default=0.5, help='шаг полос в ell')
    ap.add_argument('--blk', type=int, default=1500)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    if a.radell > 0:
        a.rad = a.radell * ell
    Rw = a.t / 2.0                      # радиус мира в координатах
    rmax = 0.80 * Rw
    nb = int(max(3, math.floor((rmax - a.rad) / (a.step * ell))))
    bands = [(a.rad + k * a.step * ell, a.rad + (k + 1) * a.step * ell)
             for k in range(nb)]
    rc = np.array([0.5 * (lo + hi) for lo, hi in bands])
    print(f'N = {a.n}, ell = {ell:.5f}, радиус мира {Rw} = {Rw/ell:.1f} ell')
    print(f'дырка {a.rad:.4f} = {a.rad/ell:.2f} ell')
    print(f'полос {nb} по {a.step} ell, до {rmax:.3f} = {rmax/ell:.1f} ell '
          f'(защита от границы)')
    print(f'засевов {a.seeds}')
    print()

    PH, PS, BZ, hm = [], [], [], []
    for s in range(a.seeds):
        rs = lambda: np.random.default_rng(95000 + 197 * s)
        A = one(a.n, a.t, rs(), 0.0, a.dens, blk=a.blk)
        B = one(a.n, a.t, rs(), a.rad, a.dens, blk=a.blk)
        S = one(a.n, a.t, rs(), a.rad, a.dens, scatter=True, blk=a.blk)
        dh, bz = profile(A, B, bands, a.dens)
        ds, _ = profile(A, S, bands, a.dens)
        PH.append(dh); PS.append(ds); BZ.append(bz)
        hm.append(B['hmax'])
        del A, B, S
        if s == 0 or (s + 1) % 10 == 0:
            print(f'  засев {s}: высота мира {hm[-1]} '
                  f'({time.time()-t0:.0f} с)', flush=True)
    PH = np.array(PH); PS = np.array(PS); BZ = np.array(BZ)
    print(f'  высота мира {np.mean(hm):.1f} — естественная, не цепь')
    print()

    yh = np.nanmean(PH, axis=0)
    ys = np.nanmean(PS, axis=0)
    yd = np.nanmean(PH - PS, axis=0)
    sd = np.nanstd(PH - PS, axis=0, ddof=1) / math.sqrt(len(PH))
    base = np.nanmean(BZ, axis=0)
    # ОТНОСИТЕЛЬНЫЙ дефицит — отношение считается ВНУТРИ засева, потом
    # усредняется (ошибка №10: среднее отношений не есть отношение средних)
    REL = (PH - PS) / np.where(np.isfinite(BZ) & (BZ != 0), BZ, np.nan)
    yr = np.nanmean(REL, axis=0)
    sr = np.nanstd(REL, axis=0, ddof=1) / math.sqrt(len(REL))
    print(f"{'r в ell':>9}{'РАЗНОСТЬ':>12}{'основа':>10}"
          f"{'ОТНОСИТ.':>12}{'ст.ош':>10}{'сигм':>7}")
    for k in range(nb):
        if not np.isfinite(yd[k]):
            continue
        print(f'{rc[k]/ell:>9.2f}{yd[k]:>+12.6f}{base[k]:>10.4f}'
              f'{yr[k]:>+12.6f}{sr[k]:>10.6f}'
              f'{abs(yr[k])/max(sr[k],1e-12):>7.1f}')

    bb, _ = fit_power(rc, base)
    print(f'  ОСНОВА сама идёт как r^{-bb:+.3f} — вот слагаемое, '
          f'которое сидело в «законе»')

    print()
    print('МЕСТНЫЙ ПОКАЗАТЕЛЬ в скользящем окне из трёх полос')
    print('Две шкалы напечатаны рядом НАРОЧНО: при росте N при постоянном T')
    print('они расходятся в разные стороны, и это разводит двух подозреваемых')
    print('(граница мира — по r/R; дискретность и размазка — по r/ell).')
    print(f"{'r/ell':>8}{'r/R мира':>11}{'разность':>12}"
          f"{'ОТНОСИТЕЛЬНЫЙ':>16}")
    loc = []
    for k in range(1, nb - 1):
        w = slice(k - 1, k + 2)
        bd, _ = fit_power(rc[w], -yd[w])
        br, _ = fit_power(rc[w], -yr[w])
        loc.append((rc[k] / ell, bd, br, rc[k] / Rw))
        print(f'{rc[k]/ell:>8.2f}{rc[k]/Rw:>11.3f}{-bd:>12.3f}'
              f'{-br:>16.3f}')

    L = np.array([x for x in loc if np.isfinite(x[2])])
    print()
    if len(L) >= 3:
        # полка: окно, где местный показатель наиболее постоянен
        best = None
        for i in range(len(L) - 2):
            for j in range(i + 2, len(L)):
                seg = L[i:j + 1, 2]
                if len(seg) < 3:
                    continue
                sc = seg.std() / math.sqrt(len(seg))
                if best is None or (sc < best[0] and len(seg) >= 3):
                    best = (sc, i, j, seg.mean(), seg.std())
        if best:
            sc, i, j, mu, sg = best
            print(f'  ПОЛКА: от {L[i,0]:.1f} до {L[j,0]:.1f} ell, '
                  f'{j-i+1} окон')
            print(f'  местный показатель на полке: '
                  f'{-mu:.3f} ± {sg/math.sqrt(j-i+1):.3f}')
    # ЗАРАНЕЕ ЗАДАННОЕ ОКНО, не выбранное по плоскости. Выбирать окно по
    # тому, где показатель ровнее, — это отбор по исходу: любая кривая
    # где-нибудь ровна, и такой «полке» верить нельзя. Поэтому границы
    # окна берутся из двух УЖЕ ИЗМЕРЕННЫХ шкал загрязнения:
    #   снизу  r >= 2.5 ell   — носитель размазки действия ровно 2.5 ell
    #                           (`bdfield.py`), внутри него профиль сглажен;
    #   сверху r <= 0.40 R    — схлопывание (`collapse.py`) показало, что
    #                           внешнее крутение идёт по r/R, и до 0.40
    #                           кривые ещё держатся вместе.
    # Ошибка — складным ножом ПО ЗАСЕВАМ: выбрасывается засев, показатель
    # пересчитывается, разброс по выбрасываниям даёт ошибку. Это ловит и
    # то, что полосы в одном засеве связаны общим миром.
    win = (rc / ell >= 2.5) & (rc <= 0.40 * Rw)
    print()
    print('ЗАРАНЕЕ ЗАДАННОЕ ОКНО (не по плоскости, а по двум шкалам '
          'загрязнения)')
    print(f'  r/ell >= 2.5 (размазка) и r/R <= 0.40 (граница)  ->  '
          f'r/ell от {rc[win].min()/ell:.2f} до {rc[win].max()/ell:.2f}, '
          f'{win.sum()} полос')
    if win.sum() >= 3:
        RELS = PS / np.where(np.isfinite(BZ) & (BZ != 0), BZ, np.nan)
        for lbl, M in (('разность      ', PH - PS),
                       ('ОТНОСИТ.      ', REL),
                       ('КОНТРОЛЬ рассып', RELS)):
            full, _ = fit_power(rc[win], -np.nanmean(M, axis=0)[win])
            jk = []
            for s in range(len(M)):
                keep = [i for i in range(len(M)) if i != s]
                b, _ = fit_power(rc[win],
                                 -np.nanmean(M[keep], axis=0)[win])
                if np.isfinite(b):
                    jk.append(b)
            jk = np.array(jk)
            n = len(jk)
            se = math.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum()) \
                if n > 2 else np.nan
            dev = (f'{abs(-full + 1)/se:.1f} сигм от -1'
                   if np.isfinite(se) and se > 0 else '')
            if lbl.startswith('КОНТРОЛЬ'):
                amp = (np.nanmean(np.abs(np.nanmean(PH - PS, 0)[win]))
                       / max(np.nanmean(np.abs(np.nanmean(PS, 0)[win])),
                             1e-12))
                dev = f'по величине сигнал больше в {amp:.0f} раз'
            print(f'  {lbl} r^{-full:+.3f} +- {se:.3f}   {dev}')
    else:
        print('  окно пусто: мир слишком мал, чтобы в нём было чистое место')

    print()
    bd_all, _ = fit_power(rc, -yd)
    br_all, _ = fit_power(rc, -yr)
    br_free, cfree = fit_power(rc, -yr, free_c=True)
    print()
    print('ПОЛНАЯ подгонка по всем полосам (для сравнения — она и давала')
    print('прежние «-0.95...-1.04», смешивая чистую область с краевой):')
    print(f'  разность:                 r^{-bd_all:+.3f}')
    print(f'  ОТНОСИТЕЛЬНЫЙ:            r^{-br_all:+.3f}')
    print(f'  он же со свободной добавкой: r^{-br_free:+.3f}'
          f'   c = {cfree:+.2e}')
    print()
    print('  ОТО слабого поля: -1')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
