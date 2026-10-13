#!/usr/bin/env python3
"""МИР ОДНОГО РАЗМЕРА: есть ли в задаче масштаб дискретности вообще.

ЕГО МЫСЛЬ. «А что если мир не увеличивается, а постоянен, точнее то, что
ты считаешь ростом мира, является чем-то другим, а сам мир одного размера
постоянно?»

Он прав, и в проекте это записано как пойманная ошибка №7, в которую я
вернулся через слова. Когда я «растил мир», я менял `N` при постоянном
`T` — это ИЗМЕЛЬЧЕНИЕ: координатный размер стоит, растёт только `R/ℓ`. А
`ℓ` — величина НЕ ВНУТРЕННЯЯ: чтобы её узнать, нужен объём вложения, то
есть то, чего у причинного множества нет.

И выведенный закон (`theory-doc.md`) это подтверждает:

    Δτ/τ = −a / (t + r)

Здесь нет `ℓ` вовсе — только отношения `a/T`, `t/T`, `r/T`. Закон
БЕЗМАСШТАБЕН, значит измельчение обязано не менять профиль ни на сколько.

ГДЕ Я ПОДМЕНИЛ. Все замеры шли при радиусе дырки, заданном в единицах
`ℓ` (`--radell 1.21`). То есть при «росте мира» дырка СЪЁЖИВАЛАСЬ
относительно мира: `a/T` падало с 0.0770 (N = 8000) до 0.0465
(N = 60000). Я менял объект, думая, что меняю мир.

ОПЫТ, КОТОРОГО НЕ БЫЛО НИ РАЗУ. Держать `a/T` ПОСТОЯННЫМ и менять `N`.

    если мир «одного размера» и закон безмасштабен — профили обязаны
    СОВПАСТЬ целиком: и амплитудой, и наклоном, при любом N;

    если разойдутся — в задаче есть настоящая зависимость от `ℓ`, то
    есть дискретность сама по себе что-то значит.

Раскладка по `r/T` — величине внутренней (радарное расстояние в номерах
цепи, делённое на длину той же цепи), а не по `r/ℓ`, которая требует
знания плотности.

КОНТРОЛИ:
  1. рассыпанное удаление того же числа событий — профиля быть не должно;
  2. нулевая дырка — ровно нуль;
  3. высота считается ТОЛЬКО по фону (пойманная ошибка про цепь);
  4. полосы одни и те же в единицах `r/T` для всех `N`, иначе сравнение
     подменится разницей отрезков — та самая ловушка, в которую я попал
     раньше.

  python3 samesize.py --seeds 10
"""
import argparse
import math
import time

import numpy as np

from bigfall import one


def profile(A, B, S, bands, dens):
    """Относительный дефицит хода часов по полосам r/T."""
    dif, base = [], []
    for lo, hi in bands:
        # rho в НОМЕРАХ цепи; доля от длины цепи = r/T, внутренняя величина
        mA = A['ok'] & (A['rho'] >= lo * dens) & (A['rho'] < hi * dens)
        mB = B['ok'] & (B['rho'] >= lo * dens) & (B['rho'] < hi * dens)
        mS = S['ok'] & (S['rho'] >= lo * dens) & (S['rho'] < hi * dens)
        if mA.sum() < 20 or mB.sum() < 20 or mS.sum() < 20:
            dif.append(np.nan); base.append(np.nan)
            continue
        rA = A['h'][mA] / np.maximum(A['tau'][mA], 1e-9)
        rB = B['h'][mB] / np.maximum(B['tau'][mB], 1e-9)
        rS = S['h'][mS] / np.maximum(S['tau'][mS], 1e-9)
        dif.append(float((rB.mean() - rS.mean())))
        base.append(float(rA.mean()))
    return np.array(dif), np.array(base)


def main():
    ap = argparse.ArgumentParser(description='Мир одного размера')
    ap.add_argument('--t', type=float, default=1.0)
    ap.add_argument('--radT', type=float, default=0.077,
                    help='радиус дырки в долях T — ПОСТОЯНЕН по N')
    ap.add_argument('--dens', type=int, default=300)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--ns', type=str, default='8000,20000,60000')
    ap.add_argument('--blk', type=int, default=1500)
    a = ap.parse_args()

    t0 = time.time()
    Ns = [int(x) for x in a.ns.split(',')]
    rad = a.radT * a.t
    # полосы в долях T — ОДНИ И ТЕ ЖЕ для всех N
    edges = np.arange(a.radT + 0.02, 0.40, 0.03)
    bands = list(zip(edges[:-1], edges[1:]))
    bc = np.array([0.5 * (lo + hi) for lo, hi in bands])
    print('ОПЫТ: радиус дырки ПОСТОЯНЕН в долях T, меняется только N.')
    print('Если закон безмасштабен — профили обязаны совпасть целиком.')
    print()
    print(f'дырка a/T = {a.radT} (постоянна), T = {a.t}')
    print(f'полосы r/T: ' + ' '.join(f'{c:.3f}' for c in bc))
    print(f'засевов {a.seeds}, N: {Ns}')
    print()
    print(f"{'N':>8}{'ell':>9}{'a/ell':>8}{'R/ell':>8}" +
          ''.join(f'{c:>9.3f}' for c in bc) + f"{'наклон':>9}")

    res = {}
    for N in Ns:
        ell = (math.pi / 24.0 / N) ** 0.25
        acc = []
        for s in range(a.seeds):
            g = lambda: np.random.default_rng(41000 + 167 * s)
            A = one(N, a.t, g(), 0.0, a.dens, blk=a.blk)
            B = one(N, a.t, g(), rad, a.dens, blk=a.blk)
            S = one(N, a.t, g(), rad, a.dens, scatter=True, blk=a.blk)
            d, b = profile(A, B, S, bands, a.dens)
            acc.append(d / b)
            del A, B, S
        M = np.array(acc)
        y = np.nanmean(M, axis=0)
        se = np.nanstd(M, axis=0, ddof=1) / math.sqrt(len(M))
        ok = np.isfinite(y) & (y < 0)
        sl = (-np.polyfit(np.log(bc[ok]), np.log(-y[ok]), 1)[0]
              if ok.sum() >= 3 else np.nan)
        res[N] = (y, se, sl)
        print(f'{N:>8}{ell:>9.5f}{rad/ell:>8.2f}{0.5/ell:>8.1f}' +
              ''.join(f'{v:>9.4f}' if np.isfinite(v) else f'{"—":>9}'
                      for v in y) + f'{-sl:>9.3f}')
        print(f'{"":>33}' + ''.join(f'{v:>9.4f}' for v in se) +
              f'{"ст.ош":>9}', flush=True)

    print()
    print('СОВПАДАЮТ ЛИ ПРОФИЛИ (главный вопрос)')
    keys = sorted(res)
    print(f"{'полоса r/T':>12}" + ''.join(f'{k:>11}' for k in keys) +
          f"{'размах/ст.ош':>14}")
    worst = 0.0
    for q in range(len(bc)):
        vals = [res[k][0][q] for k in keys]
        ses = [res[k][1][q] for k in keys]
        if not all(np.isfinite(v) for v in vals):
            continue
        spread = max(vals) - min(vals)
        sed = math.sqrt(sum(s * s for s in ses))
        worst = max(worst, abs(spread) / max(sed, 1e-12))
        print(f'{bc[q]:>12.3f}' + ''.join(f'{v:>11.4f}' for v in vals) +
              f'{abs(spread)/max(sed,1e-12):>14.1f}')
    print()
    sls = [res[k][2] for k in keys]
    print(f'  наклоны: ' + ', '.join(f'{-s:.3f}' for s in sls) +
          f'   размах {max(sls)-min(sls):.3f}')
    print(f'  худшее расхождение по полосам: {worst:.1f} сигм')
    if worst < 3:
        print('  => ПРОФИЛИ СОВПАЛИ: масштаба дискретности в задаче нет,')
        print('     измельчение ничего не меняет, «размер мира» — не')
        print('     параметр. Мир и правда один.')
    else:
        print('  => ПРОФИЛИ РАЗОШЛИСЬ: зависимость от ell настоящая,')
        print('     дискретность сама по себе что-то значит.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
