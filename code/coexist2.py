#!/usr/bin/env python3
"""ЕСТЬ ЛИ В КРИТИЧЕСКОМ АНСАМБЛЕ ГЕОМЕТРИЧЕСКАЯ ПОДСЕМЬЯ.

coexist.py остановился сам: при N=24 слоистый враг попадает в область
засева 16/60, величина не различает, и «доля геометрических» была бы
бессмысленной. При N=1000 разделение было полным (0/40). Стена та же:
критический сэмплер доходит до N~24, различитель требует сотен.

Но идею сосуществования можно спросить иначе, не требуя чистого
разделения. Если в точке перехода фазы сосуществуют, критический
ансамбль обязан содержать ПОДСЕМЬЮ конфигураций, более близких к
геометрии, чем их содержит ЧИСТАЯ энтропийная фаза при том же N. То есть
сравнивать надо не с идеалом, а с beta=0 - и тогда «протечка» 27%
становится общим фоном у обоих, а не помехой.

Комбинная ловушка, на которой умер прежний детектор бимодальности
(`crit5.py` мерил решётку дискретных значений действия), здесь не
срабатывает: расстояние до области засева - величина НЕПРЕРЫВНАЯ.

ЧТО РАЗЛИЧАЕТ ИСХОДЫ: доля при beta=0.04 заметно выше, чем при beta=0 =>
действие ПРИТЯГИВАЕТ к геометрии, и сосуществование есть; доля такая же
или ниже => действие к геометрии не притягивает, и «союз» фаз в этом
месте не обнаруживается.

КОНТРОЛИ: засев обязан попадать ~95%; доли считаются с биномиальной
ошибкой; доля принятых обменов печатается.
"""
import math

import numpy as np

from lkdeg import layered
from crit9 import Ch
from critgeom3 import box
from critmix import NEW

N = 24
MS = (1, 2, 3, 4)


def prof(Rc):
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    n0 = float((Rc & (Iv == 0)).sum())
    return np.array([float((Rc & (Iv == m)).sum()) / max(n0, 1.0)
                     for m in MS])


def crit_profiles(n, bets, nsweep, rs, every=6):
    ch = [Ch(n, b) for b in bets]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npr = len(pairs)
    burn = nsweep // 3
    out = [[] for _ in ch]
    sok = np.zeros(len(ch) - 1)
    for it in range(nsweep + burn):
        for c in ch:
            for _ in range(npr):
                i, j = pairs[rs.integers(npr)]
                c.step(i, j, rs)
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            if rs.random() < math.exp(min(
                    0.0, (a.beta - b.beta) * (a.S - b.S))):
                sok[k] += 1
                a.R, b.R = b.R, a.R
                a.Iv, b.Iv = b.Iv, a.Iv
                a.c, b.c = b.c, a.c
                a.S, b.S = b.S, a.S
                a.nrel, b.nrel = b.nrel, a.nrel
        if it >= burn and (it - burn) % every == 0:
            for k, c in enumerate(ch):
                out[k].append(prof(c.R))
    return out, sok / (nsweep + burn)


def frac(ch_vals, thr):
    k = int((ch_vals <= thr).sum())
    n = len(ch_vals)
    p = k / n
    return p, math.sqrt(max(p * (1 - p), 1e-9) / n), k, n


def main():
    print(__doc__)
    rs = np.random.default_rng(6082045)
    A = np.array([prof(box(N, 3, 1.56, rs)) for _ in range(160)])
    mu, sd = A.mean(axis=0), A.std(axis=0, ddof=1)

    def chi(X):
        return (((X - mu) / sd) ** 2).sum(axis=1)

    thr = float(np.percentile(chi(A), 95))
    B = np.array([prof(box(N, 3, 1.56, rs)) for _ in range(120)])
    p, e, k, n = frac(chi(B), thr)
    print(f'область: d=3 a=1.56, N={N}, порог хи2={thr:.1f}')
    print(f'КОНТРОЛЬ засев: {k}/{n} = {p:.3f}±{e:.3f} '
          f'=> {"ОК" if p > 0.85 else "СБОЙ"}')

    out, swr = crit_profiles(N, NEW, 900, rs)
    print(f'\nобмены мин {swr.min():.2f}, образцов на ступень '
          f'{len(out[0])}')
    print(f"{'beta':>6}{'доля в области':>18}{'медиана хи2':>14}")
    res = {}
    for i, b in enumerate(NEW):
        c = chi(np.array(out[i]))
        p, e, kk, nn = frac(c, thr)
        res[b] = (p, e)
        if b in (0.0, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.25):
            print(f'{b:>6.2f}{f"{kk}/{nn} = {p:.3f}±{e:.3f}":>18}'
                  f'{np.median(c):>14.1f}', flush=True)

    p0, e0 = res[0.0]
    p4, e4 = res[0.04]
    d = (p4 - p0) / math.hypot(e0, e4)
    print(f'\nСРАВНЕНИЕ beta=0.04 против beta=0 (чистая энтропия):')
    print(f'  {p4:.3f}±{e4:.3f} против {p0:.3f}±{e0:.3f} '
          f'=> разница {d:+.1f}σ')
    print('\nдля фона — слоистый враг при том же N:')
    for kk, dl in ((3, 1.0), (6, 2.0)):
        E = np.array([prof(layered(N, kk, min(dl * kk / N, .95), rs))
                      for _ in range(120)])
        p, e, a1, a2 = frac(chi(E), thr)
        print(f'  k={kk} delta={dl}: {a1}/{a2} = {p:.3f}±{e:.3f}')


if __name__ == '__main__':
    main()
