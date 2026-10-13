#!/usr/bin/env python3
"""СКАЛЯР РИЧЧИ ИЗ ДЕЙСТВИЯ: второй путь к кривизне.

ЗАЧЕМ. Скаляр `R` у нас измеряется ПЛОХО: обилия цепей дают его разностью
сильно скоррелированных величин, точность 120 % (`nullcon-doc.md`). А
действие БД есть скаляр Риччи НАПРЯМУЮ — теорема Бенинкасы–Доукера
говорит `⟨B·1(x)⟩ → −R(x)/2`, откуда, суммируя по событиям и помня, что
безразмерный счёт есть `ℓ²B`,

    S/N = ± ℓ² ⟨R⟩ / 2,     ⟨R⟩ — среднее по сприклингу.

Предсказание БЕЗ СВОБОДНЫХ ЧИСЕЛ. Если сойдётся, у нас появится второй,
независимый путь к кривизне, и как раз к той её части, которую первый
берёт хуже всего.

УСТРОЙСТВО. `curve.sprinkle` сыплет с постоянной ФИЗИЧЕСКОЙ плотностью 1
в конформных координатах (по `η` плотность идёт как `a⁴`), значит `ℓ = 1`
и `S/N = ⟨R⟩/2` прямо.

КОНТРОЛЬ, и он хорош. У ЛИНЕЙНОГО семейства `a = 1 + h(η−η₀)` скаляр
`R ≡ 0` ТОЖДЕСТВЕННО, хотя плотность по `η` меняется как `a⁴` и краевые
члены вместе с ней. Значит отклик в линейном плече есть загрязнение, а
не кривизна. Это та же схема, что дала 11.2 σ в `pencil.py`.

ОТКУДА БЕРЁТСЯ КРАЕВОЙ ЧЛЕН И ПОЧЕМУ ОН НЕ МЕШАЕТ. У ограниченной
области действие БД имеет краевой вклад, потому что звенья тянутся вдоль
всего светового конуса. Поэтому мерится НАКЛОН `S/N` по `⟨R⟩`, а краевой
член уходит в свободный член подгонки. Вдобавок берутся оба знака
кривизны: краевой вклад чётен по параметру, отклик — нечётен.

ЧЕМ МЕРИТЬ. Сглаженное действие: у него разброс в 200 раз меньше
резкого (`smear-doc.md`), а непрерывный предел тот же. Резкое считается
рядом для сверки. У сглаженного коэффициент может отличаться от ½ на
`O(ε)`, поэтому `ε` берётся двумя значениями.

  python3 ricci.py
"""
import math
import time

import numpy as np

from curve import a_of, curvature, sprinkle
from bd import action
from smear import smeared

T, L, ETA0 = 7.0, 7.1, 3.5


def causal(X, block=512):
    n = len(X)
    t = X[:, 0]
    Y = X[:, 1:]
    sq = (Y * Y).sum(axis=1)
    A = np.zeros((n, n), dtype=bool)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        d2 = sq[i0:i1, None] + sq[None, :] - 2.0 * (Y[i0:i1] @ Y.T)
        np.maximum(d2, 0.0, out=d2)
        dt = t[None, :] - t[i0:i1, None]
        A[i0:i1] = (dt > 0) & (dt * dt > d2)
    return A


def one(fam, par, rs):
    X = sprinkle(fam, par, L, T, ETA0, rs)
    n = len(X)
    if n < 200:
        return None
    A = causal(X)
    Rm = float(np.mean([curvature(e, fam, par, ETA0)[0] for e in X[:, 0]]))
    s4, _, _ = action(A, 4)
    out = (n, Rm, s4 / n,
           smeared(A, 0.3, 4) / n, smeared(A, 0.1, 4) / n)
    del A, X
    return out


def arm(fam, par, seeds, root):
    rows = [one(fam, par, np.random.default_rng(root + 101 * s))
            for s in range(seeds)]
    rows = [r for r in rows if r is not None]
    if len(rows) < 3:
        return None
    P = np.array(rows, float)
    return P.mean(axis=0), P.std(axis=0, ddof=1) / math.sqrt(len(P)), len(P)


def fit(xs, ys, es):
    """Взвешенная прямая y = A + B x; ошибки складным ножом по точкам."""
    x, y, e = map(np.asarray, (xs, ys, es))
    w = 1.0 / np.maximum(e, 1e-12) ** 2
    M = np.vstack([np.ones_like(x), x]).T
    C = np.linalg.inv(M.T @ (w[:, None] * M))
    b = C @ (M.T @ (w * y))
    jk = []
    for i in range(len(x)):
        k = np.arange(len(x)) != i
        Mk, wk = M[k], w[k]
        Ck = np.linalg.inv(Mk.T @ (wk[:, None] * Mk))
        jk.append((Ck @ (Mk.T @ (wk * y[k])))[1])
    jk = np.array(jk)
    se = math.sqrt((len(x) - 1) / len(x) * ((jk - jk.mean()) ** 2).sum())
    r = y - M @ b
    chi2 = float((w * r * r).sum())
    return float(b[0]), float(b[1]), se, chi2, len(x) - 2


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    SD = 14
    COLS = ('S/N резкое', 'S_ε/N, ε=0.3', 'S_ε/N, ε=0.1')
    for fam, pars, name in (
        ('quad', (-0.020, -0.010, -0.005, 0.0, 0.005, 0.010, 0.020),
         'КВАДРАТИЧНОЕ: R = 6·par/a³ ≠ 0 — СИГНАЛ'),
        ('lin', (-0.060, -0.030, 0.0, 0.030, 0.060),
         'ЛИНЕЙНОЕ: R ≡ 0 тождественно — КОНТРОЛЬ'),
    ):
        print('=' * 104)
        print(name)
        print(f"{'par':>9}{'N':>7}{'⟨R⟩':>10}" +
              ''.join(f'{c:>22}' for c in COLS))
        data = []
        for p in pars:
            r = arm(fam, p, SD, 4000 + int(1e5 * abs(p)) + (7 if p < 0 else 0))
            if r is None:
                print(f'{p:>9.3f}   мало событий')
                continue
            m, e, k = r
            print(f'{p:>9.3f}{m[0]:>7.0f}{m[1]:>10.4f}' +
                  ''.join(f'{m[2+i]:>13.4f}±{e[2+i]:<8.4f}' for i in range(3)))
            data.append((m[1], m[2], e[2], m[3], e[3], m[4], e[4]))
        print(f'  ({time.time()-t0:.0f} с, {SD} засевов на точку)', flush=True)
        if len(data) >= 3:
            D = np.array(data, float)
            print()
            print(f"  {'наклон d(S/N)/d⟨R⟩':>24}{'':>4}"
                  f"{'предсказание ±0.5':>20}{'χ²/степ':>12}")
            for i, c in enumerate(COLS):
                A_, B_, se, chi2, dof = fit(D[:, 0], D[:, 1 + 2 * i],
                                            D[:, 2 + 2 * i])
                flag = ('СОГЛАСИЕ' if abs(abs(B_) - 0.5) < 3 * se
                        else f'{(abs(B_)-0.5)/max(se,1e-9):+.1f} σ от ½')
                print(f'  {c:>24}{B_:>8.4f}±{se:<7.4f}{flag:>20}'
                      f'{chi2/max(dof,1):>12.2f}   своб. член {A_:+.4f}')
        print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
