#!/usr/bin/env python3
"""СКАЛЯР РИЧЧИ ИЗ РЕЗКОГО ДЕЙСТВИЯ: нормировка однозначна.

ЧТО ПОКАЗАЛ `ricci.py`.
  * КОНТРОЛЬ ПРОШЁЛ: у линейного семейства `R ≡ 0`, и сглаженное `S/N`
    там ПОСТОЯННО (−0.1059…−0.1047 при ошибках ±0.002) на размахе `h`
    от −0.06 до +0.06, тогда как в квадратичном плече оно гуляет в 20
    раз сильнее. Действие отзывается на кривизну и НЕ отзывается на
    профиль плотности.
  * Наклон резкого действия `+3.21 ± 1.97` — знак верный, с ½ совместим
    (1.4 σ), но ошибка огромна.
  * У СГЛАЖЕННОГО `ε`-нормировка кривизнного члена не та же, что у члена
    со звеньями (наклон −0.26 при `ε = 0.1` против −4.41 при `ε = 0.3`),
    и я её не выводил. Поэтому сглаженным мерить коэффициент нельзя.

ЗДЕСЬ. Мерится РЕЗКИМ действием, у которого нормировка однозначна:

    S = (4/√6)(N − N₀ + 9N₁ − 16N₂ + 8N₃),
    Σ_x ℓ²B·1(x) = −S,   ⟨B·1⟩ → −R/2   ⟹   S/N = +ℓ²⟨R⟩/2.

Предсказание наклона: **+0.5**, без свободных чисел.

Шум глушится РАЗМАХОМ кривизны, а не только числом засевов: в
`ricci.py` было `⟨R⟩ ≤ 0.11`, а предел применимости — `Rℓ² ~ 1`. Короб
сделан ниже (`T = 4` вместо 7), поэтому при том же `par` множитель
`(η−η₀)²` меньше, `a` не уходит к нулю, и `par` можно поднять вчетверо.
Короб заодно стал площе (`8.5³ × 4`), значит световые конусы целиком
влезают по пространству и краевой вклад меньше.

КОНТРОЛЬ переделан: у линейного плеча `⟨R⟩ ≡ 0`, поэтому наклон по `⟨R⟩`
не определён (матрица плана вырождена — так `ricci.py` и упал). Нуль
проверяется иначе: отклик на САМ ПАРАМЕТР `h` должен быть нулевым.

  python3 ricci2.py
"""
import math
import time

import numpy as np

from curve import a_of, curvature, sprinkle
from bd import action
from ricci import causal, fit

T, L, ETA0 = 4.0, 8.5, 2.0


def one(fam, par, rs):
    X = sprinkle(fam, par, L, T, ETA0, rs)
    n = len(X)
    if n < 200:
        return None
    A = causal(X)
    s4, _, _ = action(A, 4)
    Rm = float(np.mean([curvature(e, fam, par, ETA0)[0] for e in X[:, 0]]))
    amin = float(a_of(X[:, 0], fam, par, ETA0).min())
    del A, X
    return n, Rm, s4 / n, amin


def arm(fam, par, seeds, root):
    rows = []
    for s in range(seeds):
        r = one(fam, par, np.random.default_rng(root + 101 * s))
        if r is not None:
            rows.append(r)
    if len(rows) < 5:
        return None
    P = np.array(rows, float)
    return P.mean(axis=0), P.std(axis=0, ddof=1) / math.sqrt(len(P)), len(P)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    SD = 110
    print('=' * 92)
    print('КВАДРАТИЧНОЕ: R = 6·par/a³ — СИГНАЛ')
    print(f"{'par':>9}{'засев':>7}{'N':>7}{'min a':>8}{'⟨R⟩':>9}"
          f"{'⟨R⟩ℓ²':>8}{'S/N':>18}")
    data = []
    for p in (-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10):
        r = arm('quad', p, SD, 9000 + int(1e4 * abs(p)) + (3 if p < 0 else 0))
        if r is None:
            print(f'{p:>9.3f}   мало событий')
            continue
        m, e, k = r
        print(f'{p:>9.3f}{k:>7}{m[0]:>7.0f}{m[3]:>8.3f}{m[1]:>9.4f}'
              f'{m[1]:>8.4f}{m[2]:>11.4f}±{e[2]:<6.4f}')
        data.append((m[1], m[2], e[2]))
        print(f'     ({time.time()-t0:.0f} с)', flush=True)
    print()
    if len(data) >= 3:
        D = np.array(data, float)
        A_, B_, se, chi2, dof = fit(D[:, 0], D[:, 1], D[:, 2])
        print(f'  наклон d(S/N)/d⟨R⟩ = {B_:+.4f} ± {se:.4f}')
        print(f'  предсказание БД    = +0.5000')
        print(f'  расхождение        = {(B_-0.5)/max(se,1e-12):+.2f} σ')
        print(f'  χ²/степ = {chi2/max(dof,1):.2f},  своб. член '
              f'{A_:+.4f} (краевой вклад)')
    print()
    print('=' * 92)
    print('ЛИНЕЙНОЕ: R ≡ 0 — КОНТРОЛЬ. Отклик на сам параметр h?')
    print(f"{'h':>9}{'засев':>7}{'N':>7}{'min a':>8}{'⟨R⟩':>9}{'S/N':>18}")
    nul = []
    for p in (-0.20, -0.10, 0.0, 0.10, 0.20):
        r = arm('lin', p, SD, 9500 + int(1e4 * abs(p)) + (3 if p < 0 else 0))
        if r is None:
            continue
        m, e, k = r
        print(f'{p:>9.3f}{k:>7}{m[0]:>7.0f}{m[3]:>8.3f}{m[1]:>9.4f}'
              f'{m[2]:>11.4f}±{e[2]:<6.4f}')
        nul.append((p, m[2], e[2]))
        print(f'     ({time.time()-t0:.0f} с)', flush=True)
    if len(nul) >= 3:
        Dn = np.array(nul, float)
        A0, B0, se0, c0, d0 = fit(Dn[:, 0], Dn[:, 1], Dn[:, 2])
        print()
        print(f'  наклон d(S/N)/dh = {B0:+.4f} ± {se0:.4f}  '
              f'= {B0/max(se0,1e-12):+.1f} σ от нуля')
        print(f'  (кривизны там нет, поэтому ноль обязателен; χ²/степ '
              f'{c0/max(d0,1):.2f})')
        if len(data) >= 3:
            print(f'  для сравнения: в квадратичном плече размах S/N = '
                  f'{np.ptp(np.array(data)[:,1]):.4f}, '
                  f'в линейном {np.ptp(Dn[:,1]):.4f}')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
