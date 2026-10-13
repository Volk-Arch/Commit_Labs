#!/usr/bin/env python3
"""АМПЛИТУДНЫЙ СЛОЙ одним проходом: распределение, мера, протечка.

ОТКУДА. §17 `PICTURE.md` закрыл счётный слой: наблюдатель даёт координаты
и не возмущает безопорную структуру (зависимость от его скорости краевая,
гаснет как `T^-1.8`). Амплитудный слой остался непроверенным, и утром я
ошибочно объявил его закрытым стеной `bdwall-doc.md`.

ПОПРАВКА, на которой стоит этот замер. Стена стоит на ИЗВЛЕЧЕНИИ кривизны
из действия: надо вынуть сигнал `ℓ²R ≈ 0.002` из числа с разбросом 70 на
событие. Амплитуде извлекать нечего — `B` есть ТОЧНОЕ ЦЕЛОЕ, посчитанное
счётом, значит `exp(iB)` вычисляется без погрешности вовсе. И по модулю 8
веса `(1, −9, 16, −8)` сворачиваются (`9 ≡ 1`, `16 ≡ 0`, `8 ≡ 0`):

    B ≡ N − n₀ + n₁   (mod 8)      сверено на 40 мирах, расхождений 0

ТРИ ЧАСТИ ОДНИМ ПРОХОДОМ.

A. РАСПРЕДЕЛЕНИЕ. `⟨B⟩`, разброс, и распределение `B` по модулям 2, 4, 8
   с хи-квадратом против равномерного. Если остаток неравномерен, в
   действии есть арифметическая структура, которой никто не искал.

B. МЕРА СОРКИНА. Амплитуда на историю, `μ(A) = |Σ амплитуд|²`,
   интерференция `I₂(A,B) = μ(A∪B) − μ(A) − μ(B)`, и `I₃ = 0` как
   контроль арифметики. Разбиение на `A` и `B` — по БЕЗОПОРНОМУ
   порядковому признаку (доля пар выше/ниже медианы ансамбля).
   Согласованность: `|Σa|²/M`. Единица — случайный блуждатель, то есть
   интерференции нет; `M` — полная согласованность.
   Три выбора амплитуды: `exp(iB)`, `ω^B` при `ω = exp(2πi/8)` (тогда
   важен только остаток, и он точен), и `exp(iB/ħ)` со сканом по `ħ`.
   КОНТРОЛЬ: случайные целые вместо `B` дают эталон случайного
   блуждателя.

C. ПРОТЕЧКА В АМПЛИТУДУ. Вставляется наблюдатель по СОБСТВЕННОМУ времени
   (один объект при любой скорости) с быстротой 0 и 0.9. Меняется ли
   распределение `B mod 8`? Целые числа, значит статистика счётная.
   КОНТРОЛЬ, как в §17: растить ОБЛАСТЬ при постоянной плотности —
   краевая зависимость обязана убывать. Наращивать `N` при том же `T`
   нельзя, это измельчение (моя ошибка в первом контроле §17).

  python3 amplitude.py
  python3 amplitude.py --n 400 --m 600 --hbars 1 4 16 64
"""
import argparse
import math
import time

import numpy as np

W = (-1, 9, -16, 8)


def diamond(N, T, rs):
    out, got = [], 0
    while got < N:
        m = (N - got) * 9 + 256
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= r) & (T - Y[:, 0] >= r)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    X = np.vstack(out)[:N]
    return X[np.argsort(X[:, 0])]


def action_counts(X):
    """Точное целое B, доля пар, и счёты n0..n3. Всё одним счётом."""
    t = X[:, 0]
    dt = t[None, :] - t[:, None]
    S = X[:, 1:]
    g = (S * S).sum(1)
    d2 = g[:, None] + g[None, :] - 2.0 * (S @ S.T)
    np.maximum(d2, 0.0, out=d2)
    R = (dt > 0) & (dt * dt > d2)
    A = R.astype(np.float32)
    c = np.bincount((A @ A)[R].astype(np.int64), minlength=4)
    n = len(X)
    B = n + sum(W[m] * int(c[m]) for m in range(4))
    frac = float(R.sum()) / (n * (n - 1) / 2)
    return int(B), frac, [int(c[m]) for m in range(4)]


def observer_proper(dens, tau, rap, t0, x0):
    ch, sh = math.cosh(rap), math.sinh(rap)
    s = np.linspace(-tau / 2.0, tau / 2.0, dens)
    Z = np.zeros((dens, 4))
    Z[:, 0] = t0 + ch * s
    Z[:, 1] = x0 + sh * s
    return Z


def chi2_uniform(vals, mod):
    h = np.bincount(np.asarray(vals) % mod, minlength=mod)
    e = len(vals) / mod
    return float(((h - e) ** 2 / e).sum()), h


def measure(amps, mask):
    """μ(A) = |Σ амплитуд|² по подмножеству."""
    return float(abs(amps[mask].sum()) ** 2)


def part_b(Bs, fr, hbars):
    """Мера Соркина, интерференция, согласованность.

    ПОЙМАННАЯ ОШИБКА, найденная собственным контролем `I₃`. Тождество
    `I₃ = 0` верно только для ПОПАРНО НЕПЕРЕСЕКАЮЩИХСЯ подмножеств. В
    первой версии `mA` и `mB` были дополнительны (выше/ниже медианы), а
    `mC` бралось как каждый третий индекс и накрывало оба; строка
    `mC &= ~(mA & mB)` была пустой, потому что пересечение дополнительных
    множеств пусто. `I₃` вышло ±10¹…10² вместо машинного нуля — контроль
    сработал. Здесь разбиение на ТРИ непересекающиеся трети по доле пар.
    """
    Bs = np.asarray(Bs, float)
    M = len(Bs)
    fr = np.asarray(fr)
    q1, q2 = np.quantile(fr, [1 / 3.0, 2 / 3.0])
    mA = fr <= q1
    mB = (fr > q1) & (fr <= q2)
    mC = fr > q2
    assert not (mA & mB).any() and not (mA & mC).any() \
        and not (mB & mC).any(), 'трети обязаны не пересекаться'
    print(f'  три НЕПЕРЕСЕКАЮЩИЕСЯ трети по доле пар (безопорный признак):'
          f' {int(mA.sum())} / {int(mB.sum())} / {int(mC.sum())}')
    print(f"{'амплитуда':>22}{'μ(A)':>12}{'μ(B)':>12}{'μ(AuB)':>12}"
          f"{'I₂':>12}{'I₃':>11}{'соглас.':>10}")
    rows = []
    for nm, a in ([('exp(iB)', np.exp(1j * Bs)),
                   ('ω^B, ω=exp(2πi/8)', np.exp(2j * np.pi * Bs / 8.0))] +
                  [(f'exp(iB/{h:g})', np.exp(1j * Bs / h)) for h in hbars] +
                  [('КОНТРОЛЬ случайн.',
                    np.exp(1j * np.random.default_rng(9).uniform(
                        0, 2 * np.pi, M)))]):
        f = lambda m: measure(a, m)
        mu_a, mu_b, mu_c = f(mA), f(mB), f(mC)
        mu_ab = f(mA | mB)
        i2 = mu_ab - mu_a - mu_b
        i3 = (f(mA | mB | mC) - f(mA | mB) - f(mA | mC) - f(mB | mC)
              + mu_a + mu_b + mu_c)
        coh = f(mA | mB | mC) / M          # согласованность всего ансамбля
        rows.append((nm, coh))
        print(f'{nm:>22}{mu_a:>12.1f}{mu_b:>12.1f}{mu_ab:>12.1f}'
              f'{i2:>+12.1f}{i3:>+11.2e}{coh:>10.3f}')
    print('  I₃ обязано быть нулём машинной точности — контроль арифметики')
    print('  согласованность 1 = случайный блуждатель (интерференции нет),')
    print(f'  M = {M} = полная согласованность')
    return rows


def main():
    ap = argparse.ArgumentParser(description='Амплитудный слой')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--m', type=int, default=600)
    ap.add_argument('--hbars', type=float, nargs='+',
                    default=[4, 16, 64, 256])
    ap.add_argument('--dens', type=int, default=20)
    ap.add_argument('--tau', type=float, default=0.22)
    ap.add_argument('--ell', type=float, default=0.107)
    ap.add_argument('--ts', type=float, nargs='+', default=[1.0, 1.3, 1.6])
    ap.add_argument('--cseeds', type=int, default=10)
    a = ap.parse_args()

    t0 = time.time()
    print('=' * 70)
    print('ЧАСТЬ A: распределение точного целого действия')
    print('=' * 70)
    Bs, fr = [], []
    for s in range(a.m):
        X = diamond(a.n, 1.0, np.random.default_rng(40000 + 7 * s))
        B, f, c = action_counts(X)
        Bs.append(B); fr.append(f)
        del X
    Bs = np.array(Bs)
    print(f'миров {a.m} по {a.n} событий')
    print(f'  ⟨B⟩ = {Bs.mean():.1f} ± {Bs.std(ddof=1)/math.sqrt(a.m):.1f}'
          f'   разброс {Bs.std(ddof=1):.1f}'
          f'   размах {Bs.min()} .. {Bs.max()}')
    print()
    print(f"{'модуль':>8}{'хи-квадрат':>13}{'степеней':>10}"
          f"{'порог 5%':>11}{'равномерно?':>14}")
    thr = {2: 3.84, 4: 7.81, 8: 14.07, 16: 25.00}
    for mod in (2, 4, 8, 16):
        chi, h = chi2_uniform(Bs, mod)
        print(f'{mod:>8}{chi:>13.1f}{mod-1:>10}{thr[mod]:>11.2f}'
              f'{"да" if chi < thr[mod] else "НЕТ":>14}')
    chi8, h8 = chi2_uniform(Bs, 8)
    print(f'  распределение B mod 8: {list(h8)}')
    print()

    print('=' * 70)
    print('ЧАСТЬ B: мера Соркина и интерференция')
    print('=' * 70)
    part_b(Bs, fr, a.hbars)
    print()

    print('=' * 70)
    print('ЧАСТЬ C: протечка отсчёта в амплитуду')
    print('=' * 70)
    print(f'наблюдатель {a.dens} событий, собственное время {a.tau} — '
          f'ОДИН объект при любой быстроте')
    print(f'ell = {a.ell} фиксировано, растёт ОБЛАСТЬ (не измельчение)')
    print(f"{'T':>6}{'N':>8}{'засевов':>9}{'⟨ΔB⟩ ρ=0':>12}"
          f"{'⟨ΔB⟩ ρ=0.9':>13}{'разница':>11}{'сигм':>7}"
          f"{'хи² мод8':>10}")
    rows = []
    for T in a.ts:
        N = int(round(math.pi / 24.0 * T ** 4 / a.ell ** 4))
        ns = max(4, int(a.cseeds * (1.0 / T) ** 3))
        d0, d9, m0, m9 = [], [], [], []
        for s in range(ns):
            X = diamond(N, T, np.random.default_rng(41000 + 13 * s))
            B0 = action_counts(X)[0]
            for rap, dd, mm in ((0.0, d0, m0), (0.9, d9, m9)):
                Z = observer_proper(a.dens, a.tau, rap, T / 2.0, 0.0)
                Bw = action_counts(np.vstack([X, Z]))[0]
                dd.append(Bw - B0)
                mm.append((Bw - B0) % 8)
            del X
        d0 = np.array(d0, float); d9 = np.array(d9, float)
        df = d9.mean() - d0.mean()
        sd = math.hypot(d0.std(ddof=1) / math.sqrt(len(d0)),
                        d9.std(ddof=1) / math.sqrt(len(d9)))
        h0 = np.bincount(np.array(m0), minlength=8)
        h9 = np.bincount(np.array(m9), minlength=8)
        ch = float((((h0 - h9) ** 2) /
                    np.maximum(h0 + h9, 1)).sum())
        rows.append((T, abs(df), abs(df) / max(sd, 1e-9)))
        print(f'{T:>6.1f}{N:>8}{ns:>9}{d0.mean():>12.1f}{d9.mean():>13.1f}'
              f'{df:>+11.1f}{abs(df)/max(sd,1e-9):>7.1f}{ch:>10.1f}',
              flush=True)
    if len(rows) >= 3:
        A = np.array(rows)
        b = np.polyfit(np.log(A[:, 0]),
                       np.log(np.maximum(A[:, 1], 1e-9)), 1)[0]
        print(f'  |разница| ~ T^{b:+.3f}')
    print('  разница не выходит за ошибку, или убывает с T =>')
    print('  ПРОТЕЧКИ НЕТ и в амплитудном слое')
    print()
    print(f'время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
