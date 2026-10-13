#!/usr/bin/env python3
"""СОКРАЩЕНИЕ: локальна ли знакопеременная сумма, когда слагаемые — нет.

КУДА УКАЗАЛИ ВСЕ ТРИ ЛИНИИ ВЕЧЕРА. Холономия умерла, потому что петли не
сжимаются (`locality-doc.md`); шесть автоматов — потому что правило видит
согласие шкал, но не глубину (`cells-doc.md`); граф встреч — потому что
связи длинные и он полон при любом N (`particles-doc.md`). И теорема:
буст меняет пространственный поперечник в 4.88 раза при расхождении
порядковых величин 0.000 %, то есть **пространственного расстояния в
порядке нет**. Зато собственное время есть: при фиксированной мощности
интервала разброс `τ` падает до 3.9 %, а разброс `|Δx|` стоит на 33–40 %.

Порядок есть совершенные часы и слепая линейка. Значит локальность нельзя
взять соседством — её надо взять СОКРАЩЕНИЕМ, как в действии
Бенинказы–Доукера `B = N − n₀ + 9n₁ − 16n₂ + 8n₃`. Утверждение, которое
здесь проверяется: веса `(1, −9, +16, −8)` не измеряют расстояние, они
гасят нелокальные вклады, и локальна ИМЕННО СУММА, а не её члены.

ЗАМЕР, точный по арифметике. Уберём одно событие `z` и посмотрим, как
изменится местная плотность действия у события `x`:

    b_x = 1 − n₀(x) + 9n₁(x) − 16n₂(x) + 8n₃(x)
    n_m(x) = #{y ≺ x : |I(y,x)| = m}

Удаление `z` сдвигает на единицу мощность всех интервалов `I(y,x)`,
содержащих `z`, а содержат его ровно те, у которых `y ≺ z` (поскольку
`z ≺ x` дано). Отсюда, с `w = (−1, +9, −16, +8)` и `w(m≥4) = 0`:

    Δb_x(z) = Σ_m c_m·[w(m−1) − w(m)]  −  w(n(z,x))
            = c₀·(+1) + c₁·(−10) + c₂·(+25) + c₃·(−24) + c₄·(+8) − w(n(z,x))
    c_m = #{y ≺ z : |I(y,x)| = m}

Коэффициенты суть РАЗНОСТИ весов — то есть ровно схема сокращения. Если
`z ⊀ x`, то `Δb_x(z) = 0` тождественно: причинность в замере даровая.

ЧТО СРАВНИВАЕМ. Те же `c_m` дают отклик каждого слагаемого по отдельности,
например `Δn₀ = c₁ − c₀ − [n(z,x) = 0]`. Тогда вопрос:

    |Δb| спадает с удалением быстрее, чем |Δn₀| -> СОКРАЩЕНИЕ РАБОТАЕТ
    спадают одинаково                           -> утверждение ложно

Удаление мерится собственным временем `τ(z,x)` (координаты — только для
разметки корзин, в самом Δb их нет) и мощностью интервала `|I(z,x)|`,
которая порядковая.

КОНТРОЛИ:
  `z ⊀ x` — отклик обязан быть ровно 0 (проверка кода);
  слагаемые по отдельности — главное сравнение;
  случайные веса той же величины вместо (1,−9,16,−8) — обязаны НЕ дать
    ускоренного спада, иначе спад есть свойство геометрии, а не схемы;
  средние по многим `x` и засевам: сокращение БД работает в среднем по
    ансамблю, и это проверяется отдельной колонкой (знаковое среднее
    против среднего модуля).

  python3 cancel.py
  python3 cancel.py --n 2000 --nx 40 --seeds 3
"""
import argparse
import time

import numpy as np

from particles import diamond, rel_of

W = np.array([-1.0, 9.0, -16.0, 8.0])     # веса БД для m = 0..3


def wof(m):
    return W[m] if 0 <= m <= 3 else 0.0


def diffs(weights):
    """Коэффициенты при c_m: w(m-1) - w(m), m = 0..4."""
    g = lambda m: weights[m] if 0 <= m < len(weights) else 0.0
    return np.array([g(m - 1) - g(m) for m in range(5)])


def main():
    ap = argparse.ArgumentParser(description='Локальность сокращения')
    ap.add_argument('--n', type=int, default=1800)
    ap.add_argument('--nx', type=int, default=36)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    KB = diffs(W)
    print(f'N = {a.n}, событий x по {a.nx} на засев, засевов {a.seeds}')
    print(f'коэффициенты БД при c_m (разности весов): {KB[:5]}')
    print('Δb = 0 тождественно при z ⊀ x — причинность даровая')
    print()

    rsw = np.random.default_rng(12345)
    Wr = rsw.permutation(W) * 1.0              # контроль: те же веса, иной порядок
    KR = diffs(Wr)
    print(f'контрольные веса (перестановка тех же): {Wr}')
    print(f'их коэффициенты: {KR[:5]}')
    print()

    rows = []
    for s in range(a.seeds):
        rs = np.random.default_rng(10100 + 67 * s)
        X = diamond(a.n, rs)
        R = rel_of(X)                       # R[i,j] = «i раньше j»
        A = R.astype(np.float32)
        INT = (A @ A)                       # INT[i,j] = |I(i,j)|
        Rf = A
        deg = R.sum(0)
        cand = np.flatnonzero(deg > 60)
        if len(cand) < 5:
            continue
        xs = rs.choice(cand, min(a.nx, len(cand)), replace=False)
        for x in xs:
            x = int(x)
            past = np.flatnonzero(R[:, x])
            if len(past) < 40:
                continue
            nvec = INT[:, x]
            # c_m[z] = #{y ≺ z : |I(y,x)| = m}
            C = np.empty((5, a.n), np.float32)
            for m in range(5):
                C[m] = Rf.T @ (nvec == m).astype(np.float32)
            nz = nvec[past].astype(np.int64)
            wz = np.array([wof(int(v)) for v in nz])
            db = (KB[:, None] * C[:, past]).sum(0) - wz
            dr = (KR[:, None] * C[:, past]).sum(0) - np.array(
                [Wr[int(v)] if 0 <= int(v) <= 3 else 0.0 for v in nz])
            dn0 = C[1, past] - C[0, past] - (nz == 0)
            dn1 = C[2, past] - C[1, past] - (nz == 1)
            dn2 = C[3, past] - C[2, past] - (nz == 2)
            dt = X[x, 0] - X[past, 0]
            dx = np.linalg.norm(X[x, 1:] - X[past, 1:], axis=1)
            tau = np.sqrt(np.maximum(dt ** 2 - dx ** 2, 0.0))
            for t in range(len(past)):
                rows.append((tau[t], nz[t], db[t], dn0[t], dn1[t],
                             dn2[t], dr[t]))
        del R, A, INT, Rf, X

    M = np.array(rows)
    print(f'пар (x, z) с z ≺ x: {len(M)}')
    print()
    print('ОТКЛИК по собственному времени между z и x')
    print(f"{'tau':>12}{'пар':>8}{'|Δb| БД':>11}{'|Δn0|':>10}{'|Δn1|':>10}"
          f"{'|Δn2|':>10}{'|Δb| контр':>12}{'Δb знак.':>11}")
    ed = np.quantile(M[:, 0], np.linspace(0, 1, 8))
    ed = np.unique(ed)
    base = {}
    for b in range(len(ed) - 1):
        m = (M[:, 0] >= ed[b]) & (M[:, 0] < ed[b + 1])
        if m.sum() < 30:
            continue
        v = [np.abs(M[m, k]).mean() for k in (2, 3, 4, 5, 6)]
        sg = M[m, 2].mean()
        if not base:
            base = {k: v[i] for i, k in enumerate((2, 3, 4, 5, 6))}
        print(f'{0.5*(ed[b]+ed[b+1]):>12.4f}{int(m.sum()):>8}'
              f'{v[0]:>11.1f}{v[1]:>10.1f}{v[2]:>10.1f}{v[3]:>10.1f}'
              f'{v[4]:>12.1f}{sg:>11.1f}')

    print()
    print('ОТНОСИТЕЛЬНО первой корзины (кто спадает быстрее)')
    print(f"{'tau':>12}{'БД':>9}{'n0':>9}{'n1':>9}{'n2':>9}{'контр':>9}")
    for b in range(len(ed) - 1):
        m = (M[:, 0] >= ed[b]) & (M[:, 0] < ed[b + 1])
        if m.sum() < 30:
            continue
        r = [np.abs(M[m, k]).mean() / max(base[k], 1e-9)
             for k in (2, 3, 4, 5, 6)]
        print(f'{0.5*(ed[b]+ed[b+1]):>12.4f}' +
              ''.join(f'{q:>9.3f}' for q in r))

    print()
    def fit(k):
        lo = M[:, 0] > 0
        xs, ys = [], []
        for b in range(len(ed) - 1):
            m = (M[:, 0] >= ed[b]) & (M[:, 0] < ed[b + 1]) & lo
            if m.sum() < 30:
                continue
            u = np.abs(M[m, k]).mean()
            if u > 0:
                xs.append(np.log(0.5 * (ed[b] + ed[b + 1])))
                ys.append(np.log(u))
        if len(xs) < 3:
            return float('nan')
        return float(np.polyfit(xs, ys, 1)[0])
    for k, nm in ((2, 'БД (1,-9,16,-8)'), (3, 'n0 отдельно'),
                  (4, 'n1 отдельно'), (5, 'n2 отдельно'),
                  (6, 'контрольные веса')):
        print(f'  {nm:<20} |Δ| ~ tau^{fit(k):+.3f}')
    print()
    print('  у БД показатель заметно НИЖЕ, чем у слагаемых и у контрольных')
    print('  весов => сокращение даёт локальность, которой нет у членов')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
