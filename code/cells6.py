#!/usr/bin/env python3
"""СВЯЗАННЫЙ АНСАМБЛЬ: плотность закреплена, форма — свободное предсказание.

ОТКУДА. `cells5.py` дал правило БЕЗ РУЧЕК («сделай любой доступный ход»),
и оно воспроизвело у четырёхмерного сприклинга обилие интервалов и
профиль слоёв (три засева на сторону):

    n0        правило 0.4810   мерило 0.4842   промах −0.6 %
    n1        правило 0.1416   мерило 0.1422   промах −0.4 %
    высота    правило 7.00     мерило 7.67     промах −8.7 %
    разброс   правило 0.1426   мерило 0.1151   промах +23.9 %
    ДОЛЯ ПАР  правило 0.1542   мерило 0.1045   промах +47.5 %

Промах один и конкретный: верная локальная структура при завышенной
плотности. Контроль формы области (`scratchpad/shape.py`) показал, что
списать его на форму нельзя: ни один цилиндр не ловит все величины
сразу, а ромб и цилиндр сами расходятся по n0 при равной доле пар — то
есть «четырёхмерное» есть полоса, и правило лежит рядом с ней, но что НА
ней, не доказано.

ЧТО ЗДЕСЬ. Отнимаем у правила свободу менять плотность. Закрепляется
ОДНО число — доля связанных пар, взятая у мерила. Всё остальное
(n0..n3, высота, профиль слоёв, разброс счёта) никто не трогает: это
свободные предсказания. Способ тот же, которым проект уже пользовался
(`weight-doc.md`, связанный ансамбль).

И тогда ставится главный вопрос начисто: при РАВНОЙ плотности сходятся
ли разные `k`? Если да — размерность приходит сама, а `k` был лишь
способом задать плотность.

СЧЁТ СВЯЗЕЙ ЗА O(N). Поскольку `AG[z,y] = k − AG[y,z]`, все записи с
индексом `y` определяются СТРОКОЙ `y`: столбец `y` равен `k` там, где
строка равна нулю. Ход меняет только строки `i` и `x`, поэтому

    C = A(i)+B(i) + A(x)+B(x) − ([AG[i,x]=k] + [AG[i,x]=0])
    A(y) = #{AG[y,·] = k},  B(y) = #{AG[y,·] = 0}

(вычитание — потому что записи (i,x) и (x,i) попадают в сумму дважды), и
полное число связей меняется на `C_после − C_до`.

КОНТРОЛИ:
  слепое — принимать ходы без разбора доступности при той же связи:
    обязано остаться у ящика по форме;
  ограничение проверяется печатью доли пар: обязана стоять на мерильной;
  четыре `k` при ОДНОЙ плотности — расхождение по форме и есть ответ;
  перешкалирование соблюдено тождественно (шкалы суть ранги).

  python3 cells6.py
  python3 cells6.py --ks 4 6 8 12 --sweeps 300
"""
import argparse
import time

import numpy as np

from cells3 import box, ag_of, facts, target


def nrel(AG, k):
    return int((AG == k).sum())


def lines(AG, i, x, k):
    """Вклад строк i и x в полное число связей."""
    ri, rx = AG[i], AG[x]
    c = (int((ri == k).sum()) + int((ri == 0).sum()) +
         int((rx == k).sum()) + int((rx == 0).sum()))
    a = int(AG[i, x])
    return c - ((1 if a == k else 0) + (1 if a == 0 else 0))


def do_swap(P, AG, inv, i, x, s, k):
    ri, rx = int(P[i, s]), int(P[x, s])
    P[i, s], P[x, s] = rx, ri
    inv[s, ri], inv[s, rx] = x, i
    for y in (i, x):
        row = (P[y, :][None, :] < P).sum(1).astype(np.int64)
        row[y] = 0
        AG[y] = row.astype(np.int16)
        AG[:, y] = (k - row).astype(np.int16)
        AG[y, y] = 0


def step(P, AG, inv, k, rs, S, want, blind=False):
    """Ход с ограничением: принимается, если не уводит от нужной плотности."""
    N = len(P)
    for _ in range(40):
        i, x = int(rs.integers(0, N)), int(rs.integers(0, N))
        if i == x:
            continue
        a = int(AG[i, x])
        if blind:
            s = int(rs.integers(0, k))
        elif a == k or a == 0:
            s = int(rs.integers(0, k))
        elif a == k - 1:
            s = int(np.flatnonzero(P[x, :] < P[i, :])[0])
        elif a == 1:
            s = int(np.flatnonzero(P[i, :] < P[x, :])[0])
        else:
            continue
        c0 = lines(AG, i, x, k)
        do_swap(P, AG, inv, i, x, s, k)
        c1 = lines(AG, i, x, k)
        Sn = S + (c1 - c0)
        if abs(Sn - want) <= abs(S - want):
            return Sn
        do_swap(P, AG, inv, i, x, s, k)      # откат
        return S
    return S


def prof(R):
    N = len(R)
    h = np.ones(N, np.int64)
    for i in np.argsort(R.sum(0), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[:, i])
        if len(pre):
            h[i] = h[pre].max() + 1
    return np.bincount(h)[1:] / N


def fmt(p, m=7):
    return ' '.join(f'{p[i]:.3f}' if i < len(p) else '  .  '
                    for i in range(m))


def main():
    ap = argparse.ArgumentParser(description='Связанный ансамбль')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--ks', type=int, nargs='+', default=[4, 6, 8, 12])
    ap.add_argument('--sweeps', type=int, default=250)
    ap.add_argument('--every', type=int, default=50)
    a = ap.parse_args()

    t0 = time.time()
    tg = target(a.n, np.random.default_rng(31337))
    want = int(round(tg['f'] * a.n * (a.n - 1) / 2))
    rs0 = np.random.default_rng(31337)
    out, got = [], 0
    while got < a.n:
        m = (a.n - got) * 9 + 128
        Y = np.empty((m, 4))
        Y[:, 0] = rs0.uniform(0, 1, m)
        Y[:, 1:] = rs0.uniform(-0.5, 0.5, (m, 3))
        rad = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= rad) & (1 - Y[:, 0] >= rad)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    X = np.vstack(out)[:a.n]
    X = X[np.argsort(X[:, 0])]
    dt = X[:, 0][None, :] - X[:, 0][:, None]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=2)
    tp = prof((dt > 0) & (dt > dx))

    print(f'N = {a.n}. Закреплено ОДНО число — доля связанных пар '
          f'{tg["f"]:.4f} ({want} пар).')
    print('Всё остальное — свободные предсказания, их никто не трогает.')
    print()
    print(f'МЕРИЛО (сприклинг в четырёхмерный ромб): n0..n3 '
          f'{tg["n0"]:.4f} {tg["n1"]:.4f} {tg["n2"]:.4f} {tg["n3"]:.4f}  '
          f'выс. {tg["hi"]}  разброс {tg["cv"]:.4f}')
    print(f'            профиль слоёв  {fmt(tp)}')
    print()
    print(f"{'k':>4}{'проход':>8}{'доля пар':>10}{'n0':>8}{'n1':>8}"
          f"{'n2':>8}{'n3':>8}{'выс.':>6}{'разброс':>9}   профиль слоёв")

    fin = {}
    for k in a.ks:
        rs = np.random.default_rng(6060 + k)
        P = box(a.n, k, rs)
        AG = ag_of(P)
        inv = np.empty((k, a.n), np.int32)
        for s in range(k):
            inv[s] = np.argsort(P[:, s])
        S = nrel(AG, k)
        for w in range(1, a.sweeps + 1):
            for _ in range(a.n):
                S = step(P, AG, inv, k, rs, S, want)
            if w % a.every == 0 or w == a.sweeps:
                R = AG == k
                fc = facts(R)
                print(f"{k:>4}{w:>8}{fc['f']:>10.4f}{fc['n0']:>8.4f}"
                      f"{fc['n1']:>8.4f}{fc['n2']:>8.4f}{fc['n3']:>8.4f}"
                      f"{fc['hi']:>6}{fc['cv']:>9.4f}   {fmt(prof(R))}",
                      flush=True)
        fin[k] = fc

    print()
    print('СВОДКА при ОДНОЙ плотности: сходятся ли разные k по форме')
    print(f"{'k':>4}{'доля пар':>10}{'n0':>9}{'n1':>9}{'n2':>9}{'n3':>9}"
          f"{'выс.':>6}{'разброс':>9}")
    for k in a.ks:
        fc = fin[k]
        print(f"{k:>4}{fc['f']:>10.4f}{fc['n0']:>9.4f}{fc['n1']:>9.4f}"
              f"{fc['n2']:>9.4f}{fc['n3']:>9.4f}{fc['hi']:>6}"
              f"{fc['cv']:>9.4f}")
    print(f"{'мер':>4}{tg['f']:>10.4f}{tg['n0']:>9.4f}{tg['n1']:>9.4f}"
          f"{tg['n2']:>9.4f}{tg['n3']:>9.4f}{tg['hi']:>6}{tg['cv']:>9.4f}")
    print()
    print('  форма совпала при разных k => РАЗМЕРНОСТЬ ПРИХОДИТ САМА,')
    print('  а k был лишь способом задать плотность')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
