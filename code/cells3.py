#!/usr/bin/env python3
"""ВИЛКА: способно ли семейство вообще на Минковского.

ДВЕ ПРЕДЫДУЩИЕ ПОПЫТКИ И ЧЕМ ОНИ КОНЧИЛИСЬ.

`cells.py` — правило «счёт несравнимых тянется к среднему по сравнимым»
  замерзло: 0–2 обмена из 5600. Причина измерена: случайный ящик УЖЕ
  минимизирует разброс счёта (0.0145 против 0.1113 у сприклинга). То
  есть Минковский НЕОДНОРОДНЕЕ случайного, и «пространство хочет быть
  однородным» — цель, минимум которой есть бесструктурное состояние.

`cells2.py` — правило без цели (сцеплять с вероятностью p, расцеплять с
  1−p) не двинулось ни при каком p: доля пар 0.071…0.075 против
  стартовых 0.0730, и крайние p = 0, p = 1 не схлопнулись, а встали.
  Причина чисто механическая: обмен двух СОСЕДЕЙ по рангу меняет одно
  отношение из N²/2 ≈ 180000, а соседних по рангу пар всего
  k(N−1) ≈ 3000, из них в крайних состояниях ~100. Автомат не может
  дотронуться больше чем до 0.1 % матрицы — заморожен по построению.
  В «Жизни» переключение клетки меняет будущее всех соседей; здесь обмен
  не имеет плеча.

ЧТО ИСПРАВЛЕНО, и это арифметика, а не догадка. Ход должен менять O(N)
отношений. Клетка не меняется местами с соседом, а ПЕРЕСТАВЛЯЕТ СЕБЯ
ЦЕЛИКОМ в одной шкале: вынуть из порядка этой шкалы и вставить на новый
ранг. По-прежнему локально — клетка знает только свой счёт, — но плечо
появилось.

ЧТО ЗДЕСЬ СПРАШИВАЕТСЯ, И ПОЧЕМУ ИМЕННО ЭТО. Не «какое правило», а
«способно ли семейство в принципе». Берём два заведомо крайних правила —
сгущать до предела и разрежать до предела — и смотрим, лежит ли
сприклинг МЕЖДУ ними.

    сприклинг внутри вилки  -> промежуточное правило существует, искать
                               его осмысленно
    сприклинг снаружи       -> семейство не способно, и это твёрдый
                               дешёвый отрицательный ответ

Угадывать правило до этой проверки нельзя — это и была ошибка двух
предыдущих попыток.

БЫСТРЫЙ ХОД. Для клетки i и шкалы s все N кандидатов на новый ранг
перебираются сразу за O(N). Пусть `base[x] = AG[i,x]` без вклада шкалы s.
Тогда при вставке на ранг r

    i ≺ x  ровно когда base[x] = k−1 и ранг x больше r
    x ≺ i  ровно когда base[x] = 0   и ранг x не больше r

и оба считаются накопительной суммой по рангам. Точный перебор, не
выборка.

МЕРИЛА порядковые: доля пар, размерность Мирхейма–Мейера, высота и
обилие интервалов n0..n3 (оно же стандартная проверка многообразности и
оно же действие Бенинказы–Доукера проекта).

КОНТРОЛИ:
  слепое — вставка на случайный ранг: обязана остаться у ящика;
  два старта (мало шкал и много): крайние правила обязаны приходить к
    своим пределам независимо от старта;
  перешкалирование соблюдено тождественно (шкалы суть ранги).

  python3 cells3.py
  python3 cells3.py --n 600 --k 5 --sweeps 60
"""
import argparse
import time

import numpy as np

from core import mm_dim


def box(N, k, rs):
    P = np.empty((N, k), np.int32)
    for s in range(k):
        P[:, s] = rs.permutation(N)
    return P


def ag_of(P):
    N, k = P.shape
    AG = np.zeros((N, N), np.int16)
    for s in range(k):
        AG += (P[:, s][:, None] < P[:, s][None, :]).astype(np.int16)
    np.fill_diagonal(AG, 0)
    return AG


def abund(R):
    """n0..n3: доля связанных пар с 0..3 событиями между ними."""
    A = R.astype(np.float32)
    v = (A @ A)[R]
    nr = float(R.sum())
    if nr < 1:
        return [float('nan')] * 4
    return [float((v == m).sum() / nr) for m in range(4)]


def facts(R):
    N = len(R)
    f = float(R.sum()) / (N * (N - 1) / 2)
    h = np.ones(N, np.int64)
    for i in np.argsort(R.sum(0), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[:, i])
        if len(pre):
            h[i] = h[pre].max() + 1
    ab = abund(R)
    inc = (N - 1) - (R.sum(1) + R.sum(0))
    return dict(f=f, dim=mm_dim(f) if 0 < f < 1 else float('nan'),
                hi=int(h.max()), n0=ab[0], n1=ab[1], n2=ab[2], n3=ab[3],
                cv=float(inc.std() / max(inc.mean(), 1e-9)))


def cmp_profile(base, q, k, N):
    """Число сравнимых с i при вставке на каждый ранг r, все r сразу."""
    # i ≺ x: base = k-1 и ранг x > r  -> хвостовая сумма
    a = np.zeros(N + 1, np.int64)
    m1 = base == k - 1
    if m1.any():
        a[1:] = np.bincount(q[m1], minlength=N)
    hi = np.cumsum(a[::-1])[::-1][1:]          # hi[r] = #{m1, q > r}
    # x ≺ i: base = 0 и ранг x <= r  -> головная сумма
    b = np.zeros(N, np.int64)
    m0 = base == 0
    if m0.any():
        b = np.bincount(q[m0], minlength=N)
    lo = np.cumsum(b)                          # lo[r] = #{m0, q <= r}
    return hi + lo


def move(P, AG, inv, i, s, k, mode, rs):
    """Клетка i переставляет себя в шкале s. Меняет O(N) отношений."""
    N = len(P)
    cur = P[:, s]
    ind = (cur[i] < cur).astype(np.int16)
    base = (AG[i] - ind).astype(np.int64)
    base[i] = -1
    if mode == 'слепое':
        r = int(rs.integers(0, N))
    else:
        prof = cmp_profile(base, cur.astype(np.int64), k, N)
        r = int(np.argmax(prof) if mode == 'сгущать' else np.argmin(prof))
    order = inv[s]
    order = np.delete(order, np.flatnonzero(order == i)[0])
    order = np.insert(order, min(r, N - 1), i)
    inv[s] = order
    P[order, s] = np.arange(N, dtype=np.int32)
    newind = (P[i, s] < P[:, s]).astype(np.int64)
    row = base + newind
    row[i] = 0
    AG[i] = row.astype(np.int16)
    AG[:, i] = (k - row).astype(np.int16)
    AG[i, i] = 0


def sweep(P, AG, inv, k, mode, rs):
    N = len(P)
    for t in range(N):
        move(P, AG, inv, int(rs.integers(0, N)), int(rs.integers(0, k)),
             k, mode, rs)


def target(N, rs):
    out, got = [], 0
    while got < N:
        m = (N - got) * 9 + 128
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, 1, m)
        Y[:, 1:] = rs.uniform(-0.5, 0.5, (m, 3))
        rad = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= rad) & (1 - Y[:, 0] >= rad)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    X = np.vstack(out)[:N]
    X = X[np.argsort(X[:, 0])]
    dt = X[:, 0][None, :] - X[:, 0][:, None]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=2)
    return facts((dt > 0) & (dt > dx))


def main():
    ap = argparse.ArgumentParser(description='Вилка')
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--ks', type=int, nargs='+', default=[5, 12])
    ap.add_argument('--sweeps', type=int, default=40)
    ap.add_argument('--every', type=int, default=10)
    a = ap.parse_args()

    t0 = time.time()
    tg = target(a.n, np.random.default_rng(31337))
    print(f'N = {a.n}. Ход: клетка переставляет себя целиком в одной '
          f'шкале (меняет O(N) отношений).')
    print('Перебор всех N кандидатов за O(N), точный.')
    print()
    print(f'МЕРИЛО — сприклинг в четырёхмерный ромб, N = {a.n}:')
    print(f"  доля пар {tg['f']:.4f}  разм. {tg['dim']:.2f}  "
          f"выс. {tg['hi']}  n0..n3 {tg['n0']:.4f} {tg['n1']:.4f} "
          f"{tg['n2']:.4f} {tg['n3']:.4f}  разброс {tg['cv']:.4f}")
    print()

    for k in a.ks:
        print(f'=== k = {k} шкал, старт — ящик (доля пар '
              f'{2.0**(1-k):.5f}) ===')
        for mode in ('сгущать', 'разрежать', 'слепое'):
            rr = np.random.default_rng(909 + k)
            P = box(a.n, k, rr)
            AG = ag_of(P)
            inv = np.empty((k, a.n), np.int32)
            for s in range(k):
                inv[s] = np.argsort(P[:, s])
            print(f'  {mode}')
            print(f"{'проход':>8}{'доля пар':>10}{'разм.':>7}{'выс.':>6}"
                  f"{'n0':>8}{'n1':>8}{'n2':>8}{'n3':>8}{'разброс':>9}")
            fc = facts(AG == k)
            print(f"{0:>8}{fc['f']:>10.4f}{fc['dim']:>7.2f}{fc['hi']:>6}"
                  f"{fc['n0']:>8.4f}{fc['n1']:>8.4f}{fc['n2']:>8.4f}"
                  f"{fc['n3']:>8.4f}{fc['cv']:>9.4f}")
            for w in range(1, a.sweeps + 1):
                sweep(P, AG, inv, k, mode, rr)
                if w % a.every == 0 or w == a.sweeps:
                    fc = facts(AG == k)
                    print(f"{w:>8}{fc['f']:>10.4f}{fc['dim']:>7.2f}"
                          f"{fc['hi']:>6}{fc['n0']:>8.4f}{fc['n1']:>8.4f}"
                          f"{fc['n2']:>8.4f}{fc['n3']:>8.4f}"
                          f"{fc['cv']:>9.4f}", flush=True)
        print()

    print('  сприклинг ВНУТРИ вилки сгущать/разрежать => промежуточное')
    print('  правило существует; снаружи => семейство не способно')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
