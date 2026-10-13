#!/usr/bin/env python3
"""ПОЛЕ СЧЁТА, версия вторая: правило без цели, доля смеси — найденная.

ЧТО БЫЛО И ПОЧЕМУ НЕ ВЫШЛО (`cells.py`). Механика автомата верна: шкалы
хранятся РАНГАМИ (перешкалирование `Diff+(R)^k` соблюдено тождественно,
а не проверкой), обмен двух соседей по одной шкале меняет отношение ровно
одной пары, обновление за O(1), слепой контроль делает 87–99 обменов за
проход.

Но правило замерзло: 0–2 обмена из 5600 предложений. Причина измерена:

    случайный ящик           разброс счёта 0.0145
    сприклинг (Минковский)                 0.1113

Случайный ящик УЖЕ минимизирует то, что я заставлял минимизировать.
«Пространство хочет быть однородным» неверно — Минковский в семь раз
неоднороднее случайного. Я угадал цель, минимум которой есть
бесструктурное состояние; та же ошибка, что весь день до этого.

ПРАВИЛО БЕЗ ЦЕЛИ. В игре «Жизнь» нет минимизируемой величины, есть
правило. Здесь ходов ровно два, и оба — про шкалу, оказавшуюся «не как
все»:

    AG[i,j] = k   пара совпала по ВСЕМ шкалам -> может РАСЦЕПИТЬСЯ
                  (обмен вносит разногласие; счёт обоих +1)
    AG[i,j] = 1   пара разошлась по всем кроме одной -> может СЦЕПИТЬСЯ
                  (обмен убирает разногласие; счёт обоих −1)
    иначе         ход ничего не меняет

`AG[i,j]` = число шкал, где i раньше j; `i ≺ j` ровно при `AG = k`.
Правило: сцеплять с вероятностью `p`, расцеплять с вероятностью `1 − p`.
Больше в нём ничего нет — ни цели, ни соседства, ни геометрии.

ПОЧЕМУ `p` НЕ ПАРАМЕТР, А ЗАМЕР. Крайности схлопываются: `p = 1` ведёт к
полному порядку (все шкалы совпали), `p = 0` — к антицепи. Значит
существует `p`, при котором состояние СТОИТ, и эту долю не угадывают, а
находят по неподвижности. Дальше — главная проверка: совпадут ли при
найденной `p` ОСТАЛЬНЫЕ величины с настоящим сприклингом. Ручка одна,
мерил четыре.

МЕРИЛА — обилие интервалов, то есть `n0..n3` из действия
Бенинказы–Доукера. Это и стандартная проверка многообразности в теории
причинных множеств, и собственное действие проекта; все числа порядковые,
геометрия в них не входит. Плюс доля пар, доля прямых и высота.

КОНТРОЛИ:
  слепое принятие — обязано уехать к статистике ящика;
  старт с двух сторон (мало шкал и много) — если приходит в одно место,
    это аттрактор, а не память о начальных условиях;
  `p = 0` и `p = 1` — обязаны схлопнуться, иначе правило не работает.

  python3 cells2.py
  python3 cells2.py --ps 0.0 0.3 0.5 0.7 1.0 --sweeps 400
"""
import argparse
import math
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
    return AG


def abund(R):
    """n0..n3: сколько связанных пар имеют 0..3 события между собой."""
    A = R.astype(np.float32)
    B = A @ A                      # B[i,j] = |интервал| при R[i,j]
    nr = float(R.sum())
    if nr < 1:
        return [float('nan')] * 4, 0.0
    v = B[R]
    return [float((v == m).sum() / nr) for m in range(4)], nr


def facts(R):
    """Порядковые величины: доля пар, размерность, высота, прямые, обилие."""
    N = len(R)
    f = float(R.sum()) / (N * (N - 1) / 2)
    h = np.ones(N, np.int64)
    for i in np.argsort(R.sum(0), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[:, i])
        if len(pre):
            h[i] = h[pre].max() + 1
    ab, nr = abund(R)
    inc = (N - 1) - (R.sum(1) + R.sum(0))
    return dict(f=f, dim=mm_dim(f) if 0 < f < 1 else float('nan'),
                hi=int(h.max()), n0=ab[0], n1=ab[1], n2=ab[2], n3=ab[3],
                cv=float(inc.std() / max(inc.mean(), 1e-9)))


def sweep(P, AG, inv, k, p, rs, blind=False):
    """Проход: N*k предложений. Сцепить с вероятностью p, расцепить с 1-p."""
    N = len(P)
    ss = rs.integers(0, k, N * k)
    rr = rs.integers(0, N - 1, N * k)
    uu = rs.random(N * k)
    nj, nb = 0, 0
    for t in range(N * k):
        s, r = int(ss[t]), int(rr[t])
        i, j = int(inv[s, r]), int(inv[s, r + 1])
        a = int(AG[i, j])
        if a == 1:
            if not blind and uu[t] >= p:
                continue
            nj += 1
        elif a == k:
            if not blind and uu[t] >= 1.0 - p:
                continue
            nb += 1
        else:
            continue
        AG[i, j] -= 1
        AG[j, i] += 1
        P[i, s], P[j, s] = r + 1, r
        inv[s, r], inv[s, r + 1] = j, i
    return nj, nb


def target(N, rs):
    """Мерило: сприклинг в четырёхмерный ромб."""
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


def hdr():
    print(f"{'проход':>8}{'доля пар':>10}{'разм.':>7}{'выс.':>6}"
          f"{'n0':>8}{'n1':>8}{'n2':>8}{'n3':>8}"
          f"{'сцеп.':>8}{'расц.':>8}{'разброс':>9}")


def row(w, fc, nj, nb):
    print(f"{w:>8}{fc['f']:>10.4f}{fc['dim']:>7.2f}{fc['hi']:>6}"
          f"{fc['n0']:>8.4f}{fc['n1']:>8.4f}{fc['n2']:>8.4f}{fc['n3']:>8.4f}"
          f"{nj:>8}{nb:>8}{fc['cv']:>9.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description='Правило без цели')
    ap.add_argument('--n', type=int, default=700)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--k2', type=int, default=12)
    ap.add_argument('--ps', type=float, nargs='+',
                    default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument('--sweeps', type=int, default=250)
    ap.add_argument('--every', type=int, default=50)
    a = ap.parse_args()

    t0 = time.time()
    tg = target(a.n, np.random.default_rng(31337))
    print(f'N = {a.n}. Правило: сцепить с вероятностью p, расцепить с 1-p.')
    print('Ни цели, ни соседства, ни геометрии. Шкалы — ранги.')
    print()
    print(f'МЕРИЛО — сприклинг в четырёхмерный ромб, N = {a.n}:')
    print(f"  доля пар {tg['f']:.4f}  размерн. {tg['dim']:.2f}  "
          f"высота {tg['hi']}  обилие n0..n3 "
          f"{tg['n0']:.4f} {tg['n1']:.4f} {tg['n2']:.4f} {tg['n3']:.4f}  "
          f"разброс {tg['cv']:.4f}")
    print()

    for p in a.ps:
        print(f'=== p = {p:.2f},  k = {a.k} шкал (старт: доля пар '
              f'{2.0**(1-a.k):.4f}) ===')
        hdr()
        rr = np.random.default_rng(555)
        P = box(a.n, a.k, rr)
        AG = ag_of(P)
        inv = np.empty((a.k, a.n), np.int32)
        for s in range(a.k):
            inv[s, P[:, s]] = np.arange(a.n)
        row(0, facts(AG == a.k), 0, 0)
        for w in range(1, a.sweeps + 1):
            nj, nb = sweep(P, AG, inv, a.k, p, rr)
            if w % a.every == 0 or w == a.sweeps:
                row(w, facts(AG == a.k), nj, nb)
        print()

    print(f'=== аттрактор ли: тот же p от МНОГИХ шкал (k = {a.k2}, '
          f'доля пар {2.0**(1-a.k2):.5f}) ===')
    for p in a.ps:
        rr = np.random.default_rng(556)
        P = box(a.n, a.k2, rr)
        AG = ag_of(P)
        inv = np.empty((a.k2, a.n), np.int32)
        for s in range(a.k2):
            inv[s, P[:, s]] = np.arange(a.n)
        print(f'  p = {p:.2f}')
        hdr()
        for w in range(1, a.sweeps + 1):
            nj, nb = sweep(P, AG, inv, a.k2, p, rr)
            if w % a.every == 0 or w == a.sweeps:
                row(w, facts(AG == a.k2), nj, nb)

    print()
    print('  p=0 и p=1 обязаны схлопнуться; между ними ищем неподвижность,')
    print('  и при найденной p сверяем ОБИЛИЕ с мерилом — ручка одна,')
    print('  мерил четыре. Совпадение с двух сторон => аттрактор.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
