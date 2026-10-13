#!/usr/bin/env python3
"""ПОЛЕ СЧЁТА: клеточный автомат на порядке, без геометрии в правиле.

ЗАЧЕМ ЭТО, А НЕ ОЧЕРЕДНОЙ ЗАМЕР. Его замечание: проверки ходят по кругу,
доказательство ради доказательства. Причина конкретна — **проект мерил
статичный объект**. Правило «A ≺ B, если A раньше по всем k шкалам» есть
определение, а не процесс: причинное множество из k шкал не растёт, а
вычисляется. Мерить его можно бесконечно, меняться там нечему
(`no-dynamics-doc.md`).

ЕГО ХОД, приложенный сюда. В своём проекте про мышление он спросил не
«как работают нейроны», а ЧТО ОНИ ПРОИЗВОДЯТ — вышло поле, — и построил
поле с операциями и правилами. Здесь тот же вопрос: что производит
порядок?

Сегодняшний замер (`locality-doc.md`) отвечает жёстко: порядок НЕ ВИДИТ
ни поперечника, ни площади, ни направления — буст меняет их в 4.88 раза
при расхождении порядковых величин 0.000 %. Единственное, что порядок
производит, — СЧЁТ: сколько со мной несравнимо, сколько в прошлом,
сколько в будущем, по скольким шкалам мы совпали. Все удачи проекта суть
функции счёта (`W = 2λr` — счёт несравнимых; `m/k` — счёт совпавших
шкал); все провалы — попытки построить то, чего счёт не видит.

Значит поле здесь — поле счёта.

УСТРОЙСТВО АВТОМАТА. Шкалы хранятся РАНГАМИ, не числами: этим
перешкалировочная симметрия `Diff+(R)^k` соблюдена тождественно, по
построению, а не проверкой.

Элементарный ход — обмен двух соседей по одной шкале. Он меняет
отношение РОВНО ОДНОЙ ПАРЫ, потому что между соседями по рангу никто не
стоит, так что сравнения со всеми третьими сохраняются. Пусть
`AG[i,j] = #{шкал, где i раньше j}`. Тогда `i ≺ j` ровно при `AG = k`, а
несравнимость при `0 < AG < k`. Обмен i и j в шкале s:

    AG[i,j] = k    -> стали НЕСРАВНИМЫ   (счёт обоих +1)
    AG[i,j] = 1    -> стали СРАВНИМЫ     (счёт обоих −1)
    иначе          -> ничего не меняется

Обновление за O(1): меняются только `AG[i,j]` и `AG[j,i]`.

ПРАВИЛО, без единого параметра: клетка тянет свой счёт несравнимых к
среднему по тем, с кем она СРАВНИМА (своё прошлое и будущее — больше у
неё ничего нет). «Пространство хочет быть однородным»; что не
выровнялось — вещество. Обмен принимается, если сдвигает счёт обеих
клеток в сторону их средних.

ПОЧЕМУ ПРОВЕРКА НЕ КРУГОВАЯ. Старт — СЛУЧАЙНЫЙ ЯЩИК из k независимых
перестановок. Он заведомо не Минковский (`not-minkowski-doc.md`), и его
доля пар равна `2^(1-k)` точно. Мерило — настоящий сприклинг в
четырёхмерный ромб того же N, и все мерила ПОРЯДКОВЫЕ: доля пар,
размерность Мирхейма–Мейера, высота, доля прямых, обилие интервалов.
Геометрия не входит ни в правило, ни в мерило.

    приходит к сприклингу с ДВУХ сторон -> аттрактор
    уезжает мимо или не останавливается  -> правило не то

КОНТРОЛИ:
  случайное принятие — обязано уехать к статистике ящика, не к сприклингу;
  нулевой ход — при выключенном правиле ничего не меняется;
  перешкалирование соблюдено тождественно (ранги).

  python3 cells.py
  python3 cells.py --n 1200 --ks 3 8 --sweeps 400
"""
import argparse
import math
import time

import numpy as np

from core import mm_dim


def box(N, k, rs):
    """Случайный ящик: k независимых перестановок. Доля пар = 2^(1-k)."""
    P = np.empty((N, k), np.int32)
    for s in range(k):
        P[:, s] = rs.permutation(N)
    return P


def ag_of(P):
    """AG[i,j] = число шкал, где i раньше j."""
    N, k = P.shape
    AG = np.zeros((N, N), np.int16)
    for s in range(k):
        AG += (P[:, s][:, None] < P[:, s][None, :]).astype(np.int16)
    return AG


def facts(AG, k):
    """Порядковые величины. R[i,j] = «i раньше j»."""
    R = AG == k
    N = len(AG)
    npair = N * (N - 1) / 2
    f = R.sum() / npair
    # прямые связи: нет посредника
    A = R.astype(np.float32)
    L = R & ~(A @ A > 0.5)
    h = np.ones(N, np.int64)
    for i in np.argsort(R.sum(0), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[:, i])
        if len(pre):
            h[i] = h[pre].max() + 1
    inc = (N - 1) - (R.sum(1) + R.sum(0))
    return dict(f=float(f), dim=mm_dim(float(f)) if 0 < f < 1 else float('nan'),
                hi=int(h.max()), lf=float(L.sum() / max(R.sum(), 1)),
                cv=float(inc.std() / max(inc.mean(), 1e-9)))


def census(AG, k):
    """Счёт несравнимых у каждой клетки."""
    N = len(AG)
    return (N - 1) - ((AG == k).sum(1) + (AG == 0).sum(1) - 0)


def sweep(P, AG, inv, k, rs, blind=False):
    """Один проход: N*k предложенных обменов при замороженных средних."""
    N = len(P)
    cmp_ = (AG == k)
    cmp_ = cmp_ | cmp_.T
    c = (N - 1) - cmp_.sum(1)
    deg = cmp_.sum(1)
    tot = cmp_.astype(np.float64) @ c.astype(np.float64)
    cbar = np.where(deg > 0, tot / np.maximum(deg, 1), c)
    acc = 0
    ss = rs.integers(0, k, N * k)
    rr = rs.integers(0, N - 1, N * k)
    for t in range(N * k):
        s, r = int(ss[t]), int(rr[t])
        i, j = int(inv[s, r]), int(inv[s, r + 1])
        a = int(AG[i, j])
        if a == k:
            d = +1.0                      # станут несравнимы
        elif a == 1:
            d = -1.0                      # станут сравнимы
        else:
            continue
        if not blind:
            gain = d * ((cbar[i] - c[i]) + (cbar[j] - c[j]))
            if gain <= 0:
                continue
        AG[i, j] -= 1
        AG[j, i] += 1
        P[i, s], P[j, s] = r + 1, r
        inv[s, r], inv[s, r + 1] = j, i
        c[i] += d
        c[j] += d
        acc += 1
    return acc


def sprinkling_facts(N, k, rs):
    """Мерило: настоящий сприклинг в четырёхмерный ромб, те же величины."""
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
    R = (dt > 0) & (dt > dx)
    AGs = np.where(R, k, 0).astype(np.int16)
    AGs[R.T] = 0
    AGs[~(R | R.T)] = 1          # несравнимые: 0 < AG < k
    return facts(AGs, k)


def main():
    ap = argparse.ArgumentParser(description='Поле счёта')
    ap.add_argument('--n', type=int, default=900)
    ap.add_argument('--ks', type=int, nargs='+', default=[3, 8])
    ap.add_argument('--sweeps', type=int, default=300)
    ap.add_argument('--every', type=int, default=50)
    a = ap.parse_args()

    t0 = time.time()
    rs = np.random.default_rng(31337)
    tgt = sprinkling_facts(a.n, 4, rs)
    print(f'N = {a.n}. Правило без параметров: счёт несравнимых тянется к')
    print('среднему по сравнимым. Шкалы — РАНГИ, перешкалирование')
    print('соблюдено тождественно. Геометрии в правиле нет.')
    print()
    print(f'МЕРИЛО — сприклинг в четырёхмерный ромб, N = {a.n}:')
    print(f"  доля пар {tgt['f']:.4f}   размерн. {tgt['dim']:.2f}   "
          f"высота {tgt['hi']}   доля прямых {tgt['lf']:.4f}   "
          f"разброс счёта {tgt['cv']:.4f}")
    print()

    for k in a.ks:
        print(f'=== k = {k} шкал.  старт: ящик, доля пар '
              f'{2.0**(1-k):.4f} (точно) ===')
        print(f"{'проход':>8}{'доля пар':>11}{'размерн.':>10}{'высота':>8}"
              f"{'доля прям.':>12}{'разброс счёта':>15}{'обменов':>9}")
        for blind, lbl in ((False, 'ПРАВИЛО'), (True, 'контроль: случайно')):
            rr = np.random.default_rng(777 + k)
            P = box(a.n, k, rr)
            AG = ag_of(P)
            inv = np.empty((k, a.n), np.int32)
            for s in range(k):
                inv[s, P[:, s]] = np.arange(a.n)
            print(f'  {lbl}')
            fc = facts(AG, k)
            print(f"{0:>8}{fc['f']:>11.4f}{fc['dim']:>10.2f}{fc['hi']:>8}"
                  f"{fc['lf']:>12.4f}{fc['cv']:>15.4f}{'—':>9}")
            for w in range(1, a.sweeps + 1):
                acc = sweep(P, AG, inv, k, rr, blind=blind)
                if w % a.every == 0 or w == a.sweeps:
                    fc = facts(AG, k)
                    print(f"{w:>8}{fc['f']:>11.4f}{fc['dim']:>10.2f}"
                          f"{fc['hi']:>8}{fc['lf']:>12.4f}{fc['cv']:>15.4f}"
                          f"{acc:>9}", flush=True)
        print()

    print('  с двух сторон приходит к мерилу => АТТРАКТОР;')
    print('  случайное принятие уезжает мимо => правило содержательно')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
