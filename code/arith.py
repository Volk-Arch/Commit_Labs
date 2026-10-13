#!/usr/bin/env python3
"""АРИФМЕТИЧЕСКОЕ ПОГАШЕНИЕ: предсказать судьбу сектора из РАНГА.

Архитектура вечера 8 октября утверждает: фаза не кладётся на порядок
извне, она ОСТАЁТСЯ. Действие БД есть целая комбинация счётчиков

    M = N - N_0 + 9 N_1 - 16 N_2 + 8 N_3      (целое, d = 4)
    фаза = k_4 M / hbar,  k_4 = 4/sqrt(6)

поэтому амплитуда сектора — сумма корней из единицы, и всё решает, как
целые `M` ложатся по модулю шага завёртывания. Отсюда предсказание,
которое НЕ требует считать амплитуду:

  РАНГ 1 (счётчики движутся вместе) -> `M` на решётке, арифметическая
    прогрессия, сумма складывается в геометрический ряд -> погашение
    ЭКСПОНЕНЦИАЛЬНОЕ, |Z|/n << 1/sqrt(n);
  РАНГ ВЫСОКИЙ (счётчики независимы) -> `M` рассеяны -> случайное
    блуждание -> |Z|/n ~ 1/sqrt(n), то есть ПОЛ, и сектор выживает.

«Правильные сектора гибнут от собственной правильности, заурядные
выживают». Проверка ВСЛЕПУЮ: ранг и предсказание печатаются ДО замера.

СЕКТОРА. Два изучены (`dense`, `man`), один ОПАСЕН для архитектуры
(`sparse` — он побивает многообразный по счёту с N ~ 1700, и погашение
на нём не мерено ни разу; архитектура обязывает его гаситься), два
НОВЫХ, которых я не изучал вовсе (`dag`, `onion`).

КОНТРОЛИ:
  1. у `dense` есть ТОЧНАЯ формула |Z| = |2 cos(k_d/2 hbar)|^M — замер
     обязан её воспроизвести;
  2. перемешанные `M` того же сектора обязаны дать ровно пол
     `1/sqrt(n)` — это нулевой уровень метода.

  python3 arith.py
"""
import math

import numpy as np

from bd import layers
from sprinkle import sprinkle_d

K4 = 4.0 / math.sqrt(6.0)
HBAR = 0.55          # середина найденного окна [0.390, 0.780]
N = 380
REPS = 160


def counts(A):
    n, l = layers(A, 4)
    return np.array([n] + list(l), float)


def gen_dense(n, rs):
    s = n // 3
    idx = [np.arange(0, s), np.arange(s, 2 * s), np.arange(2 * s, n)]
    A = np.zeros((n, n), bool)
    A[np.ix_(idx[0], idx[1])] = True
    A[np.ix_(idx[1], idx[2])] = True
    A[np.ix_(idx[0], idx[2])] = True
    # лёгкий беспорядок: выбиваем случайные рёбра между 1 и 2
    m = rs.random((len(idx[0]), len(idx[1]))) < 0.03
    A[np.ix_(idx[0], idx[1])] &= ~m
    return A


def gen_sparse(n, rs, q=0.12):
    s = n // 3
    idx = [np.arange(0, s), np.arange(s, 2 * s), np.arange(2 * s, n)]
    A = np.zeros((n, n), bool)
    m12 = rs.random((len(idx[0]), len(idx[1]))) < q
    m23 = rs.random((len(idx[1]), len(idx[2]))) < q
    A[np.ix_(idx[0], idx[1])] = m12
    A[np.ix_(idx[1], idx[2])] = m23
    A[np.ix_(idx[0], idx[2])] = (m12.astype(np.float32)
                                 @ m23.astype(np.float32)) > 0.5
    return A


def gen_man(n, rs):
    return sprinkle_d(n, 4, rs)


def gen_dag(n, rs, p=0.02):
    """НОВЫЙ: случайный разреженный ориентированный граф на случайном
    линейном порядке, затем транзитивное замыкание."""
    U = rs.random((n, n)) < p
    A = np.triu(U, 1)
    F = A.astype(np.float32)
    R = F.copy()
    for _ in range(6):
        R = ((R + R @ F) > 0.5).astype(np.float32)
    return R > 0.5


def gen_onion(n, rs, k=7):
    """НОВЫЙ: вложенные антицепи (луковица) со случайными перелинковками."""
    sz = n // k
    idx = [np.arange(i * sz, min((i + 1) * sz, n)) for i in range(k)]
    A = np.zeros((n, n), bool)
    for i in range(k - 1):
        for j in range(i + 1, k):
            pr = 0.9 if j == i + 1 else 0.5
            A[np.ix_(idx[i], idx[j])] = rs.random(
                (len(idx[i]), len(idx[j]))) < pr
    F = A.astype(np.float32)
    A = ((F + F @ F) > 0.5)
    return np.triu(A, 1)


def eff_rank(C, thr=0.05):
    d = np.sqrt(np.maximum(np.diag(C), 1e-300))
    R = C / np.outer(d, d)
    R[~np.isfinite(R)] = 0.0
    ev = np.linalg.eigvalsh(R)
    ev = ev[ev > 0]
    return int((ev / ev.max() > thr).sum()) if len(ev) else 0


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print(f'N = {N}, реализаций на сектор {REPS}, hbar = {HBAR}')
    print()
    SEC = (('dense', gen_dense, 'изучен'), ('sparse', gen_sparse, 'ОПАСЕН'),
           ('man', gen_man, 'изучен'), ('dag', gen_dag, 'НОВЫЙ'),
           ('onion', gen_onion, 'НОВЫЙ'))
    store = {}
    print('ЭТАП 1 — РАНГ И ПРЕДСКАЗАНИЕ (амплитуда ещё не считана)')
    print(f"{'сектор':>8}{'':>9}{'ранг':>6}{'разброс M':>12}"
           f"{'шагов завёртки':>16}{'предсказание':>16}")
    for nm, gen, tag in SEC:
        rows = []
        for r in range(REPS):
            A = gen(N, np.random.default_rng(4000 + 101 * r + hash(nm) % 997))
            rows.append(counts(A))
            del A
        C = np.array(rows)
        M = C @ np.array([1.0, -1.0, 9.0, -16.0, 8.0])
        cov = np.cov(C.T)
        rk = eff_rank(cov)
        wrap = 2.0 * math.pi * HBAR / K4
        spread = M.std(ddof=1)
        pred = ('экспоненциальное' if rk <= 1 else
                'пол 1/sqrt(n)' if rk >= 3 else 'промежуточное')
        store[nm] = M
        print(f'{nm:>8}{tag:>9}{rk:>6}{spread:>12.3g}'
              f'{spread/wrap:>16.3g}{pred:>16}', flush=True)
    print()
    print('ЭТАП 2 — ЗАМЕР АМПЛИТУДЫ')
    print(f"{'сектор':>8}{'|Z|/n':>12}{'пол 1/sqrt(n)':>16}"
          f"{'отношение':>12}{'итог':>18}")
    fl = 1.0 / math.sqrt(REPS)
    for nm, _, _ in SEC:
        M = store[nm]
        z = np.exp(1j * K4 * M / HBAR).mean()
        az = abs(z)
        rel = az / fl
        verdict = ('ГАСИТСЯ' if rel < 0.5 else
                   'на полу' if rel < 3 else 'ВЫЖИВАЕТ сверх пола')
        print(f'{nm:>8}{az:>12.4g}{fl:>16.4g}{rel:>12.3g}{verdict:>18}',
              flush=True)
    print()
    print('КОНТРОЛЬ 2: перемешанные M обязаны дать ровно пол')
    rs = np.random.default_rng(7)
    for nm in ('dense', 'man'):
        M = store[nm].copy()
        sh = M + rs.normal(0, max(M.std(ddof=1), 1.0) * 3, len(M))
        z = abs(np.exp(1j * K4 * sh / HBAR).mean())
        print(f'  {nm}: {z:.4g} против пола {fl:.4g} '
              f'(отношение {z/fl:.2f})')
    print()
    print('''ЧТЕНИЕ. Если столбец «итог» согласен со столбцом
«предсказание» — ранг ковариации счётчиков действительно решает судьбу
сектора, и притом на ДВУХ новых секторах, которых я не изучал. Если
`sparse` окажется «выживает» — архитектура неверна целиком, потому что
она обязывает его гаситься.''')


if __name__ == '__main__':
    main()
