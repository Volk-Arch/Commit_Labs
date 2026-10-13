#!/usr/bin/env python3
"""АТТРАКТОР: одна ручка, четыре мерила. Долю смеси ищем, не угадываем.

ЦЕПОЧКА (каждая попытка исправляла предыдущую; все три нужны, чтобы эта
была осмысленна).

`cells.py`  правило «счёт тянется к среднему по сравнимым» замерзло.
            Причина измерена: случайный ящик УЖЕ минимизирует разброс
            счёта (0.0145 против 0.1113 у сприклинга), то есть Минковский
            НЕОДНОРОДНЕЕ случайного. Угаданная цель, минимум которой —
            бесструктурное состояние.
`cells2.py` правило без цели (сцеплять с вероятностью p) не двинулось ни
            при каком p. Причина арифметическая: обмен СОСЕДЕЙ по рангу
            меняет одно отношение из N²/2, а соседних пар всего k(N−1);
            автомат не достаёт до 0.1 % матрицы.
`cells3.py` ход исправлен — клетка переставляет себя ЦЕЛИКОМ в одной
            шкале, O(N) отношений за ход, перебор всех кандидатов за O(N).
            Все три контроля легли: сгущать -> доля пар 0.972, высота 115,
            размерность 1.05; разрежать -> ровно 0.0000, высота 1;
            слепое -> стоит у ящика. И вилка ответила ДА:

                           доля пар      n0    разброс
              ящик           0.0702  0.5557     0.0821
              СПРИКЛИНГ      0.0947  0.4664     0.1050
              сгущено        0.9720  0.0157     0.8666

            Минковский есть СЛАБОЕ сгущение ящика: чуть плотнее, чуть
            неоднороднее, и всё по одну сторону.

ЧТО ЗДЕСЬ. Правило: с вероятностью g клетка переставляет себя сгущающим
ходом, иначе — слепым. Сгущение тянет к полному порядку, слепое тянет
назад к ящику, значит есть g с неподвижной точкой между ними, и это
настоящий аттрактор, а не память о старте.

ПОЧЕМУ ЭТО НЕ КРУГОВО. Ручка ОДНА, и настраивается она по ОДНОМУ числу —
доле пар. Остальные три мерила (обилие интервалов n0..n3, высота,
разброс счёта) при этом никто не трогает: если они совпадут сами, это
результат, а не подгонка. Обилие интервалов — стандартная проверка
многообразности в теории причинных множеств и оно же действие
Бенинказы–Доукера проекта.

КОНТРОЛИ:
  g = 0 — обязано стоять у ящика (слепое);
  старт с ДВУХ сторон (k = 5 — плотнее цели, k = 14 — много разреженнее):
    при одной и той же g обязаны прийти в одно место, иначе это не
    аттрактор;
  перешкалирование соблюдено тождественно (шкалы суть ранги);
  геометрии нет ни в состоянии, ни в правиле — только счёт.

  python3 cells4.py
  python3 cells4.py --gs 0.0 0.02 0.05 0.1 --sweeps 120
"""
import argparse
import time

import numpy as np

from cells3 import box, ag_of, facts, cmp_profile, target


def move(P, AG, inv, i, s, k, g, rs):
    """Клетка переставляет себя: сгущающим ходом с вероятностью g."""
    N = len(P)
    cur = P[:, s]
    base = (AG[i] - (cur[i] < cur).astype(np.int16)).astype(np.int64)
    base[i] = -1
    if rs.random() < g:
        r = int(np.argmax(cmp_profile(base, cur.astype(np.int64), k, N)))
    else:
        r = int(rs.integers(0, N))
    order = inv[s]
    order = np.delete(order, np.flatnonzero(order == i)[0])
    inv[s] = np.insert(order, min(r, N - 1), i)
    P[inv[s], s] = np.arange(N, dtype=np.int32)
    row = base + (P[i, s] < P[:, s]).astype(np.int64)
    row[i] = 0
    AG[i] = row.astype(np.int16)
    AG[:, i] = (k - row).astype(np.int16)
    AG[i, i] = 0


def run(N, k, g, sweeps, every, rs, label, tg):
    P = box(N, k, rs)
    AG = ag_of(P)
    inv = np.empty((k, N), np.int32)
    for s in range(k):
        inv[s] = np.argsort(P[:, s])
    print(f'  {label}')
    print(f"{'проход':>8}{'доля пар':>10}{'разм.':>7}{'выс.':>6}"
          f"{'n0':>8}{'n1':>8}{'n2':>8}{'n3':>8}{'разброс':>9}")
    fc = facts(AG == k)
    print(f"{0:>8}{fc['f']:>10.4f}{fc['dim']:>7.2f}{fc['hi']:>6}"
          f"{fc['n0']:>8.4f}{fc['n1']:>8.4f}{fc['n2']:>8.4f}"
          f"{fc['n3']:>8.4f}{fc['cv']:>9.4f}")
    last = fc
    for w in range(1, sweeps + 1):
        for _ in range(N):
            move(P, AG, inv, int(rs.integers(0, N)),
                 int(rs.integers(0, k)), k, g, rs)
        if w % every == 0 or w == sweeps:
            fc = facts(AG == k)
            last = fc
            print(f"{w:>8}{fc['f']:>10.4f}{fc['dim']:>7.2f}{fc['hi']:>6}"
                  f"{fc['n0']:>8.4f}{fc['n1']:>8.4f}{fc['n2']:>8.4f}"
                  f"{fc['n3']:>8.4f}{fc['cv']:>9.4f}", flush=True)
    return last


def main():
    ap = argparse.ArgumentParser(description='Аттрактор')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--k2', type=int, default=14)
    ap.add_argument('--gs', type=float, nargs='+',
                    default=[0.0, 0.01, 0.02, 0.05])
    ap.add_argument('--sweeps', type=int, default=80)
    ap.add_argument('--every', type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    tg = target(a.n, np.random.default_rng(31337))
    print(f'N = {a.n}. Ручка ОДНА: g — доля сгущающих ходов.')
    print('Настраивается по ОДНОМУ числу (доля пар). Остальные три')
    print('мерила никто не трогает — совпадут сами или нет.')
    print()
    print(f'МЕРИЛО — сприклинг в четырёхмерный ромб, N = {a.n}:')
    print(f"  доля пар {tg['f']:.4f}  разм. {tg['dim']:.2f}  "
          f"выс. {tg['hi']}  n0..n3 {tg['n0']:.4f} {tg['n1']:.4f} "
          f"{tg['n2']:.4f} {tg['n3']:.4f}  разброс {tg['cv']:.4f}")
    print()

    fin = {}
    for g in a.gs:
        print(f'=== g = {g:.3f},  k = {a.k} шкал ===')
        fin[g] = run(a.n, a.k, g, a.sweeps, a.every,
                     np.random.default_rng(2024), f'g={g:.3f}', tg)
        print()

    print('СВОДКА: расхождение с мерилом')
    print(f"{'g':>7}{'доля пар':>10}{'откл.':>9}{'n0':>9}{'откл.':>9}"
          f"{'выс.':>6}{'разброс':>9}")
    for g in a.gs:
        fc = fin[g]
        print(f"{g:>7.3f}{fc['f']:>10.4f}{fc['f']-tg['f']:>+9.4f}"
              f"{fc['n0']:>9.4f}{fc['n0']-tg['n0']:>+9.4f}"
              f"{fc['hi']:>6}{fc['cv']:>9.4f}")

    best = min(a.gs, key=lambda g: abs(fin[g]['f'] - tg['f']))
    print(f'\n  ближе всех по доле пар: g = {best:.3f}')
    print(f'=== аттрактор ли: та же g = {best:.3f} от МНОГИХ шкал '
          f'(k = {a.k2}, старт {2.0**(1-a.k2):.6f}) ===')
    o = run(a.n, a.k2, best, a.sweeps, a.every,
            np.random.default_rng(2025), f'k={a.k2}, g={best:.3f}', tg)
    print()
    print(f"  с двух сторон: доля пар {fin[best]['f']:.4f} против "
          f"{o['f']:.4f} (мерило {tg['f']:.4f})")
    print(f"               n0 {fin[best]['n0']:.4f} против {o['n0']:.4f} "
          f"(мерило {tg['n0']:.4f})")
    print('  сошлись друг с другом => аттрактор; совпали с мерилом =>')
    print('  четыре мерила на одну ручку')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
