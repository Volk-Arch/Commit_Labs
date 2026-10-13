#!/usr/bin/env python3
"""Проверка предсказания simple-doc для 2+1: гранёный конус из k граней.

Предсказание (записано ДО замера): ошибка гранёного конуса падает как
1/k² на окружности (2+1) и как 1/k на сфере (3+1). Следствие: при одном
и том же допуске в 2+1 граней нужно примерно корень из числа граней 3+1.

Мера без подгонки: на ОДНИХ И ТЕХ ЖЕ точках сравниваем точный конус и
гранёный (он описан вокруг точного, значит только добавляет пары).
  лишних = E/R, E — пары гранёного сверх точного, R — пары точного.
"""
import math, sys, time
import numpy as np

def sprinkle(N, d, rs):
    out = []
    while sum(len(o) for o in out) < N:
        c = rs.uniform(-1, 1, (4 * N, d))
        ok = np.abs(c[:, 0]) + np.linalg.norm(c[:, 1:], axis=1) <= 1
        out.append(c[ok])
    return np.concatenate(out)[:N]

def dirs(k, d, rs):
    if d == 3:                       # правильный k-угольник, случайный поворот
        a = 2 * np.pi * (np.arange(k) + rs.uniform()) / k
        return np.stack([np.cos(a), np.sin(a)], 1)
    i = np.arange(k) + 0.5           # решётка Фибоначчи + случайный поворот
    z = 1 - 2 * i / k; r = np.sqrt(1 - z * z)
    ph = np.pi * (1 + 5 ** 0.5) * i
    v = np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)
    Q, _ = np.linalg.qr(rs.normal(size=(3, 3)))
    return v @ Q

def run(N, d, ks, seeds):
    res = {k: [] for k in ks}; Rs = []
    for s in range(seeds):
        rs = np.random.default_rng(1000 * d + 7 * s + N)
        X = sprinkle(N, d, rs)
        dt = X[:, None, 0] - X[None, :, 0]
        dx = X[:, None, 1:] - X[None, :, 1:]
        exact = dt > np.linalg.norm(dx, axis=-1)
        R = exact.sum(); Rs.append(R)
        for k in ks:
            U = dirs(k, d, rs)
            M = np.full(dt.shape, -np.inf)
            for u in U:
                np.maximum(M, dx @ u, out=M)
            poly = (dt > 0) & (dt > M)
            assert not (exact & ~poly).any()      # описан: точный ⊂ гранёного
            res[k].append((poly.sum() - R) / R)
    return {k: (np.mean(v), np.std(v) / math.sqrt(len(v))) for k, v in res.items()}, np.mean(Rs)

def slope(ks, ys):
    ks, ys = np.array(ks, float), np.array(ys)
    m = ys > 0
    return np.polyfit(np.log(ks[m]), np.log(ys[m]), 1)

if __name__ == '__main__':
    t0 = time.time()
    KS = {3: [4, 6, 8, 12, 16, 24, 32, 48, 64],
          4: [6, 10, 20, 40, 80, 160, 320, 640]}
    fits = {}
    for d in (3, 4):
        for N in (500, 1000, 2000):
            r, R = run(N, d, KS[d], 4)
            print(f'd={d} N={N} пар точного конуса R={R:.0f}')
            for k in KS[d]:
                print(f'   k={k:>4}  лишних {r[k][0]:.5f} ± {r[k][1]:.5f}'
                      f'   per element {r[k][0]*R/N:8.3f}')
            p, c = slope(KS[d][2:], [r[k][0] for k in KS[d][2:]])
            fits[(d, N)] = (p, c, R)
            print(f'   наклон (без двух первых k): {p:+.3f}   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
    print('\nСКОЛЬКО ГРАНЕЙ НУЖНО (из подгонки E/R = e^c k^p)')
    print('  критерий A: лишних пар меньше 1 на событие  (E/N < 1)')
    print('  критерий B: лишних меньше шума числа пар    (E < √R)')
    for N in (500, 1000, 2000):
        row = []
        for d in (3, 4):
            p, c, R = fits[(d, N)]
            kA = math.exp((math.log(N / R) - c) / p)
            kB = math.exp((math.log(1 / math.sqrt(R)) - c) / p)
            row.append((kA, kB))
        print(f'  N={N}:  2+1  A {row[0][0]:6.1f}  B {row[0][1]:6.1f}'
              f'   |  3+1  A {row[1][0]:7.1f}  B {row[1][1]:7.1f}'
              f'   |  √k(3+1): A {math.sqrt(row[1][0]):5.1f} B {math.sqrt(row[1][1]):5.1f}')
