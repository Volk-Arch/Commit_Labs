#!/usr/bin/env python3
"""СКОЛЬКО ЛУЧЕЙ НУЖНО КОНУСУ: гранёный свет против настоящего.

Повод - замечание про Z_{n+1} = Z_n^d + C: у этого множества симметрия
вращения порядка d-1, отсюда d-1 лучей. А в физике d-1 есть число
ПРОСТРАНСТВЕННЫХ направлений, и весь сегодняшний день уперся ровно в это
число: световой конус РАЗДЕЛЯЕТСЯ только при d-1 = 1 (отсюда точная
решённость 1+1), а S_k строится именно потому, что при d-1 = 3 единичных
векторов сколько угодно (отсюда запрет на модель из k шкал).

Звезда даёт точную формулировку того, чего не хватает орте. Световой
конус есть пересечение КОНТИНУУМА полупространств - по одному на каждое
направление сферы:
    dt > |dx|   <=>   dt > n·dx  для ВСЕХ единичных n.
Орта берёт k осей. Звезда взяла бы d-1, расставленных симметрично.
Вопрос: СКОЛЬКО ЛУЧЕЙ НУЖНО?

Правило: i < j, если dt > 0 и dt > max_m (n_m · dx) по k направлениям
n_m, расставленным по сфере (Фибоначчи). При k -> бесконечность это
точный световой конус. Порядковая размерность при этом <= k+1, то есть
при каждом конечном k теорема `orthant-doc.md` ещё действует - и мы
увидим, как быстро она перестаёт мешать.

КОНТРОЛИ:
  (a) при большом k профиль обязан войти в область настоящего засева -
      иначе сломан код, а не физика;
  (b) доля связанных печатается: гранёный конус ОБЪЕМНЕЕ светового
      (описан вокруг), значит при малом k связей больше, и это надо
      видеть, а не путать с геометрией.
"""
import numpy as np

from sectors import sprinkle_d
from orthant import prof, region, rf
from ordim import best_sk

N = 500


def fib_sphere(k):
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi),
                     np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def pts(n, rs):
    out = []
    while sum(len(o) for o in out) < n:
        k = 3 * n + 500
        t = rs.random(k)
        v = rs.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = rs.random(k) ** (1 / 3) * 0.5
        ok = rad < np.minimum(t, 1 - t)
        out.append(np.hstack([t[ok, None], v[ok] * rad[ok, None]]))
    return np.vstack(out)[:n]


def faceted(P, U):
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx = P[:, 1:][None, :, :] - P[:, 1:][:, None, :]
    proj = np.einsum('ijc,mc->ijm', dx, U).max(axis=2)
    return (dt > 0) & (dt > proj)


def main():
    print(__doc__)
    rs = np.random.default_rng(161803398)
    chi4, thr4 = region(lambda: sprinkle_d(N, 4, rs), 50, rs)
    print(f'порог области 4d при N={N}: {thr4:.1f}')
    cs = [float(chi4(prof(sprinkle_d(N, 4, rs)))) for _ in range(5)]
    print(f'КОНТРОЛЬ: пять настоящих засевов -> '
          + ' '.join(f'{v:.1f}' for v in cs))

    print(f"\n{'лучей k':>9}{'доля связ.':>12}{'хи2 к 4d':>11}{'S_k':>5}")
    for k in (3, 4, 6, 10, 20, 40, 100, 400):
        U = fib_sphere(k)
        vals, rr, sk = [], [], []
        for _ in range(3):
            P = pts(N, rs)
            R = faceted(P, U)
            vals.append(float(chi4(prof(R))))
            rr.append(rf(R))
            sk.append(best_sk(R, rs))
        print(f'{k:>9}{np.mean(rr):>12.4f}{np.mean(vals):>11.1f}'
              f'{max(sk):>5}', flush=True)
    print('\nЧТЕНИЕ: наименьшее k, при котором хи2 входит в разброс')
    print('контроля, и есть «сколько лучей нужно».')


if __name__ == '__main__':
    main()
