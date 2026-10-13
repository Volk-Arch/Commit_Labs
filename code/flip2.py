#!/usr/bin/env python3
"""ЛАЗЕЙКА БОЛЬШИХ ИНТЕРВАЛОВ: видит ли её функционал, смотрящий дальше?

ЧТО ПОКАЗАЛ `flip.py`. Среди линейных комбинаций счётов МАЛЫХ интервалов
(`m ≤ 15`, то есть ровно поле зрения действия БД) нет направления, где
слоистые шумят сильнее многообразных: `λ_max` = 0.506, на отложенной
половине 0.653. Причина видна по спектру: 0.506, затем 5e−17, 3e−17 —
ковариация слоистых имеет РАНГ ОДИН. У плотного слоистого порядка
колеблется только `N₀`, остальные малые счёты почти детерминированы.

НО ЭТО И ЕСТЬ ОГРАНИЧЕНИЕ ВЫВОДА. У плотного слоистого порядка вся
структура сидит при `m ≈ λ = q²a₂N ≈ 200` — вне поля зрения `m ≤ 15`.
Там он не жёсток: число промежуточных само пуассоново с `λ`, то есть
разброс `√λ ≈ 14`. Значит функционал, смотрящий ДАЛЬШЕ, мог бы увидеть
его флуктуации и перевернуть знак.

ЗДЕСЬ. Те же счёты, но в ЛОГАРИФМИЧЕСКИХ корзинах по `m` до `N/2`:
покрыты все масштабы интервалов, а размерность остаётся малой, чтобы
ковариация была определима из 200 реализаций.

Условия на среднее и перекрёстная проверка — те же, что в `flip.py`.

  python3 flip2.py
"""
import math
import time

import numpy as np

from sprinkle import sprinkle_d
from uphill import gen3
from design import curved_sample, bd_w
from curve import curvature, sprinkle as csprinkle
from ricci import causal

PARTS = (0.25, 0.5, 0.25)
NB = 14


def edges(n):
    """Логарифмические границы корзин по m: 0,1,2,3 поимённо, дальше лог."""
    e = [0, 1, 2, 3, 4, 6, 9]
    hi = max(16, n // 2)
    k = NB - len(e)
    e += list(np.unique(np.round(np.geomspace(14, hi, k)).astype(int)))
    return np.array(sorted(set(e)) + [10 ** 9])


def vecb(A, ed):
    n = A.shape[0]
    F = A.astype(np.float32)
    B = np.rint(F @ F).astype(np.int64)
    rel = B[A]
    out = np.zeros(len(ed))
    out[0] = n
    idx = np.searchsorted(ed[1:], rel, side='right')
    bc = np.bincount(idx, minlength=len(ed) - 1)
    out[1:] = bc[:len(ed) - 1]
    return out


def collect(gen, n, seeds, root, ed, **kw):
    out = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 97 * s), **kw)
        if isinstance(A, tuple):
            A = A[0]
        out.append(vecb(A, ed))
        del A
    return np.array(out, float)


def curved_b(par, seeds, root, ed, L=8.5, T=4.0, E0=2.0):
    rows, Rs = [], []
    for s in range(seeds):
        X = csprinkle('quad', par, L, T, E0,
                      np.random.default_rng(root + 97 * s))
        A = causal(X)
        rows.append(vecb(A, ed))
        Rs.append(np.mean([curvature(e, 'quad', par, E0)[0]
                           for e in X[:, 0]]) * len(X))
        del A, X
    return np.array(rows, float), np.array(Rs, float)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N1, N2, SD = 800, 1600, 200
    ed = edges(N2)
    print(f'Корзины по m: {list(ed[:-1])}  (всего {len(ed)-1} + N)')
    print('Сбор выборок', flush=True)
    M1 = collect(sprinkle_d, N1, 60, 31000, ed, d=4)
    MAN = collect(sprinkle_d, N2, SD, 32000, ed, d=4)
    print(f'  многообразные {len(MAN)} ({time.time()-t0:.0f} с)', flush=True)
    LAY = collect(gen3, N2, SD, 33000, ed, parts=PARTS, q=0.5)
    print(f'  слоистые {len(LAY)} ({time.time()-t0:.0f} с)', flush=True)
    Cp, Rp = curved_b(+0.10, 30, 34000, ed)
    Cm, Rm = curved_b(-0.10, 30, 35000, ed)
    dmu = Cp.mean(axis=0) - Cm.mean(axis=0)
    dR = float(Rp.mean() - Rm.mean())
    print(f'  кривые ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 92)
    print('Где у кого структура: средние по корзинам (доля от всех пар)')
    print(f"{'корзина m':>14}{'многообразные':>16}{'слоистые':>14}")
    tm, tl = MAN[:, 1:].sum(axis=1), LAY[:, 1:].sum(axis=1)
    for i in range(len(ed) - 1):
        lo = ed[i] if i > 0 else 0
        hi = ed[i + 1] - 1 if i + 1 < len(ed) - 1 else '…'
        a = MAN[:, 1 + i].mean() / tm.mean()
        b = LAY[:, 1 + i].mean() / tl.mean()
        if a > 1e-4 or b > 1e-4:
            print(f'{f"{lo}..{hi}":>14}{a:>16.4f}{b:>14.4f}')
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    h = SD // 2
    trM, teM = MAN[:h], MAN[h:]
    trL, teL = LAY[:h], LAY[h:]
    A = np.column_stack([M1.mean(axis=0), trM.mean(axis=0), dmu])
    U, S_, Vt = np.linalg.svd(A.T)
    Z = Vt[len(S_):].T

    def cov(X):
        return np.cov((X - X.mean(axis=0)).T)

    SM = Z.T @ cov(trM) @ Z
    SL = Z.T @ cov(trL) @ Z
    SMr = SM + 1e-9 * np.trace(SM) / SM.shape[0] * np.eye(SM.shape[0])
    w_, v_ = np.linalg.eig(np.linalg.solve(SMr, SL))
    o = np.argsort(-np.real(w_))
    print('=' * 92)
    print(f'Нулевое подпространство: {Z.shape[1]} из {Z.shape[0]}')
    print('λ (обуч.): ' + '  '.join(f'{np.real(w_[i]):.3g}' for i in o[:6]))
    print()
    print(f"{'направление':>26}{'σ_л/σ_м обуч.':>16}{'σ_л/σ_м ОТЛОЖ.':>17}"
          f"{'вывод':>10}")
    best = 0.0
    for r in range(min(4, len(o))):
        v = Z @ np.real(v_[:, o[r]])
        a = np.std(trL @ v, ddof=1) / max(np.std(trM @ v, ddof=1), 1e-30)
        b = np.std(teL @ v, ddof=1) / max(np.std(teM @ v, ddof=1), 1e-30)
        best = max(best, b)
        print(f'{"собств. #" + str(r+1):>26}{a:>16.4f}{b:>17.4f}'
              f'{("ДА" if b > 1 else "нет"):>10}')
    print()
    print('=' * 92)
    if best > 1.0:
        print(f'''ЛАЗЕЙКА ОТКРЫТА. На отложенной половине σ_л/σ_м = {best:.3f} > 1.
   Значит функционал, видящий БОЛЬШИЕ интервалы, даёт слоистым шуметь
   сильнее, и фазовый отбор возможен — но такой функционал уже не
   локален в смысле БД, и это надо обсудить отдельно.''')
    else:
        print(f'''ЛАЗЕЙКА ЗАКРЫТА ТОЖЕ. Лучшее отложенное σ_л/σ_м = {best:.3f} ≤ 1
   уже при обзоре всех масштабов интервалов до N/2. Значит дело не в
   узости поля зрения действия БД: многообразные порядки шумят сильнее
   слоистых ВО ВСЕХ счётах интервалов, на всех масштабах.''')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
