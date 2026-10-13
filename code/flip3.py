#!/usr/bin/env python3
"""ЛАЗЕЙКА: настоящая или привязана к рабочей точке?

ЧТО НАШЁЛ `flip2.py`. При обзоре всех масштабов интервалов нашлось
направление с `σ_л/σ_м` = 64.3 на отложенной половине. Но таблица
корзин показала, ЧЕМ оно открылось: у слоистого порядка вся структура —
пик при `m ≈ λ = q²a₂N ≈ 200`, где у посева почти пусто (0.0216 против
0.244).

ДВА ПОДВОХА, и оба надо снять.
  1. Отношение может быть велико не потому, что слоистые шумят, а
     потому что `σ_м` там почти нуль. Это надо назвать прямо.
  2. ГЛАВНОЕ: пик стоит при `λ ∝ N`, а корзины в `flip2.py` заданы в
     АБСОЛЮТНЫХ `m` с границей `N/2`. Значит направление привязано к
     `N = 1600` — ровно ошибка №26 («совпадение в одной точке — не
     калибровка»). Вдобавок кривые образцы имеют другое `N`, и те же
     абсолютные корзины означают у них другой масштаб: условие на
     отклик было наложено неряшливо.

ЗДЕСЬ. Корзины в ОТНОСИТЕЛЬНЫХ `u = m/N` — тогда функционал свободен от
масштаба, у посева распределение `u` от `N` не зависит, и у слоистого
пик стоит при `u = q²a₂ = 0.125` при любом `N`. И главная проверка:
направление строится при `N = 1600`, а отношение `σ_л/σ_м` считается
ПРИ ДРУГИХ `N`. Если держится — лазейка настоящая.

  python3 flip3.py
"""
import math
import time

import numpy as np

from sprinkle import sprinkle_d
from uphill import gen3
from curve import curvature, sprinkle as csprinkle
from ricci import causal

PARTS = (0.25, 0.5, 0.25)
UED = np.array([0.0, 1e-9, 0.002, 0.005, 0.012, 0.03, 0.06, 0.10,
                0.16, 0.25, 0.40, 0.65, 1.0, 1e9])


def vecu(A):
    """(N, счёты по корзинам относительного размера интервала u = m/N)."""
    n = A.shape[0]
    F = A.astype(np.float32)
    B = np.rint(F @ F).astype(np.int64)
    u = B[A] / float(n)
    out = np.zeros(len(UED))
    out[0] = n
    idx = np.searchsorted(UED[1:], u, side='right')
    bc = np.bincount(idx, minlength=len(UED) - 1)
    out[1:] = bc[:len(UED) - 1]
    return out / n          # на событие, чтобы величины были сравнимы


def collect(gen, n, seeds, root, **kw):
    out = []
    for s in range(seeds):
        A = gen(n, rs=np.random.default_rng(root + 97 * s), **kw)
        if isinstance(A, tuple):
            A = A[0]
        out.append(vecu(A))
        del A
    return np.array(out, float)


def curved_u(par, seeds, root, L=8.5, T=4.0, E0=2.0):
    rows, Rs = [], []
    for s in range(seeds):
        X = csprinkle('quad', par, L, T, E0,
                      np.random.default_rng(root + 97 * s))
        A = causal(X)
        rows.append(vecu(A))
        Rs.append(float(np.mean([curvature(e, 'quad', par, E0)[0]
                                 for e in X[:, 0]])))
        del A, X
    return np.array(rows, float), np.array(Rs, float)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    SD = 200
    print('Сбор при N = 1600 (обучение) и при 800, 3200 (перенос)',
          flush=True)
    MAN = collect(sprinkle_d, 1600, SD, 42000, d=4)
    LAY = collect(gen3, 1600, SD, 43000, parts=PARTS, q=0.5)
    print(f'  N=1600: {len(MAN)} и {len(LAY)} ({time.time()-t0:.0f} с)',
          flush=True)
    M8 = collect(sprinkle_d, 800, 80, 44000, d=4)
    L8 = collect(gen3, 800, 80, 45000, parts=PARTS, q=0.5)
    M32 = collect(sprinkle_d, 3200, 40, 46000, d=4)
    L32 = collect(gen3, 3200, 40, 47000, parts=PARTS, q=0.5)
    print(f'  перенос собран ({time.time()-t0:.0f} с)', flush=True)
    Cp, Rp = curved_u(+0.10, 30, 48000)
    Cm, Rm = curved_u(-0.10, 30, 49000)
    dmu = Cp.mean(axis=0) - Cm.mean(axis=0)
    dR = float(Rp.mean() - Rm.mean())
    print(f'  кривые ({time.time()-t0:.0f} с)')
    print()
    print('=' * 96)
    print('Средние по корзинам u = m/N (на событие). Пик слоистых при 0.125')
    print(f"{'u':>16}{'многообр. 1600':>16}{'слоист. 1600':>14}"
          f"{'многообр. 3200':>16}{'слоист. 3200':>14}")
    for i in range(len(UED) - 1):
        a, b = MAN[:, 1 + i].mean(), LAY[:, 1 + i].mean()
        c, d = M32[:, 1 + i].mean(), L32[:, 1 + i].mean()
        if max(a, b, c, d) > 1e-3:
            print(f'{f"{UED[i]:.3g}..{UED[i+1]:.3g}":>16}'
                  f'{a:>16.3f}{b:>14.3f}{c:>16.3f}{d:>14.3f}')
    print()
    h = SD // 2
    trM, teM, trL, teL = MAN[:h], MAN[h:], LAY[:h], LAY[h:]
    A = np.column_stack([trM.mean(axis=0), M8.mean(axis=0), dmu])
    U, S_, Vt = np.linalg.svd(A.T)
    Z = Vt[len(S_):].T

    def cov(X):
        return np.cov((X - X.mean(axis=0)).T)

    SM = Z.T @ cov(trM) @ Z
    SL = Z.T @ cov(trL) @ Z
    SMr = SM + 1e-9 * np.trace(SM) / SM.shape[0] * np.eye(SM.shape[0])
    w_, v_ = np.linalg.eig(np.linalg.solve(SMr, SL))
    o = np.argsort(-np.real(w_))
    print('=' * 96)
    print(f'Нулевое подпространство {Z.shape[1]} из {Z.shape[0]};  '
          f'λ: ' + '  '.join(f'{np.real(w_[i]):.3g}' for i in o[:4]))
    print()
    print('ГЛАВНАЯ ПРОВЕРКА: направление строится при N=1600, '
          'проверяется при других N')
    print(f"{'напр.':>10}{'1600 обуч.':>13}{'1600 ОТЛОЖ.':>14}"
          f"{'N=800':>11}{'N=3200':>11}{'держится?':>12}")
    verdict = []
    for r in range(min(3, len(o))):
        v = Z @ np.real(v_[:, o[r]])

        def rat(L_, M_):
            sm = np.std(M_ @ v, ddof=1)
            return np.std(L_ @ v, ddof=1) / max(sm, 1e-30)

        a = rat(trL, trM); b = rat(teL, teM)
        c = rat(L8, M8); d = rat(L32, M32)
        ok = (b > 1) and (c > 1) and (d > 1)
        verdict.append(ok)
        print(f'{"#" + str(r+1):>10}{a:>13.3f}{b:>14.3f}{c:>11.3f}'
              f'{d:>11.3f}{("ДА" if ok else "нет"):>12}')
    print()
    print('=' * 96)
    if any(verdict):
        print('''ЛАЗЕЙКА НАСТОЯЩАЯ. Направление, построенное при одном `N`,
   сохраняет `σ_л/σ_м > 1` при других `N`, причём корзины относительные,
   то есть функционал свободен от масштаба. Фазовый отбор возможен.
   Оговорка, которую надо держать: такой функционал смотрит на ОТНОСИТЕЛЬНЫЙ
   размер интервалов, то есть НЕ локален в смысле БД, и действием
   плотности его называть нельзя без отдельного разбора.''')
    else:
        print('''ЛАЗЕЙКА БЫЛА АРТЕФАКТОМ РАБОЧЕЙ ТОЧКИ. Направление не
   переносится на другие `N`: отношение 64 в `flip2.py` держалось на
   абсолютных корзинах, привязанных к `N = 1600`, где пик слоистых
   случайно попадал в корзину, почти пустую у посева. Ошибка №26.''')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
