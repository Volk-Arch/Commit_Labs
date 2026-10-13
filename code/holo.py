#!/usr/bin/env python3
"""Голограмма или тень: восстанавливается ли объём по границе.

ПОВОД. Найдены две голографические ПОДПИСИ: площадной закон энтропии
горизонта (horizon-cone-doc.md, показатель 2.286 +- 0.060, 12 сигма от
объёмного) и граничность остатка взвешенной суммы (weight-doc.md,
показатель N^(-1/4) = площадь/объём, внутренность 0.0029 +- 0.0150).

Но это СЧЁТ: «на поверхности ровно столько информации». Голография
утверждает ВОССТАНОВИМОСТЬ: «поверхность определяет объём». Это разные
утверждения, и второе не проверялось.

ПРАВИЛО ВОССТАНОВЛЕНИЯ. Для объёмного события A известен только его след
на экране: какие экранные события в его прошлом и какие в будущем.
Предсказание:

    A < B   <=>   pastS(A) <= pastS(B)  И  futS(B) <= futS(A)

Если A < B, то past(A) вложено в past(B) всегда, значит ЛОЖНЫХ
ОТРИЦАНИЙ обязано быть ровно нуль. Это встроенная проверка правильности
кода. Вопрос в ложных положительных.

КОНТРОЛЬ, который и решает дело: экран НА ГРАНИЦЕ против СЛУЧАЙНОГО
экрана того же размера. Если граница не лучше случайной выборки —
голографии нет, есть «больше данных помогает».

  python3 holo.py
  python3 holo.py --ns 2000
"""
import argparse
import time

import numpy as np

T = 2.0


def diamond(N, rs):
    out, got = [], 0
    while got < N:
        m = int((N - got) * 9) + 64
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        ok = (Y[:, 0] >= r) & (T - Y[:, 0] >= r)
        if ok.any():
            out.append(Y[ok])
            got += int(ok.sum())
    return np.vstack(out)[:N]


def cone_rel(X, chunk=400):
    """Конусный порядок, по частям — иначе N^2 x 3 временные массивы."""
    N = len(X)
    R = np.empty((N, N), bool)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        dt = X[a:b, None, 0] - X[None, :, 0]
        dx = np.linalg.norm(X[a:b, None, 1:] - X[None, :, 1:], axis=-1)
        R[a:b] = dt > dx
    np.fill_diagonal(R, False)
    return R


def depth(X):
    r = np.linalg.norm(X[:, 1:], axis=1)
    return np.minimum(X[:, 0] - r, T - X[:, 0] - r)


def subset_le(P):
    """M[i,j] = True, если множество i вложено в множество j."""
    A = P.astype(np.float32)
    return (A @ (1.0 - A).T) == 0


def score(R, bulk, screen):
    """Точность восстановления порядка на объёмных парах."""
    pastS = R[np.ix_(bulk, screen)]          # pastS[a,s]: s раньше a
    futS = R[np.ix_(screen, bulk)].T         # futS[a,s]: s позже a
    ip = subset_le(pastS)                    # ip[a,b]: pastS(a) <= pastS(b)
    if_ = subset_le(futS)                    # if_[b,a]: futS(b) <= futS(a)
    same = ip & ip.T & if_ & if_.T           # неразличимые следы
    pred = ip & if_.T & ~same
    Rb = R[np.ix_(bulk, bulk)]
    truth = Rb.T                             # truth[a,b]: a раньше b
    np.fill_diagonal(pred, False)
    np.fill_diagonal(truth, False)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    k = len(bulk)
    return dict(tp=tp, fp=fp, fn=fn,
                prec=tp / max(tp + fp, 1),
                rel=float(truth.sum() / (k * (k - 1))),
                same=float((same.sum() - k) / max(k * (k - 1), 1)))


def main():
    ap = argparse.ArgumentParser(description='Голограмма или тень')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[500, 1000, 2000, 4000, 8000])
    ap.add_argument('--deep', type=float, default=0.35)
    ap.add_argument('--k', type=int, default=250)
    ap.add_argument('--frac', type=float, default=0.25)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    print(f'причинный ромб собственного времени {T}, конусный порядок')
    print(f'объёмные: {a.k} событий глубже {a.deep}; '
          f'экран: {a.frac:.0%} мира')
    print('ложных отрицаний обязано быть 0 (вложенность прошлых)')
    print()
    print(f"{'N':>7}{'экран':>7}{'связано':>9}"
          f"{'граница: точность':>19}{'лож.+':>8}{'лож.-':>7}{'неразл.':>9}"
          f"{'случайный: точн.':>18}{'лож.+':>8}")

    for N in a.ns:
        rows = []
        for t in range(a.seeds):
            rs = np.random.default_rng(2200 + 17 * t)
            X = diamond(N, rs)
            R = cone_rel(X)
            d = depth(X)
            deep = np.flatnonzero(d >= a.deep)
            if len(deep) < a.k:
                rows.append(None)
                continue
            bulk = rs.choice(deep, a.k, replace=False)
            ns = int(a.frac * N)
            rest = np.setdiff1d(np.arange(N), bulk)
            # экран на границе: самые мелкие по глубине
            edge = rest[np.argsort(d[rest])[:ns]]
            # случайный экран того же размера
            rnd = rs.choice(rest, ns, replace=False)
            rows.append((score(R, bulk, edge), score(R, bulk, rnd), ns))
            del R, X
        rows = [r for r in rows if r is not None]
        if not rows:
            print(f'{N:>7}   мало глубоких событий')
            continue
        g = lambda key, i: float(np.mean([r[i][key] for r in rows]))
        ns = int(np.mean([r[2] for r in rows]))
        print(f'{N:>7}{ns:>7}{g("rel",0):>9.3f}'
              f'{g("prec",0):>19.4f}{g("fp",0):>8.0f}{g("fn",0):>7.0f}'
              f'{g("same",0):>9.4f}'
              f'{g("prec",1):>18.4f}{g("fp",1):>8.0f}', flush=True)

    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
