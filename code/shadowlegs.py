#!/usr/bin/env python3
"""«НОГИ В ТЕНЬ», шаги 1–2: можно ли спарить лишние ноги событий.

Событие делаем унитарным переходом размера D = max(вх, исх). Если вх > исх, событию не хватает выходов:
у него (вх − исх) скрытых выходов. Если исх > вх — (исх − вх) скрытых входов. Скрытая нога ведёт из события X
(лишний выход) в более позднее событие Y (лишний вход): X ≺ Y — причинность сохраняется.
Это та же картина, что «испускание и поглощение»: лишний выход — испущенный квант, лишний вход — поглощённый.
Область — центр ромба |t| + |x| < 14 (звенья наружу — внешние ноги). Спаривание — минимальная стоимость
(scipy linear_sum_assignment) по единицам лишних ног; стоимость — расстояние теней (вариант «через тень»)
или собственное время (вариант «через мир, ближайший»), запрет — если X не раньше Y."""
import sys, math, numpy as np
from scipy.optimize import linear_sum_assignment
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a

def legs(Lw=24.0, inner=14.0, rho=4.0):
    w = CH.sym_world(rho, Lw); R = CH.build_order(w, 'proj'); L = CH.links_of(R); X = w.X
    reg = np.abs(X).sum(1) < inner
    ins, outs = L.sum(0), L.sum(1)
    exo = np.maximum(ins - outs, 0) * reg      # нужно скрытых ВЫХОДОВ
    exi = np.maximum(outs - ins, 0) * reg      # нужно скрытых ВХОДОВ
    return w, R, L, X, reg, exo, exi

def match(w, R, X, exo, exi, cost_kind):
    U = np.repeat(np.arange(w.n), exo); V = np.repeat(np.arange(w.n), exi)
    caus = R[np.ix_(U, V)]                                      # X ≺ Y
    if cost_kind == 'тень':
        C = np.sqrt(((w.W[U][:, None, :] - w.W[V][None, :, :]) ** 2).sum(-1))
    else:
        d = X[V][None, :, :] - X[U][:, None, :]; C = np.sqrt(np.maximum(d[..., 0] ** 2 - d[..., 1] ** 2, 0))
    BIG = 1e6; C = np.where(caus, C, BIG)
    r, c = linear_sum_assignment(C)
    ok = C[r, c] < BIG
    return U[r[ok]], V[c[ok]], len(U), len(V)

if __name__ == '__main__':
    w, R, L, X, reg, exo, exi = legs()
    print(f'область |t|+|x| < 14: событий {reg.sum()}; лишних выходов (скрытых ног наружу) {exo.sum()}, лишних входов {exi.sum()}')
    # равенство по времени: накопленный баланс — нельзя спарить, если раньше «входов» требуется больше, чем «выходов» было
    o = np.argsort(X[:, 0]); bal = np.cumsum(exo[o] - exi[o])
    print(f'накопленный баланс (выходы − входы) по времени: мин {bal.min()}, макс {bal.max()} (отрицательный — скрытому входу неоткуда взяться)')
    for kind in ('тень', 'мир'):
        a, b, nu, nv = match(w, R, X, exo, exi, kind)
        d = X[b] - X[a]; tau = np.sqrt(np.maximum(d[:, 0] ** 2 - d[:, 1] ** 2, 0)); dx = np.abs(d[:, 1]); dt = d[:, 0]
        sh = np.sqrt(((w.W[b] - w.W[a]) ** 2).sum(1))
        # «близость в мире»: какое место по собственному времени занимает партнёр среди всех допустимых
        print(f'\nспаривание по {kind}: спарено {len(a)} из {nu} (лишних входов {nv})')
        print(f'  расстояние теней: медиана {np.median(sh):.3f}, 90 % {np.percentile(sh, 90):.3f} (радиус окна {w.r:.3f})')
        print(f'  в мире: собственное время медиана {np.median(tau):.2f}, 90 % {np.percentile(tau, 90):.2f}; Δt медиана {np.median(dt):.2f}; '
              f'|Δx| медиана {np.median(dx):.2f}, 90 % {np.percentile(dx, 90):.2f}, макс {dx.max():.2f}')
        # для сравнения: средняя собственная длина обычного звена
        Ii, Jj = np.nonzero(L); dl = X[Jj] - X[Ii]; s = np.sqrt(dl[:, 0] ** 2 - dl[:, 1] ** 2)
        print(f'  (обычное звено: собственная длина медиана {np.median(s):.2f})')
        np.save(f'/home/claude/data/shadowlegs_{kind}.npy', np.stack([a, b]))
