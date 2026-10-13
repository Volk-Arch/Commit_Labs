#!/usr/bin/env python3
"""ОГРУБЛЕНИЕ: кто неподвижная точка, а кто течёт.

Синтез 11 октября: отличительное свойство пространства-времени — не
величина, а взаимная согласованность вееров связей, то есть
существование общего реализатора. Фильтр (предикат на конфигурациях)
враг имитирует всегда — проверено шесть раз. Параметризация задаёт
геометрию руками. Третий путь — ПОТОК с неподвижными точками: он ничего
не давит, он уносит лишнее.

Огрубление = прореживание: оставляем элементы с вероятностью p и берём
ИНДУЦИРОВАННЫЙ порядок. Он сохраняет все отношения между выжившими,
даже если промежуточные вымерли.

ВЫВОД НА БУМАГЕ, с чем сверяться:
  * засев: прореживание пуассоновского засева даёт пуассоновский засев
    той же области меньшей плотности, значит форма сохраняется, а доля
    связанных r зависит только от d и ИНВАРИАНТНА;
  * слоистый с k слоями размера a и плотностью q:
    r = [2q + 1 - (1-q^2)^a] / 4.5   при k = 3,
    и при убывании a третий член уходит, так что r ТЕЧЁТ от
    (2q+1)/4.5 = 0.311 до 2q/4.5 = 0.089 при q = 0.2.

Плюс второе: доля прямых связей L/R у засева обязана течь как
N^{-2/d}, то есть поток сам даёт ИЗМЕРИТЕЛЬ РАЗМЕРНОСТИ:
d ln(L/R) / d ln N = -2/d.

ПРЕДРЕГИСТРАЦИЯ: r у засевов плоская по p; у слоистых убывает к 0.089
(q=0.2) и 0.044 (q=0.1); показатель потока L/R даёт -1.00, -0.67, -0.50
при d = 2, 3, 4.

КОНТРОЛЬ: всё мерится одним кодом в одном прогоне; формула для
слоистого сверяется с замером при p = 1.
"""
import math

import numpy as np

from sectors import sprinkle_d
from lkdeg import layered, linkmat


def obs(Rc):
    n = Rc.shape[0]
    if n < 4:
        return None
    R = float(Rc.sum())
    L = float(linkmat(Rc).sum())
    pairs = n * (n - 1) / 2
    return dict(n=n, r=R / pairs, LR=(L / R if R > 0 else float('nan')))


def thin(Rc, p, rs):
    n = Rc.shape[0]
    k = max(4, int(round(p * n)))
    idx = rs.choice(n, size=k, replace=False)
    idx.sort()
    return Rc[np.ix_(idx, idx)]


def flow(Rc, rs, ps=(1.0, 0.5, 0.25, 0.125, 0.0625), reps=3):
    out = []
    for p in ps:
        vs = [obs(thin(Rc, p, rs)) for _ in range(reps)]
        vs = [v for v in vs if v]
        if not vs:
            continue
        out.append((float(np.mean([v['n'] for v in vs])),
                    float(np.mean([v['r'] for v in vs])),
                    float(np.mean([v['LR'] for v in vs]))))
    return out


def main():
    print(__doc__)
    rs = np.random.default_rng(161803)
    N = 2400
    print(f'исходное N = {N}')
    print(f"\n{'ансамбль':>20}" + ''.join(
        f'{f"n={int(N*p)}":>9}' for p in (1.0, .5, .25, .125, .0625))
        + f"{'поток r':>10}{'пок. L/R':>10}{'=> d':>7}")

    def line(name, Rc, pred=None):
        fl = flow(Rc, rs)
        ns = np.array([f[0] for f in fl])
        rr = np.array([f[1] for f in fl])
        lr = np.array([f[2] for f in fl])
        drift = (rr[-1] - rr[0]) / max(rr[0], 1e-9)
        ok = np.isfinite(lr) & (lr > 0)
        pw = (np.polyfit(np.log(ns[ok]), np.log(lr[ok]), 1)[0]
              if ok.sum() >= 3 else float('nan'))
        dest = (-2.0 / pw) if pw < -1e-6 else float('nan')
        print(f'{name:>20}' + ''.join(f'{v:>9.4f}' for v in rr)
              + f'{drift:>+10.1%}{pw:>10.3f}{dest:>7.2f}'
              + (f'   ждали {pred}' if pred else ''), flush=True)

    for d in (2, 3, 4):
        line(f'засев d={d}', sprinkle_d(N, d, rs),
             pred=f'поток 0, пок. {-2/d:.2f}')
    for k, q in ((3, 0.2), (3, 0.1), (5, 0.1)):
        Rc = layered(N, k, q, rs)
        a = N // k
        if k == 3:
            form = (2 * q + 1 - (1 - q * q) ** a) / 4.5
            lim = 2 * q / 4.5
            print(f'   формула k=3 q={q}: при a={a} даёт r={form:.4f}, '
                  f'предел при малых a = {lim:.4f}')
        line(f'слоистый k={k} q={q}', Rc)

    print('\nЧТЕНИЕ: «поток r» — относительный сдвиг доли связанных от')
    print('n=2400 к n=150. У неподвижной точки он около нуля. Показатель')
    print('L/R даёт размерность как -2/показатель, и для засевов обязан')
    print('воспроизвести d.')


if __name__ == '__main__':
    main()
