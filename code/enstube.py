#!/usr/bin/env python3
"""ОБЯЗАТЕЛЬНЫЙ КОНТРОЛЬ: та же машинерия с ТРУБКОЙ в роли объекта.

ЗАЧЕМ. `ensemble.py` перемеряет архивный результат `weight-doc.md`
(6.1σ обеднение фона у трубки в связанном ансамбле `exp(−βS)`) с
РАЗРЕЖЕНИЕМ в роли объекта, потому что вчера установлено: плотная трубка
— анти-масса (лестница), а масса есть разрежение.

Но без этого контроля результат не истолковать. Если моя машинерия не
воспроизводит архивное обеднение С ТРУБКОЙ, то перевёрнутый знак с
разрежением ничего не скажет — неизвестно, та ли это конструкция.

    трубка даёт обеднение + разрежение даёт сбор  -> знак перевернулся,
                                                    гипотеза верна
    трубка даёт обеднение + разрежение тоже       -> гипотеза неверна,
                                                    диагноз архива стоит
    трубка НЕ даёт обеднения                      -> машинерия не та,
                                                    оба вывода пусты

УСТРОЙСТВО. Трубка — `M` ЗАМОРОЖЕННЫХ событий на оси `|x| = 0`,
равномерно по времени; они не двигаются (как в архиве). Фон подвижен.
Отсчёт — прогон с тем же числом подвижных событий и БЕЗ трубки, чтобы
краевая перестройка вычиталась (ловушка, обойдённая в `ensemble.py`: в
пробе корзина 0.35–0.40 дала 1.112 ± 0.024 при объекте в центре — это
край, а не объект).

  python3 enstube.py
  python3 enstube.py --n 280 --tube 30 --steps 30000 --seeds 4
"""
import argparse
import math
import time

import numpy as np

from ensemble import inside_diamond, action_and_pairs, allowed, W


def sprinkle_mobile(N, T, rs):
    out, got = [], 0
    while got < N:
        m = (N - got) * 9 + 128
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T / 2, T / 2, (m, 3))
        Y = Y[inside_diamond(Y, T)]
        if len(Y):
            out.append(Y)
            got += len(Y)
    return np.vstack(out)[:N]


def tube_events(M, T, half):
    """Замороженная трубка на оси: M событий, время в [half, T-half]."""
    Z = np.zeros((M, 4))
    Z[:, 0] = np.linspace(half, T - half, M)
    return Z


def run(N, T, beta, steps, M, tol, rs, jump):
    """Метрополис: подвижный фон + замороженная трубка (если M > 0)."""
    mob = sprinkle_mobile(N, T, rs)
    frz = tube_events(M, T, 0.12) if M > 0 else np.zeros((0, 4))
    X = np.vstack([mob, frz])
    nm = len(mob)
    S, P0 = action_and_pairs(X)
    acc, prof, nsamp, pairs = 0, np.zeros(10), 0, []
    edges = np.linspace(0.0, 0.5, 11)
    burn = steps // 4
    for t in range(steps):
        i = int(rs.integers(0, nm))          # только подвижные
        old = X[i].copy()
        p = old.copy()
        p[0] += rs.normal(0, jump)
        p[1:] += rs.normal(0, jump, 3)
        if not allowed(p, 0.0, [], T):
            continue
        X[i] = p
        Sn, Pn = action_and_pairs(X)
        if abs(Pn - P0) > tol * P0:
            X[i] = old
            continue
        d = Sn - S
        if d <= 0 or rs.random() < math.exp(-beta * d):
            S = Sn
            acc += 1
        else:
            X[i] = old
        if t >= burn and t % 25 == 0:
            r = np.linalg.norm(X[:nm, 1:], axis=1)   # профиль ФОНА
            prof += np.histogram(r, bins=edges)[0]
            nsamp += 1
            pairs.append(Pn)
    return prof / max(nsamp, 1), acc / steps, np.array(pairs)


def main():
    ap = argparse.ArgumentParser(description='Контроль трубкой')
    ap.add_argument('--n', type=int, default=280)
    ap.add_argument('--tube', type=int, default=30)
    ap.add_argument('--betas', type=float, nargs='+', default=[0.0, 1.0])
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--seeds', type=int, default=4)
    ap.add_argument('--tol', type=float, default=0.004)
    ap.add_argument('--jump', type=float, default=0.06)
    a = ap.parse_args()

    t0 = time.time()
    ell = (math.pi / 24.0 / a.n) ** 0.25
    print(f'N подвижных = {a.n}, трубка = {a.tube} замороженных событий '
          f'на оси')
    print(f'ell = {ell:.4f}; закрепление пар {100*a.tol:.1f} %, '
          f'шагов {a.steps}, засевов {a.seeds}')
    print('АРХИВ: 6.1σ ОБЕДНЕНИЕ фона внутри 0.27 от оси, подъём 0.38-0.5')
    print('ЭТО КОНТРОЛЬ: воспроизводится ли архив той же машинерией')
    print()

    store = {}
    for nm, M in (('БЕЗ трубки', 0), ('с трубкой', a.tube)):
        print(f'--- {nm}')
        base = {}
        for beta in a.betas:
            P, AC, PR = [], [], []
            for s in range(a.seeds):
                pr, ac, pp = run(a.n, 1.0, beta, a.steps, M, a.tol,
                                 np.random.default_rng(22100 + 97 * s),
                                 a.jump)
                P.append(pr); AC.append(ac)
                if len(pp):
                    PR.append(pp.std() / max(pp.mean(), 1))
            base[beta] = np.array(P)
            print(f'  beta = {beta:<5.1f} принято {100*np.mean(AC):5.1f} %'
                  f'   разброс числа пар {100*np.mean(PR):.2f} %')
        store[nm] = base

    ed = np.linspace(0.0, 0.5, 11)
    for beta in [b for b in a.betas if b != 0.0]:
        print()
        print(f'ОТНОШЕНИЕ плотности фона beta={beta:g} к beta=0, '
              f'и разница с прогоном БЕЗ трубки')
        print(f"  {'радиус':>12}{'без трубки':>14}{'с трубкой':>14}"
              f"{'разница':>13}{'сигм':>7}")
        for q in range(10):
            u0 = store['БЕЗ трубки'][0.0][:, q]
            v0 = store['с трубкой'][0.0][:, q]
            if u0.mean() < 3 or v0.mean() < 3:
                continue
            u = store['БЕЗ трубки'][beta][:, q] / max(u0.mean(), 1e-9)
            v = store['с трубкой'][beta][:, q] / max(v0.mean(), 1e-9)
            d = v.mean() - u.mean()
            sd = math.hypot(u.std(ddof=1) / math.sqrt(len(u)),
                            v.std(ddof=1) / math.sqrt(len(v)))
            print(f'  {f"{ed[q]:.2f}-{ed[q+1]:.2f}":>12}{u.mean():>14.3f}'
                  f'{v.mean():>14.3f}{d:>+13.3f}'
                  f'{abs(d)/max(sd,1e-9):>7.1f}')

    print()
    print('  разница < 0 у оси => обеднение, архив воспроизведён')
    print('  разницы нет => машинерия не та, и вывод по разрежению пуст')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
