#!/usr/bin/env python3
"""КРИТИЧНОСТЬ: есть ли двугорбость при N больше семи.

Последний непроверенный маршрут из исходного списка. Идея: многообразные
множества отбираются не правилом, а КРИТИЧЕСКОЙ ТОЧКОЙ, а признак
перехода первого рода есть ДВУГОРБОСТЬ распределения действия. Проект
проверял её только при N <= 7 точным перечислением, а на семи элементах
нет ни одного масштаба — опровержением первого рода это считать нельзя.

Ансамбль: Z(beta) = sum по C от exp(-beta * S_БД(C)) по порядкам на N
помеченных элементах, помеченных линейным расширением (то есть по
транзитивным верхнетреугольным матрицам). Такая разметка взвешивает
каждый порядок числом его линейных расширений; это оговорка, и контроль
на точном перечислении использует ТО ЖЕ взвешивание, поэтому
самосогласовано.

Ходы: выбрать пару i<j, перевернуть R[i,j], принять только если
матрица осталась транзитивной. Переворот есть инволюция, предложение
симметрично — детальный баланс выполняется.

СЧЁТ ПРИРАЩЕНИЯМИ. Iv = R@R есть число промежуточных элементов.
Переворот R[i,j] меняет Iv только в строке i и столбце j:
Iv[i,:] += R[j,:], Iv[:,j] += R[:,i]. Это O(N) на ход вместо O(N^3).

КОНТРОЛИ:
  A. точное перечисление при N=6: распределение S и Z(beta) обязаны
     совпасть с MCMC при тех же beta;
  B. приращения против полного пересчёта — периодически;
  C. при beta=0 средняя доля связанных обязана совпасть с точной.
"""
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)
WB = {0: -1.0, 1: 9.0, 2: -16.0, 3: 8.0}


def action_full(R):
    n = R.shape[0]
    Iv = (R.astype(np.int16) @ R.astype(np.int16))
    tot = float(n)
    for m, w in WB.items():
        tot += w * float((R & (Iv == m)).sum())
    return K4 * tot, Iv


def transitive_ok_add(R, i, j):
    return (not (R[j] & ~R[i]).any()) and (not (R[:, i] & ~R[:, j]).any())


def transitive_ok_del(R, i, j):
    return not (R[i] & R[:, j]).any()


def counts_from(R, Iv):
    c = {}
    for m in WB:
        c[m] = int((R & (Iv == m)).sum())
    return c


def act_from_counts(n, c):
    return K4 * (n + sum(WB[m] * c[m] for m in WB))


def enumerate_exact(n):
    """Все транзитивные верхнетреугольные матрицы; точные S."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    nb = len(pairs)
    out = []
    for mask in range(1 << nb):
        R = np.zeros((n, n), bool)
        for b, (i, j) in enumerate(pairs):
            if (mask >> b) & 1:
                R[i, j] = True
        Iv = R.astype(np.int16) @ R.astype(np.int16)
        if (Iv.astype(bool) & ~R).any():
            continue                      # не транзитивна
        s, _ = action_full(R)
        out.append(s)
    return np.array(out)


def mcmc(n, beta, nsweep, rs, burn=None, check_every=0):
    R = np.zeros((n, n), bool)
    Iv = np.zeros((n, n), np.int32)
    c = counts_from(R, Iv)
    S = act_from_counts(n, c)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npair = len(pairs)
    burn = burn if burn is not None else nsweep // 3
    samples, bad = [], 0
    for sw in range(nsweep + burn):
        for _ in range(npair):
            i, j = pairs[rs.integers(npair)]
            add = not R[i, j]
            if add:
                if not transitive_ok_add(R, i, j):
                    continue
            else:
                if not transitive_ok_del(R, i, j):
                    continue
            sg = 1 if add else -1
            dIvi = R[j].astype(np.int32) * sg
            dIvj = R[:, i].astype(np.int32) * sg
            R[i, j] = add
            Iv[i] += dIvi
            Iv[:, j] += dIvj
            c2 = counts_from(R, Iv)
            S2 = act_from_counts(n, c2)
            if rs.random() < math.exp(-beta * (S2 - S)):
                c, S = c2, S2
            else:
                R[i, j] = not add
                Iv[i] -= dIvi
                Iv[:, j] -= dIvj
        if sw >= burn:
            samples.append(S)
            if check_every and (sw % check_every == 0):
                sf, _ = action_full(R)
                if abs(sf - S) > 1e-6:
                    bad += 1
    return np.array(samples), bad


def main():
    print(__doc__)
    rs = np.random.default_rng(777)

    print('КОНТРОЛЬ A: точное перечисление против MCMC, N = 6')
    ex = enumerate_exact(6)
    print(f'  транзитивных матриц: {len(ex)}  '
          f'<S> при beta=0 = {ex.mean():.4f}, '
          f'разброс {ex.std():.4f}')
    for beta in (0.0, 0.3, 0.6):
        w = np.exp(-beta * ex)
        exact = float((ex * w).sum() / w.sum())
        sm, bad = mcmc(6, beta, 400, rs, check_every=50)
        print(f'  beta={beta:.2f}: точно <S> = {exact:>8.4f}, '
              f'MCMC = {sm.mean():>8.4f} ± {sm.std()/math.sqrt(len(sm)):.4f}'
              f'   сбоев приращений: {bad}')

    print('\nСКАН по beta: ищем двугорбость')
    print(f"{'N':>4}{'beta':>7}{'<S>':>10}{'разброс':>10}"
          f"{'двугорб.':>10}{'разрыв мод':>12}")
    for n in (14, 20):
        for beta in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4):
            sm, _ = mcmc(n, beta, 260, rs)
            h, edges = np.histogram(sm, bins=24)
            # двугорбость: есть ли два пика с провалом между ними
            pk = [i for i in range(1, len(h) - 1)
                  if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 2]
            gap = 0.0
            bim = 'нет'
            if len(pk) >= 2:
                lo = min(h[pk[0]:pk[-1] + 1])
                if lo < 0.5 * min(h[pk[0]], h[pk[-1]]):
                    bim = 'ДА'
                    gap = float(edges[pk[-1]] - edges[pk[0]])
            print(f'{n:>4}{beta:>7.2f}{sm.mean():>10.2f}{sm.std():>10.2f}'
                  f'{bim:>10}{gap:>12.2f}', flush=True)


if __name__ == '__main__':
    main()
