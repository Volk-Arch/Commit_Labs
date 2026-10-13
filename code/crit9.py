#!/usr/bin/env python3
"""КРИТИЧНОСТЬ: блочные ошибки и доля связанных в критической точке.

crit8.py дал расхождение 8.1 сигмы с точным ответом при согласии ~2% по
размаху. Главное подозрение — НЕ физика, а моя оценка погрешности: я
делил разброс на корень из числа выборок, то есть считал коррелированные
выборки независимыми. У MCMC в области перехода автокорреляция велика, и
настоящая ошибка много больше. Чинится блочными средними.

И добавляется величина, ради которой весь маршрут: ДОЛЯ СВЯЗАННЫХ в
критической точке. Если надежда маршрута верна, она должна отвечать
четырёхмерию (около 0.1 по Майрхейму–Майеру).

КОНТРОЛИ:
  A. при N=6 отжиг против точного перечисления, ошибка БЛОЧНАЯ;
  B. приращения против полного пересчёта;
  C. доля принятых ходов и доля обменов печатаются всегда;
  D. число отношений считается приращениями и сверяется с R.sum().
"""
import math

import numpy as np

from crit import K4, WB, action_full, enumerate_exact, transitive_ok_add
from crit4 import cells_contrib


def blockerr(x, nb=20):
    x = np.asarray(x, float)
    if len(x) < 2 * nb:
        return float(x.std() / math.sqrt(max(len(x), 1)))
    k = len(x) // nb
    bm = x[:k * nb].reshape(nb, k).mean(axis=1)
    return float(bm.std(ddof=1) / math.sqrt(nb))


class Ch:
    def __init__(self, n, beta):
        self.n, self.beta = n, beta
        self.R = np.zeros((n, n), bool)
        self.Iv = np.zeros((n, n), np.int32)
        self.c = np.zeros(4, np.int64)
        self.nrel = 0
        self.S = K4 * n
        self.acc = self.tries = 0

    def act(self):
        return K4 * (self.n + sum(WB[m] * int(self.c[m])
                                  for m in range(4)))

    def step(self, i, j, rs):
        R, Iv = self.R, self.Iv
        if (R[i] & R[:, j]).any():
            return
        if not transitive_ok_add(R, i, j):
            return
        add = not R[i, j]
        self.tries += 1
        before = cells_contrib(R, Iv, i, j)
        sg = 1 if add else -1
        dIvi = R[j].astype(np.int32) * sg
        dIvj = R[:, i].astype(np.int32) * sg
        R[i, j] = add
        Iv[i] += dIvi
        Iv[:, j] += dIvj
        after = cells_contrib(R, Iv, i, j)
        self.c += after - before
        S2 = self.act()
        if rs.random() < math.exp(min(0.0, -self.beta * (S2 - self.S))):
            self.S = S2
            self.nrel += sg
            self.acc += 1
        else:
            R[i, j] = not add
            Iv[i] -= dIvi
            Iv[:, j] -= dIvj
            self.c += before - after


def run(n, betas, nsweep, rs, burn=None, check=300):
    ch = [Ch(n, b) for b in betas]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npr = len(pairs)
    burn = burn if burn is not None else nsweep // 3
    Ss = [[] for _ in betas]
    Rs = [[] for _ in betas]
    sw = tr = bad = 0
    for it in range(nsweep + burn):
        for c in ch:
            for _ in range(npr):
                i, j = pairs[rs.integers(npr)]
                c.step(i, j, rs)
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            tr += 1
            if rs.random() < math.exp(min(
                    0.0, (a.beta - b.beta) * (a.S - b.S))):
                sw += 1
                a.R, b.R = b.R, a.R
                a.Iv, b.Iv = b.Iv, a.Iv
                a.c, b.c = b.c, a.c
                a.S, b.S = b.S, a.S
                a.nrel, b.nrel = b.nrel, a.nrel
        if it >= burn:
            for k, c in enumerate(ch):
                Ss[k].append(c.S)
                Rs[k].append(c.nrel)
            if check and it % check == 0:
                for c in ch:
                    sf, _ = action_full(c.R)
                    if abs(sf - c.S) > 1e-6 or c.nrel != int(c.R.sum()):
                        bad += 1
    return ([np.array(s) for s in Ss], [np.array(r) for r in Rs],
            [c.acc / max(c.tries, 1) for c in ch], sw / max(tr, 1), bad)


BET = [0.0, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12,
       0.15, 0.2, 0.3]


def main():
    print(__doc__)
    rs = np.random.default_rng(24680)

    print('КОНТРОЛЬ A: N = 6, блочные ошибки')
    S6 = enumerate_exact(6)
    Ss, Rs, accr, swr, bad = run(6, BET, 2400, rs)
    print(f'  обменов {swr:.3f}, сбоев приращений {bad}, '
          f'принятых {min(accr):.3f}…{max(accr):.3f}')
    print(f"{'beta':>6}{'точно':>10}{'отжиг':>10}{'блоч.ош.':>10}"
          f"{'расх.':>8}{'наивн.ош.':>11}")
    worst = 0.0
    for k, b in enumerate(BET):
        w = np.exp(-b * (S6 - S6.min())); w /= w.sum()
        ex = float((S6 * w).sum())
        m = Ss[k].mean()
        be = max(blockerr(Ss[k]), 0.02)
        nv = Ss[k].std() / math.sqrt(len(Ss[k])) + 1e-9
        dv = abs(m - ex) / be
        worst = max(worst, dv)
        print(f'{b:>6.2f}{ex:>10.3f}{m:>10.3f}{be:>10.3f}'
              f'{dv:>7.1f}с{nv:>11.3f}')
    print(f'  наибольшее {worst:.1f}с => контроль '
          f'{"ПРОЙДЕН" if worst < 5 else "ПРОВАЛЕН"}')

    print('\nПИК ТЕПЛОЁМКОСТИ и ФАЗА В НЁМ')
    print(f"{'N':>4}{'beta пика':>11}{'C пика':>9}{'доля связ.':>12}"
          f"{'±':>8}{'обменов':>9}")
    rows = []
    for n in (6, 9, 12):
        Ss, Rs, accr, swr, bad = run(n, BET, 1300, rs)
        cs = [b * b * Ss[k].var() for k, b in enumerate(BET)]
        k = int(np.argmax(cs))
        npair = n * (n - 1) / 2
        r = Rs[k].mean() / npair
        re = blockerr(Rs[k]) / npair
        rows.append((n, BET[k], cs[k], r))
        print(f'{n:>4}{BET[k]:>11.3f}{cs[k]:>9.3f}{r:>12.4f}'
              f'{re:>8.4f}{swr:>9.3f}', flush=True)
    if len(rows) >= 3:
        ns = np.array([r[0] for r in rows], float)
        cm = np.array([r[2] for r in rows])
        p = np.polyfit(np.log(ns), np.log(np.maximum(cm, 1e-9)), 1)[0]
        print(f'\n  пик ~ N^{p:.3f} (первый род ждёт около 1 и выше)')
        print('  доля связанных в пике: ' + ', '.join(
            f'N={int(r[0])}: {r[3]:.3f}' for r in rows))
        print('  четырёхмерию отвечает 0.1; двумерию 0.5')


if __name__ == '__main__':
    main()
