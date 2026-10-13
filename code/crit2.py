#!/usr/bin/env python3
"""КРИТИЧНОСТЬ, параллельный отжиг: crit.py замерзал.

Контроль в crit.py провалился: при beta=0.3 MCMC дал -53.89 ± 0.0000
против точного -107.77. Нулевая ошибка = все выборки одинаковы = цепь
ЗАМЕРЗЛА. При beta=0 контроль проходил, значит механика верна, а
равновесия при beta>0 нет, и весь скан по beta был артефактом.

Причина штатная: у перехода первого рода одиночные перевороты обязаны
замерзать. Лекарство — ПАРАЛЛЕЛЬНЫЙ ОТЖИГ: цепи на лестнице beta
обмениваются состояниями по Метрополису, и горячие цепи вытаскивают
холодные из ловушек.

Плюс здесь печатаются ДОЛЯ ПРИНЯТЫХ ходов и ДОЛЯ ОБМЕНОВ — именно их
отсутствие и позволило прошлой версии соврать.

Счёты обновляются приращениями: переворот R[i,j] меняет Iv только в
строке i и столбце j, значит и счёты N_m меняются только там — O(N) на
ход вместо O(N^2).

КОНТРОЛИ:
  A. точное перечисление при N=6 на КАЖДОМ beta лестницы;
  B. приращения против полного пересчёта, периодически;
  C. доля принятых ходов и доля обменов — если первая у нуля, выводов
     не делать.
"""
import math

import numpy as np

from crit import (K4, WB, action_full, enumerate_exact,
                  transitive_ok_add, transitive_ok_del)


def cells_contrib(R, Iv, i, j):
    """Вклад строки i и столбца j в счёты N_m (без двойного счёта)."""
    c = np.zeros(4, np.int64)
    for m in range(4):
        c[m] += int((R[i] & (Iv[i] == m)).sum())
        c[m] += int((R[:, j] & (Iv[:, j] == m)).sum())
        if R[i, j] and Iv[i, j] == m:
            c[m] -= 1
    return c


class Chain:
    def __init__(self, n, beta):
        self.n, self.beta = n, beta
        self.R = np.zeros((n, n), bool)
        self.Iv = np.zeros((n, n), np.int32)
        self.c = np.zeros(4, np.int64)
        self.S = K4 * n
        self.acc = 0
        self.tries = 0

    def act(self):
        return K4 * (self.n + sum(WB[m] * int(self.c[m])
                                  for m in range(4)))

    def step(self, i, j, rs):
        R, Iv = self.R, self.Iv
        add = not R[i, j]
        if add:
            if not transitive_ok_add(R, i, j):
                return
        else:
            if not transitive_ok_del(R, i, j):
                return
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
            self.acc += 1
        else:
            R[i, j] = not add
            Iv[i] -= dIvi
            Iv[:, j] -= dIvj
            self.c += before - after


def temper(n, betas, nsweep, rs, burn=None, check=200):
    ch = [Chain(n, b) for b in betas]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    np_ = len(pairs)
    burn = burn if burn is not None else nsweep // 3
    samples = [[] for _ in betas]
    swaps = tried = 0
    bad = 0
    for sw in range(nsweep + burn):
        for c in ch:
            for _ in range(np_):
                i, j = pairs[rs.integers(np_)]
                c.step(i, j, rs)
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            tried += 1
            d = (a.beta - b.beta) * (a.S - b.S)
            if rs.random() < math.exp(min(0.0, d)):
                swaps += 1
                a.R, b.R = b.R, a.R
                a.Iv, b.Iv = b.Iv, a.Iv
                a.c, b.c = b.c, a.c
                a.S, b.S = b.S, a.S
        if sw >= burn:
            for k, c in enumerate(ch):
                samples[k].append(c.S)
            if check and sw % check == 0:
                for c in ch:
                    sf, _ = action_full(c.R)
                    if abs(sf - c.S) > 1e-6:
                        bad += 1
    accr = [c.acc / max(c.tries, 1) for c in ch]
    return [np.array(s) for s in samples], accr, swaps / max(tried, 1), bad


def main():
    print(__doc__)
    rs = np.random.default_rng(90210)
    betas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9, 1.3]

    print('КОНТРОЛЬ A: точное перечисление против отжига, N = 6')
    ex = enumerate_exact(6)
    sm, accr, swr, bad = temper(6, betas, 500, rs)
    print(f'  обменов принято: {swr:.3f}; сбоев приращений: {bad}')
    print(f"{'beta':>7}{'точно <S>':>12}{'отжиг <S>':>12}{'±':>8}"
          f"{'расх.':>8}{'доля прин.':>12}")
    okall = True
    for k, b in enumerate(betas):
        w = np.exp(-b * (ex - ex.min()))
        exact = float((ex * w).sum() / w.sum())
        m = sm[k].mean()
        e = sm[k].std() / math.sqrt(max(len(sm[k]), 1))
        dv = abs(m - exact) / max(e, 1e-9)
        okall &= dv < 5
        print(f'{b:>7.2f}{exact:>12.3f}{m:>12.3f}{e:>8.3f}'
              f'{dv:>7.1f}с{accr[k]:>12.3f}')
    print(f'  => контроль {"ПРОЙДЕН" if okall else "ПРОВАЛЕН"}')
    if not okall:
        print('  выводов по скану не делать')
        return

    print('\nСКАН: двугорбость и доля связанных')
    for n in (14, 20):
        sm, accr, swr, bad = temper(n, betas, 320, rs)
        print(f'\n  N={n}: обменов {swr:.3f}, сбоев {bad}, '
              f'доля принятых от {min(accr):.3f} до {max(accr):.3f}')
        print(f"{'beta':>7}{'<S>':>10}{'разброс':>10}{'двугорб.':>10}"
              f"{'разрыв':>9}")
        for k, b in enumerate(betas):
            s = sm[k]
            h, ed = np.histogram(s, bins=20)
            pk = [i for i in range(1, 19)
                  if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 3]
            bim, gap = 'нет', 0.0
            if len(pk) >= 2:
                lo = h[pk[0]:pk[-1] + 1].min()
                if lo < 0.4 * min(h[pk[0]], h[pk[-1]]):
                    bim, gap = 'ДА', float(ed[pk[-1]] - ed[pk[0]])
            print(f'{b:>7.2f}{s.mean():>10.2f}{s.std():>10.2f}'
                  f'{bim:>10}{gap:>9.2f}', flush=True)


if __name__ == '__main__':
    main()
