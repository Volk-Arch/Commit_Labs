#!/usr/bin/env python3
"""СЛИЯНИЕ КУСКОВ и складной нож по засевам для показателя.

ЗАЧЕМ. Прогоны `bigfall` детерминированы: без сдвига `--seed0`
повторный запуск даёт ТЕ ЖЕ миры (`bf120.out` и `bf120c.out` совпали
посимвольно). Поэтому засевы набираются кусками со сдвигом, а здесь
куски сливаются.

ПОЧЕМУ НОЖ ПО ЗАСЕВАМ, А НЕ ПОЯЧЕЕЧНЫЕ ОШИБКИ. Полосы считаются по
одним и тем же миpам, значит их ошибки СКОРРЕЛИРОВАНЫ: статистический
сдвиг двигает все полосы вместе и на показатель почти не влияет.
Распространение поячеечных ошибок как независимых давало ±0.10, разброс
вокруг прямой ±0.013 — втрое расхождение. Нож по засевам (единица
независимости — мир) учитывает обе стороны и не требует ковариации.

  python3 merge.py [N]
"""
import glob
import math
import pickle
import sys

import numpy as np

SCRATCH = ("/tmp/claude-0/-home-claude/"
           "8f20417e-eecf-5054-8071-09c440210416/scratchpad")


def slope(cent, vals, sl):
    x = np.log(np.asarray(cent)[sl])
    y = np.asarray(vals)[sl]
    m = y < 0
    if m.sum() < 3:
        return float('nan')
    return float(np.polyfit(x[m], np.log(-y[m]), 1)[0])


def main():
    N = sys.argv[1] if len(sys.argv) > 1 else '*'
    files = sorted(glob.glob(f'{SCRATCH}/bigfall_{N}_*.pkl'))
    if not files:
        print('кусков не найдено'); return
    print(__doc__.split('  python3')[0].rstrip()); print()
    by = {}
    for f in files:
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
        key = d['n']
        by.setdefault(key, {'r': d['r_ell'], 'rows': []})
        per = d['per_seed']
        ns = min(len(p) for p in per)
        for i in range(ns):
            by[key]['rows'].append([per[q][i] for q in range(len(per))])
        print(f'  {f.split("/")[-1]}: {len(per)} полос x {ns} засевов')
    print()
    for n, d in sorted(by.items()):
        A = np.array(d['rows'], float)
        cent = d['r']
        ns, nb = A.shape
        print('=' * 76)
        print(f'N = {n}:  {nb} полос, {ns} засевов слито')
        print(f"  {'r в ell':>9}{'дефицит':>13}{'ст.ош':>11}{'сигм':>7}")
        for q in range(nb):
            v = A[:, q]
            se = v.std(ddof=1) / math.sqrt(ns)
            print(f'  {cent[q]:>9.1f}{v.mean():>13.6f}{se:>11.6f}'
                  f'{abs(v.mean())/max(se,1e-12):>7.1f}')
        half = nb // 2
        print()
        print(f"  {'окно':>10}{'показатель':>13}{'±':>9}{'от −1':>9}")
        res = {}
        for lab, sl in (('все', slice(None)), ('внутр.', slice(0, half + 1)),
                        ('внешн.', slice(half, None))):
            full = slope(cent, A.mean(axis=0), sl)
            jk = np.array([slope(cent, np.delete(A, i, axis=0).mean(axis=0), sl)
                           for i in range(ns)])
            jk = jk[np.isfinite(jk)]
            se = (math.sqrt((len(jk) - 1) / len(jk)
                            * ((jk - jk.mean()) ** 2).sum())
                  if len(jk) > 1 else float('nan'))
            print(f'  {lab:>10}{full:>13.3f}{se:>9.3f}'
                  f'{(full+1)/max(se,1e-9):>+9.1f}')
            res[lab] = (full, se, jk)
        d0 = res['внутр.'][0] - res['внешн.'][0]
        dj = res['внутр.'][2] - res['внешн.'][2]
        sd = (math.sqrt((len(dj) - 1) / len(dj)
                        * ((dj - dj.mean()) ** 2).sum())
              if len(dj) > 1 else float('nan'))
        print(f'  РАЗНОСТЬ внутр. − внешн. = {d0:+.3f} ± {sd:.3f}'
              f' = {d0/max(sd,1e-9):+.1f} сигм')
        print()


if __name__ == '__main__':
    main()
