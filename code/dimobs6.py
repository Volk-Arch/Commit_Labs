#!/usr/bin/env python3
"""ДАЁМ РАСТИ ОБОИМ. Последний кусок счёта классов.

Скан по N (`dimobs5.py`): метрика даёт 5, 5, 6, 4, 7 различимых из 12
при росте N впятеро — рост если и есть, то как sqrt(N) и утонул в
шуме. Слоистое отдаёт все 19 из 19 уже при N = 500, то есть упирается
в собственный предел семейства, а не в данные.

Остался один кусок: дать расти ОБОИМ семействам при одном N и
посмотреть, чья размерность растёт быстрее. Число классов идёт как
`exp(dim x ln(1/точность))`, значит скорость роста dim и решает, чей
сектор богаче КЛАССАМИ (в отличие от состояний, где иерархия измерена:
слоистое `e^{0.173 N^2}` против многообразного `e^{d N lnN}`).

  python3 dimobs6.py --fam lay --rich 20 --n 2000
"""
import argparse
import time

import numpy as np

import dimobs5 as D5
from dimobs import gen_conf
from dimobs4 import gen_layers_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fam', choices=('lay', 'met'), required=True)
    ap.add_argument('--rich', type=int, required=True)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--nd', type=int, default=16)
    ap.add_argument('--seeds', type=int, default=6)
    g = ap.parse_args()
    D5.NDEP = g.nd
    t0 = time.time()
    if g.fam == 'lay':
        L = g.rich
        th = ([1.0 / L] * L, [0.5] * (L - 1))
        st = [0.03] * L + [0.05] * (L - 1)
        gen, npar = gen_layers_p, 2 * L - 1
    else:
        K = g.rich
        th = ([0.0] * K,)
        st = [0.04] * K
        gen, npar = gen_conf, K
    sv, r = D5.paired(gen, g.n, th, st, g.seeds, 3100 + g.rich)
    top = ' '.join(f'{x:.0f}' for x in sv[:4])
    print(f'{g.fam:>6}{g.rich:>7}{npar:>8}{r:>8}{r/npar:>9.2f}'
          f'{time.time()-t0:>8.0f}   {top}', flush=True)


if __name__ == '__main__':
    main()
