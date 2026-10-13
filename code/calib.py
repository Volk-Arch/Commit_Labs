"""Засев 4d-ромба: как у события связаны размер прошлого |P|, его высота h
и число связей L. Нужна форма закона для однострочных правил."""
import sys, numpy as np
sys.path.insert(0,'/home/claude'); sys.path.insert(0,'.')
from drawlab import sprinkle_R
from core import links_of, height
for N in (400, 1600):
    rs = np.random.default_rng(3); R = sprinkle_R(N, rs)
    P = R.sum(1); L = links_of(R).sum(1); _, D = height(R); h = D + 1
    m = P > 0
    print(f'N={N}')
    for hh in sorted(set(h[m])):
        sel = m & (h == hh)
        if sel.sum() < 5: continue
        r = P[sel] / hh ** 4
        print(f'  h={hh:2d} k={sel.sum():4d}  |P|/h^4: медиана {np.median(r):.3f} [5%..95%] {np.quantile(r,.05):.3f}..{np.quantile(r,.95):.3f}'
              f'   L медиана {np.median(L[sel]):.0f}  L/√|P| {np.median(L[sel]/np.sqrt(P[sel])):.2f}')
