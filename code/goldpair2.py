import sys, math, numpy as np
import goldpair
from goldpair import run
goldpair.NORM = 1.0
from universe import golden_flat, poisson_flat
for name, make in (('кристалл', golden_flat), ('засев', poisson_flat)):
    for sh in (1.2, 2.5, 4.0):
        if sh == 1.2: continue
        res = [run(make, math.pi, s, src_half=sh) for s in (21, 22, 23, 24, 25, 26)]
        C = np.array([r[1] for r in res]); S = np.array([r[2] for r in res]); Sm = np.array([r[3] for r in res])
        print(f'{name}, источник ±{sh}: событий {[r[0] for r in res]}; C = {C.mean():.3f} (от {C.min():.3f} до {C.max():.3f}); '
              f'CHSH когерентно {S.mean():.3f} (макс {S.max():.3f}); смесь макс {Sm.max():.3f}'); sys.stdout.flush()
