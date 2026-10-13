import sys, math, numpy as np, os
sys.path.insert(0, '/home/claude/code'); sys.argv = ['x']
import vacuumcrystal as vc
kind = os.environ['KIND']; n = int(os.environ['N'])
rs = np.random.default_rng(424242 + (1 if kind == 'вдоль r' else 2))
d = np.array([vc.one(1.0, kind, rs, True) for _ in range(n)])
print(f'без модуляции густоты, γ=+1 {kind}: δ = {d.mean():+.5f} ± {d.std(ddof=1)/math.sqrt(n):.5f}')
