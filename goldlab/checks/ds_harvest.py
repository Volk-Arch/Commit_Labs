"""новый набор: де Ситтер (w = −1), h = 0.035, тот же ящик и тот же набор интервалов, что у наборов 5–9 окт."""
import sys, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import eos
seeds, per = int(sys.argv[1]), int(sys.argv[2]); rows, sd = [], []; t0 = time.time()
for s in range(seeds):
    rs = np.random.default_rng(520000 + 1931 * s); r = eos.harvest(eos.a_w(-1.0, 0.035), rs, per); rows += r; sd += [s] * len(r)
    if s % 10 == 9: print(s + 1, len(rows), f'{time.time() - t0:.0f} c', flush=True)
np.savez_compressed('/home/claude/code/goldlab_checks/ds_arm.npz', A=np.array(rows), sd=np.array(sd))
print('готово', len(rows))
