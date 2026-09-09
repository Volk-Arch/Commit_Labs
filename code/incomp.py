import numpy as np
import sys; sys.path.insert(0, '/home/claude')
from twistor import rel_of

def analyse(X, R, pairs, label):
    d = np.abs(X[pairs[:, 0]] - X[pairs[:, 1]])
    s = (X[pairs[:, 0]] > X[pairs[:, 1]]).sum(1)
    from collections import Counter
    c = Counter(s.tolist())
    dist = " ".join(f"{a}:{b/len(s):.2f}" for a, b in sorted(c.items()))
    print(f"  {label:>14}: |разн| медиана {np.median(d):.4f}, "
          f"мин {np.median(d.min(1)):.4f}, макс {np.median(d.max(1)):.4f}")
    print(f"  {'':>14}  шкал 'за': {dist}")
