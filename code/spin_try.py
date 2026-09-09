import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'phaselevel.py'), encoding='utf-8').read())

def orientation(U):
    """Антисимметричная величина события: чётность ранжировки его шкал.
    Меняет знак при обмене двух шкал (зеркало), не меняет при циклическом сдвиге."""
    N, k = U.shape
    s = np.ones(N)
    for i in range(k):
        for j in range(i+1, k):
            s = s*np.sign(U[:, i]-U[:, j])
    return s

def chain_forward(R, lab, want=200, minlen=4, rs=None):
    """Сохраняется ли метка вдоль мировых линий."""
    r = rs if rs is not None else np.random.default_rng(0)
    Rb = np.ascontiguousarray(np.asarray(R, dtype=bool))
    A = Rb.astype(np.float32); L = Rb & ((A@A) == 0)
    D = proper_time(np.asarray(R, dtype=np.float32))
    N = len(Rb); same = tot = 0; used = 0
    starts = np.flatnonzero(D <= np.quantile(D, 0.3))
    for _ in range(want*8):
        if used >= want: break
        cur = int(r.choice(starts)); ch = [cur]
        for _ in range(30):
            nxt = np.flatnonzero(L[:, cur])
            if len(nxt) == 0: break
            cur = int(r.choice(nxt)); ch.append(cur)
        if len(ch) < minlen: continue
        used += 1
        s = lab[np.array(ch)]
        same += int((s[1:] == s[:-1]).sum()); tot += len(s)-1
    return same, tot, used
