import numpy as np

def random_circuit(nq, depth, rs, two_frac=0.5):
    """Случайная квантовая схема: вентили на кубитах.
    Возвращает список (номер, кубиты)."""
    gates = []
    for _ in range(depth):
        used = set()
        order = rs.permutation(nq)
        for q in order:
            if q in used: continue
            if rs.random() < two_frac and len(used) < nq-1:
                partners = [p for p in order if p not in used and p != q]
                if partners:
                    p = int(partners[0])
                    gates.append((len(gates), (int(q), p)))
                    used.add(q); used.add(p); continue
            gates.append((len(gates), (int(q),)))
            used.add(q)
    return gates


def circuit_order(gates, nq):
    """Причинный порядок вентилей: A раньше B, если делят кубит и A раньше."""
    n = len(gates)
    last = {q: None for q in range(nq)}
    E = []
    for idx, qs in gates:
        for q in qs:
            if last[q] is not None: E.append((last[q], idx))
            last[q] = idx
    R = np.zeros((n, n), dtype=bool)
    for a, b in E: R[b, a] = True
    for _ in range(80):
        Rn = R | ((R.astype(np.float32) @ R.astype(np.float32)) > 0)
        if Rn.sum() == R.sum(): break
        R = Rn
    np.fill_diagonal(R, False)
    return R
