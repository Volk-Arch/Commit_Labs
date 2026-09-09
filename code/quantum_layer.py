import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from double_slit import build_causet, direct_links, depth, topo_order

def entangle_pairs(N, pairs, d=2, rs=None):
    """Каждой ПАРЕ событий даём общее состояние размерности d*d,
    не разлагающееся на произведение. Это и есть квантовый слой."""
    r = rs if rs is not None else np.random.default_rng(0)
    states = {}
    for (a, b) in pairs:
        # состояние Белла с случайной фазой
        ph = r.random()*2*np.pi
        psi = np.zeros(d*d, dtype=complex)
        psi[0] = 1/np.sqrt(2)
        psi[-1] = np.exp(1j*ph)/np.sqrt(2)
        states[(a, b)] = psi
    return states

def sigma(theta):
    return np.array([[np.cos(theta), np.sin(theta)],
                     [np.sin(theta), -np.cos(theta)]], dtype=complex)

def correlate(psi, ta, tb):
    M = np.kron(sigma(ta), sigma(tb))
    return float(np.real(psi.conj() @ M @ psi))

def chsh_from(psi, a0, a1, b0, b1):
    return abs(correlate(psi, a0, b0) + correlate(psi, a0, b1)
               + correlate(psi, a1, b0) - correlate(psi, a1, b1))
