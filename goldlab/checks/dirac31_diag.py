import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob
H, Rc, al, a, m = 20.0, 7.0, 0.45, 0.5, 1.0
U = lambda r: -al / np.sqrt(r * r + a * a); dU = lambda r: al * r / (r * r + a * a) ** 1.5
w = Wd.gold31(1.0, 'cylinder', H=H, Rc=Rc, Rs=3.0); X = w.X; z = np.load('links31_H20_R7.npz'); I, J = z['I'], z['J']; ph = Pr.wilson31(X, I, J, U)
r = np.linalg.norm(X[:, 1:], axis=1); prof = np.exp(-r * al); src0 = X[:, 0] < 0.45; sel = (X[:, 0] > 1.5) & (X[:, 0] < H - 0.3); tt = X[sel, 0]; om = np.linspace(0.8, 1.0, 2001)
rh = np.where(r[:, None] > 1e-9, X[:, 1:] / np.maximum(r, 1e-9)[:, None], 0)
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex)
sr = rh[:, 0, None, None] * sx + rh[:, 1, None, None] * sy + rh[:, 2, None, None] * sz
e = np.array([1, 0]); f = np.where(src0[:, None], prof[:, None] * e[None, :], 0)
for nm, c in (('0', 0), ('i', 1j), ('2i', 2j), ('1 (без i)', 1.0), ('i·σ_z вместо σ·r̂ (константа)', None)):
    if c is None: Bx = -m * m * np.eye(2)[None] + 1j * (dU(r)[:, None, None] * sz[None])
    else: Bx = -m * m * np.eye(2)[None] + c * (dU(r)[:, None, None] * sr)
    ps = Pr.johnston31_respond_spinor(w.n, I, J, 1.0, m, f, ph, Bx); S = Ob.spectrum(tt, prof[sel] * (ps[sel] @ e), om, 'hann'); pk = Ob.peaks(om, S, 0.0)
    print(nm, round(pk[int(np.argmax([h for _, h in pk]))][0], 5), flush=True)
