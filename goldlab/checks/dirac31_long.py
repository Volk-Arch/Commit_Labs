import sys, os, numpy as np, time
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr, observe as Ob
H, Rc, m = float(sys.argv[1]), float(sys.argv[2]), 1.0; al = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45; a = 0.5; T0 = time.time()
w = Wd.gold31(1.0, 'cylinder', H=H, Rc=Rc, Rs=3.0); X = w.X; cache = f'links31_H{H:.0f}_R{Rc:.0f}.npz'
if os.path.exists(cache): z = np.load(cache); I, J = z['I'], z['J']
else: I, J = Cs.links_sparse(X, 1.0, 1.6); np.savez(cache, I=I, J=J)
print(f'мир {w.n} событий, {len(I)} связей, {time.time() - T0:.0f} c', flush=True)
U = lambda q: -al / np.sqrt(q * q + a * a); dU = lambda q: al * q / (q * q + a * a) ** 1.5; ph = Pr.wilson31(X, I, J, U)
r = np.linalg.norm(X[:, 1:], axis=1); src0 = X[:, 0] < 0.45; sel = (X[:, 0] > 1.5) & (X[:, 0] < H - 0.3); tt = X[sel, 0]; om = np.linspace(0.8, 1.0, 4001)
rh = np.where(r[:, None] > 1e-9, X[:, 1:] / np.maximum(r, 1e-9)[:, None], 0)
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex)
sr = rh[:, 0, None, None] * sx + rh[:, 1, None, None] * sy + rh[:, 2, None, None] * sz; e = np.array([1, 0])
for wid in (1.0, 0.5):
    prof = np.exp(-r * al * wid); f = np.where(src0[:, None], prof[:, None] * e[None, :], 0); out = {}
    for nm, c in (('КГ', 0), ('Дирак', 1j), ('без i', 1.0)):
        Bx = -m * m * np.eye(2)[None] + c * (dU(r)[:, None, None] * sr)
        ps = Pr.johnston31_respond_spinor(w.n, I, J, 1.0, m, f, ph, Bx); S = Ob.spectrum(tt, prof[sel] * (ps[sel] @ e), om, 'hann'); pk = Ob.peaks(om, S, 0.0)
        out[nm] = pk[int(np.argmax([h for _, h in pk]))][0]
    print(f'H={H:.0f} R={Rc:.0f} α={al} источник ×{wid}: КГ {out["КГ"]:.4f}, Дирак {out["Дирак"]:.4f} (сдвиг {out["Дирак"] - out["КГ"]:+.4f}), без i {out["без i"] - out["КГ"]:+.4f}  [{time.time() - T0:.0f} c]', flush=True)
