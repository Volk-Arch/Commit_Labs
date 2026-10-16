import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import eigs
from goldlab import worlds as Wd, propagate as Pr, observe as Ob
H, Rc, m = 20.0, 7.0, 1.0
w = Wd.gold31(1.0, 'cylinder', H=H, Rc=Rc, Rs=3.0); X = w.X; z = np.load('links31_H20_R7.npz'); I, J = z['I'], z['J']
r = np.linalg.norm(X[:, 1:], axis=1); src0 = X[:, 0] < 0.45; sel = (X[:, 0] > 1.5) & (X[:, 0] < H - 0.3); tt = X[sel, 0]
rh = np.where(r[:, None] > 1e-9, X[:, 1:] / np.maximum(r, 1e-9)[:, None], 0)
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex)
sr = rh[:, 0, None, None] * sx + rh[:, 1, None, None] * sy + rh[:, 2, None, None] * sz; e = np.array([1, 0])
def cont(al, a, c, lo, hi):
    N = 800; R = 32.0; rr = np.linspace(R / N, R, N); h = rr[1] - rr[0]; V = -al / np.sqrt(rr * rr + a * a); dV = al * rr / (rr * rr + a * a) ** 1.5
    D2 = diags([np.ones(N - 1), -2 * np.ones(N), np.ones(N - 1)], [-1, 0, 1]) / h / h
    def mineig(E):
        A = D2 + diags((E - V) ** 2 - m * m); B = D2 + diags((E - V) ** 2 - m * m - 2 / rr ** 2)
        M = bmat([[A, diags(c * dV)], [diags(c * dV), B]]).tocsc().astype(complex); return abs(eigs(M, k=1, sigma=0, return_eigenvectors=False)[0])
    Es = np.linspace(lo, hi, 41); v = [mineig(E) for E in Es]; k = int(np.argmin(v)); Es2 = np.linspace(Es[max(k - 1, 0)], Es[min(k + 1, 40)], 41)
    v2 = [mineig(E) for E in Es2]; return Es2[int(np.argmin(v2))]
for al, a in ((0.45, 1.0), (0.8, 1.5), (1.2, 2.0)):
    U = lambda q, al=al, a=a: -al / np.sqrt(q * q + a * a); dU = lambda q, al=al, a=a: al * q / (q * q + a * a) ** 1.5
    ph = Pr.wilson31(X, I, J, U); prof = np.exp(-r * min(al, 0.7)); f = np.where(src0[:, None], prof[:, None] * e[None, :], 0)
    out = {}
    for nm, c in (('КГ', 0), ('Дирак (i)', 1j), ('без i', 1.0)):
        Bx = -m * m * np.eye(2)[None] + c * (dU(r)[:, None, None] * sr)
        ps = Pr.johnston31_respond_spinor(w.n, I, J, 1.0, m, f, ph, Bx); om = np.linspace(0.5, 1.05, 2201)
        S = Ob.spectrum(tt, prof[sel] * (ps[sel] @ e), om, 'hann'); pk = Ob.peaks(om, S, 0.0); out[nm] = pk[int(np.argmax([h for _, h in pk]))][0]
    ck = cont(al, a, 0, 0.5, 0.999); cd = cont(al, a, 1j, 0.5, 0.999)
    print(f'α={al} a={a}: кристалл КГ {out["КГ"]:.4f}, Дирак {out["Дирак (i)"]:.4f} (сдвиг {out["Дирак (i)"] - out["КГ"]:+.4f}), без i {out["без i"] - out["КГ"]:+.4f}; непр. КГ {ck:.4f}, Дирак {cd:.4f} (сдвиг {cd - ck:+.4f})', flush=True)
