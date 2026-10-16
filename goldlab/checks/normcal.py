"""Нормировка вершины пары статистической сшивкой: «вес» каждой частицы (сдвиг её уровня от постоянной вставки δb на
остановках) мерится на кристалле той же машиной; затем наклон пары предсказывается без подгонки:
dE/dλ = −2·w_a·w_b·⟨|d|⟩_канал, ⟨|d|⟩ = (J ± K)/σ (σ = −e²/2). Физический λ = σ_пары / (−2 w_a w_b)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe
e2, Z, m, rho = 0.2, 2.0, 1.0, 4.0; be = e2 / 2
xg, Ec, _, dens, vec = Fe.dirac_orbitals(lambda s: Z * e2 / 2 * np.abs(s), 1.0, m, nlev=2, N=512, Lg=60, locmax=40)
N = len(xg); h = xg[1] - xg[0]; rc0, rc1 = (d / np.trapezoid(d, xg) for d in dens); Tc = (np.conj(vec[:N, 0]) * vec[:N, 1] + np.conj(vec[N:, 0]) * vec[N:, 1]) / h
D = np.abs(xg[:, None] - xg[None]); J = rc0 @ D @ rc1 * h * h; K = np.real(Tc @ D @ np.conj(Tc)) * h * h; J00 = rc0 @ D @ rc0 * h * h   # ⟨|d|⟩ (ядро |d|)
a, b = vec[:N, 0], vec[:N, 1]; a = a / np.sqrt(np.sum(abs(a) ** 2) * h); b = b / np.sqrt(np.sum(abs(b) ** 2) * h); TR = np.conj(a) * b; KR = np.real(TR @ D @ np.conj(TR)) * h * h
print(f'⟨|d|⟩: прямое {J:.4f}, обмен (полный спинор) {K:.4f}, обмен (одна компонента) {KR:.4f}, оба внизу {J00:.4f}')
meas = {24: (-14.40, -8.78, -7.42), 26: (-11.97, -6.93, -6.09), 28: (-15.47, -11.12, -10.27)}
gs = lambda s: (1 + s) * np.exp(-s * s / 2); gs1 = lambda s: s * np.exp(-s * s / 8); om = np.linspace(0.9, 2.0, 2201)
for L in (24.0, 26.0, 28.0):
    w = Wd.gold11(rho, L); t, x = w.t, w.x; ts = -L + 10; UfN, dUN, _, _ = Fl.order_field11(w, 1.0, Z * e2, (ts, L - 4))
    sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); E = {}
    for db in (0.0, 0.004, -0.004):
        meff = np.sqrt(m * m - rho * db); row = []
        for src, Et in ((gs, Ec[0]), (gs1, Ec[1])):
            phi = Pr.corner_respond_general(w.X, UfN, dUN, meff, rho, np.where(np.abs(t - ts) < 0.45, src(x), 0), 1, 0.0)
            S = Ob.spectrum(t[sel], src(x[sel]) * phi[sel], om); row.append(min(Ob.peaks(om, S / S.max(), 0.03), key=lambda z: abs(z[0] - Et))[0])
        E[db] = row
    wa = (E[0.004][0] - E[-0.004][0]) / 0.008; wb = (E[0.004][1] - E[-0.004][1]) / 0.008
    pred = (-2 * wa * wb * (J - K), -2 * wa * wb * (J + K), -2 * wa * wa * J00); predR = (-2 * wa * wb * (J - KR), -2 * wa * wb * (J + KR))
    print(f'L={L:.0f}: E0 {E[0.0][0]:.4f} E1 {E[0.0][1]:.4f}; веса w_a {wa:+.3f} w_b {wb:+.3f} (KG-ожидание −ρ/(2E): {-rho / (2 * E[0.0][0]):+.3f}, {-rho / (2 * E[0.0][1]):+.3f})')
    print(f'   наклоны Ф/Б/ББ: предсказано {pred[0]:+.2f} / {pred[1]:+.2f} / {pred[2]:+.2f} (одна компонента: {predR[0]:+.2f} / {predR[1]:+.2f}); измерено {meas[int(L)][0]:+.2f} / {meas[int(L)][1]:+.2f} / {meas[int(L)][2]:+.2f}', flush=True)
    print(f'   измерено/предсказано: {meas[int(L)][0] / pred[0]:.2f} / {meas[int(L)][1] / pred[1]:.2f} / {meas[int(L)][2] / pred[2]:.2f};  λ_физ = σ/(−2w_a w_b) = {-be / (-2 * wa * wb):+.5f}', flush=True)
