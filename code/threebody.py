"""Три тела в зернистом мире: «восьмёрка» Шенсине–Монтгомери (Ньютон, G = m = 1) плюс толчки
дискретности по скорости. Два вида зерна (из swerve-doc):
  посыпка — толчки независимы: скорость блуждает (случайное блуждание);
  кристалл — толчки = приращения ограниченной квазипериодической последовательности
  (золотой поворот): сумма толчков ограничена, скорость дрожит, но не уходит.
Одинаковая сила толчка (rms за шаг). Сколько периодов живёт восьмёрка?
Разрушение: тела разлетелись (расстояние > 5) или форма ушла (отклонение от восьмёрки > 0.5)."""
import numpy as np, math, json, sys
TAU = (1 + 5 ** .5) / 2
X0 = np.array([[0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0.0, 0.0]])
V3 = np.array([-0.93240737, -0.86473146]); V0 = np.array([-V3 / 2, -V3 / 2, V3])
PER = 6.32591398
def acc(X):
    a = np.zeros_like(X)
    for i in range(3):
        for j in range(3):
            if i != j:
                d = X[j] - X[i]; a[i] += d / (np.dot(d, d) ** 1.5)
    return a
def ref_orbit(n_per_period):
    dt = PER / n_per_period; X, V = X0.copy(), V0.copy(); out = [X.copy()]
    a = acc(X)
    for k in range(n_per_period):
        V += 0.5 * dt * a; X += dt * V; a = acc(X); V += 0.5 * dt * a; out.append(X.copy())
    return np.array(out)
def run(kind, amp, periods=400, nper=2000, seed=0):
    dt = PER / nper; X, V = X0.copy(), V0.copy(); a = acc(X)
    ref = ref_orbit(nper)
    rng = np.random.default_rng(seed); phase = rng.random((3, 2)) * 10
    prev = np.zeros((3, 2)); traj = []
    for step in range(periods * nper):
        V += 0.5 * dt * a; X += dt * V; a = acc(X); V += 0.5 * dt * a
        if kind == 'poisson':
            V += amp * rng.normal(size=(3, 2))
        elif kind == 'crystal':
            s = ((step + 1) / TAU + phase) % 1.0 - 0.5              # ограниченная золотая последовательность
            V += amp * math.sqrt(12) * (s - prev); prev = s
        V -= V.mean(0)                                           # центр масс на месте
        if step % 50 == 0 and len(traj) < 4000: traj.append(X.copy())
        if step % nper == 0:
            k = step % nper; dmax = np.abs(X[:, None, :] - X[None, :, :]).max()
            # отклонение формы: расстояние до ближайшей точки эталонной восьмёрки (с учётом перестановок/фазы — грубо)
            dev = min(np.abs(np.sort(np.linalg.norm(X, axis=1)) - np.sort(np.linalg.norm(R, axis=1))).max() for R in ref[::20])
            if dmax > 5 or dev > 0.5: return step / nper, traj
    return periods, traj
if __name__ == '__main__':
    res = {}
    for amp in (3e-4, 1e-3, 3e-3):
        for kind in ('none', 'crystal', 'poisson'):
            if kind == 'none' and amp != 3e-4: continue
            lives = [run(kind, amp if kind != 'none' else 0, seed=s)[0] for s in range(4 if kind != 'none' else 1)]
            res[f'{kind}_{amp}'] = lives
            print(f'{kind:8s} толчок {amp:.0e}: восьмёрка прожила {lives} периодов (из 400)'); sys.stdout.flush()
    json.dump(res, open('/home/claude/data/threebody.json', 'w'))
