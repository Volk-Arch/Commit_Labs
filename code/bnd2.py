"""ПРАВИЛЬНЫЙ контроль: держим плотность, растим ОБЛАСТЬ.

Прошлый контроль был негоден: я измельчал мир (ромб всегда T = 1), то
есть расстояние наблюдателя до границы не менялось, и краевой эффект
обязан был остаться постоянным — он и остался (N^+0.000).

Здесь ell фиксировано, T растёт, наблюдатель тот же и в центре. Тогда
граница отодвигается, и если зависимость от быстроты краевая, она обязана
УБЫВАТЬ.

Считаются только отношения наблюдатель-фон: O(dens*N), влезает в память.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from leak import obs_proper

ELL, TAU, DENS, SEEDS = 0.107, 0.22, 20, 10

def sprinkle(N, T, rs, chunk=400000):
    out, got = [], 0
    while got < N:
        m = min(chunk, (N-got)*9 + 1000)
        Y = np.empty((m, 4))
        Y[:, 0] = rs.uniform(0, T, m)
        Y[:, 1:] = rs.uniform(-T/2, T/2, (m, 3))
        r = np.linalg.norm(Y[:, 1:], axis=1)
        Y = Y[(Y[:, 0] >= r) & (T - Y[:, 0] >= r)]
        if len(Y): out.append(Y); got += len(Y)
    return np.vstack(out)[:N]

def obs_rel(X, Z):
    """Число причинных пар наблюдатель-фон. O(dens*N), без матриц N*N."""
    tot = 0
    for k in range(len(Z)):
        dt = X[:, 0] - Z[k, 0]
        dx = np.linalg.norm(X[:, 1:] - Z[k, 1:], axis=1)
        tot += int((np.abs(dt) > dx).sum())
    return tot

print(f'ell фиксировано = {ELL}, наблюдатель {DENS} событий, '
      f'собственное время {TAU}, засевов {SEEDS}')
print('T растёт => граница отодвигается')
print()
print(f"{'T':>6}{'N':>9}{'радиус/tau':>12}{'отн ρ=0':>11}{'отн ρ=0.9':>12}"
      f"{'рост %':>9}{'сигм':>7}")
rows = []
for T in (1.0, 1.4, 1.8, 2.4):
    N = int(round(math.pi/24.0 * T**4 / ELL**4))
    d0, d9 = [], []
    for s in range(SEEDS):
        X = sprinkle(N, T, np.random.default_rng(1300+37*s))
        for rap, acc in ((0.0, d0), (0.9, d9)):
            Z = obs_proper(DENS, TAU, rap, T/2.0, 0.0)
            acc.append(obs_rel(X, Z))
        del X
    d0 = np.array(d0, float); d9 = np.array(d9, float)
    dd = d9.mean() - d0.mean()
    sd = math.hypot(d0.std(ddof=1)/math.sqrt(len(d0)),
                    d9.std(ddof=1)/math.sqrt(len(d9)))
    pc = 100*dd/d0.mean()
    rows.append((T, abs(pc)))
    print(f'{T:>6.1f}{N:>9}{(T/2)/TAU:>12.1f}{d0.mean():>11.0f}'
          f'{d9.mean():>12.0f}{pc:>9.2f}{abs(dd)/max(sd,1e-9):>7.1f}',
          flush=True)
A = np.array(rows)
b = np.polyfit(np.log(A[:,0]), np.log(np.maximum(A[:,1],1e-6)), 1)[0]
print()
print(f'  рост в % ~ T^{b:+.3f}')
print('  убывает с T => ГРАНИЦА, вопрос не определён в конечной области;')
print('  стоит => физическая зависимость от скорости наблюдателя')
