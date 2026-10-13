"""Монетка из хаоса по родословной. У события скрытая фаза θ на торе. Дети
(число — по золотому правилу a→acd, b→cd, c→abc, d→ab) получают θ_j = T(θ) + j·ω,
T — «кошка Арнольда» [[2,1],[1,1]] (числа τ², 1/τ²), ω = (1/τ, 1/τ²).
Монетка события = [θ₁ < 1/2]. Случайности нет: всё задаёт фаза корня.
Меряем по поколениям: e_k = (отклонение поколения)/√(n_k/4)."""
import numpy as np, math, sys
TAU = (1 + 5 ** .5) / 2
RULE = ([0, 2, 3], [2, 3], [0, 1, 2], [0, 1])
OM = np.array([1 / TAU, 1 / TAU ** 2])

def tree(roots, K, kind):
    """roots: фазы корней (R×2). Возвращает e (R×K)."""
    R = len(roots)
    let = np.zeros(R, np.int8); th = roots.copy(); owner = np.arange(R)
    E = np.zeros((R, K)); rng = np.random.default_rng(7)
    for k in range(K):
        if kind == 'random':
            coin = rng.random(len(let)) < 0.5
        else:
            coin = th[:, 0] < 0.5
        dev = np.bincount(owner, weights=coin - 0.5, minlength=R)
        cnt = np.bincount(owner, minlength=R)
        E[:, k] = dev / np.sqrt(cnt / 4)
        # дети
        nchild = np.array([len(r) for r in RULE])[let]
        par = np.repeat(np.arange(len(let)), nchild)
        j = np.concatenate([np.arange(n) for n in range(1, 4)])  # заглушка, заменяется ниже
        starts = np.cumsum(nchild) - nchild
        j = np.arange(len(par)) - np.repeat(starts, nchild)
        newlet = np.array([RULE[l][jj] for l, jj in zip(let[par], j)], np.int8) if len(par) < 2000 else \
                 np.choose(j, [np.array([r[0] for r in RULE])[let[par]],
                               np.array([r[1] for r in RULE])[let[par]],
                               np.array([r[2] if len(r) > 2 else 0 for r in RULE])[let[par]]])
        t = th[par]
        if kind.startswith('cat'):
            eps = float(kind[3:] or 0)            # нелинейная «кошка»: T(x,y) = (2x+y+p, x+y+p), p = ε/2π·sin 2πx
            t = (t + j[:, None] * OM) % 1.0
            p = eps / (2 * math.pi) * np.sin(2 * math.pi * t[:, 0])
            t = np.stack([2 * t[:, 0] + t[:, 1] + p, t[:, 0] + t[:, 1] + p], 1)
        elif kind == 'rot':                      # контроль без хаоса: только поворот
            t = t + OM + j[:, None] * OM[::-1]
        th = t % 1.0; let = newlet; owner = owner[par]
    return E

if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 14; R = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    roots = np.random.default_rng(3).random((R, 2))
    for kind in (sys.argv[3].split(',') if len(sys.argv) > 3 else ('random', 'cat', 'cat0.3', 'cat0.6', 'rot')):
        E = np.concatenate([tree(roots[b:b + 20], K, kind) for b in range(0, R, 20)])
        v = E.var(0)
        ks = np.arange(4, K)
        c1 = np.mean([np.corrcoef(E[:, k], E[:, k + 1])[0, 1] for k in ks[:-1]])
        c2 = np.mean([np.corrcoef(E[:, k], E[:, k + 2])[0, 1] for k in ks[:-2]])
        kurt = np.mean([((E[:, k] - E[:, k].mean()) ** 4).mean() / E[:, k].var() ** 2 for k in ks[-4:]])
        sl = np.polyfit(ks, np.log(v[ks] + 1e-30), 1)[0] / math.log(TAU ** 2)
        print(f'{kind:7s}: var(e_k) по поколениям {np.round(v[4::2], 2)}  тренд V^{sl:+.3f} (0 ⇔ h = 1/2); '
              f'корр. соседних поколений {c1:+.3f}, через одно {c2:+.3f}; эксцесс {kurt:.2f} (гаусс 3)')
        np.save(f'/home/claude/data/catcoin_{kind}.npy', E)
        sys.stdout.flush()
