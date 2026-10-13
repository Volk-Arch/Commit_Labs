"""РЕШАЮЩИЙ КОНТРОЛЬ: не выродилось ли сглаженное действие в счёт отношений.

При малом eps разложение даёт f(n,eps) = 1 - 9*n*eps + ..., значит
S = N*alpha + beta*(число отношений) - 9*beta*eps*(сумма n) + ...
То есть при eps -> 0 действие вырождается в СЧЁТ ОТНОШЕНИЙ, у которого
малый относительный разброс и сильный отклик на дырку. Тогда dS/sigma > 1
получается тривиально и про кривизну не говорит ничего.

Проверка: корреляция S с числом отношений, и остаток после вычитания
линейной зависимости от него. Если остаток ничего не несёт, окно пустое.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from frames import diamond, rel_of
from scipy.stats import binom

C = np.array([1.0,-9.0,16.0,-8.0])
ALPHA, BETA = -(6.0)**0.25, 2.0/math.gamma(1.5)*(math.pi/6.0)**0.5

def fw(nmax, e):
    n = np.arange(nmax+1)[:,None]; m = np.arange(4)[None,:]
    return (binom.pmf(m,n,e)*C[None,:]).sum(1)

def both(X, eps, rad=0.0):
    if rad > 0:
        X = X[np.linalg.norm(X[:,1:],axis=1) >= rad]
    R = rel_of(X); A = R.astype(np.float32)
    v = (A@A)[R].astype(np.int64)
    h = np.bincount(v, minlength=int(v.max())+1)
    nrel = int(R.sum())
    return [float(len(X)*ALPHA + BETA*(h*fw(len(h)-1,e)).sum())
            for e in eps], nrel, len(X)

N, SEEDS, RAD = 1200, 30, 0.18
eps = [1.0, 0.35, 0.10, 0.05]
F, H, NF, NH = [], [], [], []
for s in range(SEEDS):
    X = diamond(N, 1.0, np.random.default_rng(87000+173*s))
    a, nf, _ = both(X, eps); b, nh, _ = both(X, eps, RAD)
    F.append(a); H.append(b); NF.append(nf); NH.append(nh)
    del X
F=np.array(F); H=np.array(H); NF=np.array(NF,float); NH=np.array(NH,float)
print(f'N={N}, засевов {SEEDS}, дырка {RAD}')
print(f"{'eps':>7}{'корр S с n_отн':>16}{'σ(S)':>10}{'σ остатка':>12}"
      f"{'ΔS':>9}{'Δ остатка':>11}{'Δост/σост':>11}{'вывод':>22}")
for k,e in enumerate(eps):
    r = np.corrcoef(F[:,k], NF)[0,1]
    # остаток после вычитания линейной зависимости от числа отношений,
    # подгонка на ПЛОСКИХ мирах, применяется к обоим
    A = np.column_stack([NF, np.ones(SEEDS)])
    sol,*_ = np.linalg.lstsq(A, F[:,k], rcond=None)
    resF = F[:,k] - (NF*sol[0]+sol[1])
    resH = H[:,k] - (NH*sol[0]+sol[1])
    sg = resF.std(ddof=1)
    d = abs((resH-resF).mean())
    rat = d/max(sg,1e-12)
    note = 'несёт своё' if rat > 1 else 'ВЫРОДИЛОСЬ'
    print(f'{e:>7.2f}{r:>16.5f}{F[:,k].std(ddof=1):>10.1f}{sg:>12.1f}'
          f'{abs((H[:,k]-F[:,k]).mean()):>9.1f}{d:>11.1f}{rat:>11.3f}'
          f'{note:>22}')
print()
print('  корреляция близка к 1 и отношение падает => действие стало')
print('  счётом отношений, и окно по hbar пустое')
