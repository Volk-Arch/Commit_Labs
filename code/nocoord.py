"""ПУНКТ 2: Лоренц БЕЗ координат вовсе.

В §16 продольная координата x1 бралась как радиальное радарное расстояние
со ЗНАКОМ ИЗ КООРДИНАТ — единственное место, где координаты входили в
вывод. Отсюда постоянный промах -5 % по v.

Починка своим же инструментом проекта (`signed.py`): разность расстояний
до ДВУХ опор даёт u.x. Ставим две опорные цепи на +-d по оси движения;
тогда
    x1 = (rho_минус - rho_плюс) * d / (2d) ... с точностью до масштаба,
а масштаб уходит в коэффициент B подгонки. Координат в построении НЕТ.

Ожидание: A = gamma^2 как было, v ближе к tanh(rap), промах -5 % должен
уменьшиться.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from frames import diamond, observer, radar

N, DENS, SEEDS = 5000, 130, 4
D = 0.16      # разнос опор по оси 1

print(f'N={N}, цепей четыре: наблюдатель-1, наблюдатель-2, и ДВЕ ОПОРЫ')
print(f'опоры на +-{D} по оси движения; продольная координата — РАЗНОСТЬ')
print('радарных расстояний до них. Координат в построении НЕТ.')
print()
print(f"{'случай':>20}{'A':>9}{'gamma²':>9}{'промах':>9}"
      f"{'v':>9}{'ожид':>8}{'промах':>9}{'качество':>10}{'способ':>10}")
for rap in (0.3, 0.6, 0.9):
  for how in ('две опоры', 'знак(x) '):
    fa, fv, qq = [], [], []
    for s in range(SEEDS):
        rs = np.random.default_rng(85000+163*s)
        bg = diamond(N, 1.0, rs)
        Z1 = observer(DENS, 1.0, 0.0, 0.0)
        Z2 = observer(DENS, 1.0, rap, 0.0)
        Rp = observer(DENS, 1.0, 0.0, +D)
        Rm = observer(DENS, 1.0, 0.0, -D)
        X = np.vstack([bg, Z1, Z2, Rp, Rm])
        k = N
        i1 = np.arange(k, k+DENS); k += DENS
        i2 = np.arange(k, k+DENS); k += DENS
        ip = np.arange(k, k+DENS); k += DENS
        im = np.arange(k, k+DENS)
        t1, r1, o1 = radar(X, N, X[i1])
        t2, r2, o2 = radar(X, N, X[i2])
        _, rp, op = radar(X, N, X[ip])
        _, rm, om = radar(X, N, X[im])
        ok = o1 & o2 & op & om
        rr = np.linalg.norm(X[:N,1:], axis=1)
        if how == 'две опоры':
            x1 = (rm - rp) / 2.0          # чистый порядок
            m = ok
        else:
            cz = np.where(rr>1e-12, X[:N,1]/np.maximum(rr,1e-12), 0.0)
            x1 = r1 * np.sign(cz)
            m = ok & (np.abs(cz) > 0.97)
        if m.sum() < 60: continue
        M = np.column_stack([t1[m], x1[m], np.ones(int(m.sum()))])
        sol,*_ = np.linalg.lstsq(M, t2[m], rcond=None)
        fa.append(float(sol[0])); fv.append(-float(sol[1])/float(sol[0]))
        qq.append(np.corrcoef(M@sol, t2[m])[0,1])
        del X, bg
    if not fa: print(f'  rap={rap} {how}: мало данных'); continue
    g2 = math.cosh(rap)**2; vt = math.tanh(rap)
    A = np.mean(fa); V = np.mean(fv)
    # при двух опорах масштаб x1 иной, поэтому v сравниваем по форме:
    # отношение v(rap) к v(0.3) должно идти как tanh(rap)/tanh(0.3)
    print(f'{f"буст {rap}":>20}{A:>9.4f}{g2:>9.4f}'
          f'{100*(A-g2)/g2:>+9.2f}{V:>9.4f}{vt:>8.4f}'
          f'{100*(V-vt)/vt:>+9.2f}{np.mean(qq):>10.4f}{how:>10}')
print()
print('  A = gamma² обязано совпасть при ОБОИХ способах — это проверка,')
print('  что замена продольной координаты ничего не испортила.')
print('  v при двух опорах измерен в ИНЫХ единицах (масштаб опор), поэтому')
print('  сравнивать надо ФОРМУ: отношения v(rap)/v(0.3) против tanh.')
