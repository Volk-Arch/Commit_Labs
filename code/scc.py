"""Циклы как объекты: локализованы ли сильно связные компоненты.

Если ослабить единогласие до m = k-1, выше критического размера
появляются циклы. Точный объект — СИЛЬНО СВЯЗНАЯ КОМПОНЕНТА: максимальный
набор событий, порядок между которыми неразрешим. Нетривиальная
компонента есть сгусток неопределённости, возникший ИЗ ПРАВИЛА, без
добавленной структуры.

Если компоненты малы и компактны — это кандидаты в частицы.
Если размазаны по миру — шум, ветка закрыта.

КОНТРОЛЬ: протяжённость компоненты против случайного набора событий того
же размера.
"""
import math, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from core import frac_pairs

def thresh_rel(V, m):
    N, k = V.shape
    C = np.zeros((N,N), np.int16)
    for s in range(k):
        C += (V[None,:,s] < V[:,None,s]).astype(np.int16)
    R = C >= m
    np.fill_diagonal(R, False)
    return np.asarray(R, bool)

def extent(V, idx):
    """Протяжённость набора: средний разброс по шкалам (нормирован)."""
    P = V[idx]
    return float(np.mean(P.std(0)))

S = 4
print('k=7, m=6. N_крит по формуле 1969.')
print('протяжённость — средний разброс по шкалам; у случайного '
      'набора она максимальна (~0.289 для равномерных)')
print()
print(f"{'N':>7}{'доля':>8}{'компонент>1':>13}{'в них событий':>14}"
      f"{'макс размер':>12}{'<протяж>':>11}{'контроль':>10}{'отношение':>11}")
for N in (1600, 2400, 3200, 4800):
    rows=[]
    for t in range(S):
        rs=np.random.default_rng(901000+977*t)
        V=rs.random((N,7))
        R=thresh_rel(V, 6)
        n_comp, lab = connected_components(csr_matrix(R), directed=True,
                                           connection='strong')
        sizes=np.bincount(lab)
        big=np.flatnonzero(sizes>1)
        if len(big)==0:
            rows.append((frac_pairs(R), 0, 0, 0, np.nan, np.nan)); continue
        exts, ctrl, tot = [], [], 0
        for b in big:
            idx=np.flatnonzero(lab==b); tot+=len(idx)
            exts.append(extent(V, idx))
            ctrl.append(extent(V, rs.choice(N, len(idx), replace=False)))
        rows.append((frac_pairs(R), len(big), tot, int(sizes[big].max()),
                     float(np.mean(exts)), float(np.mean(ctrl))))
        del R
    A=np.array(rows,float); m=np.nanmean(A,0)
    r = m[4]/m[5] if m[5]>0 else np.nan
    print(f'{N:>7}{m[0]:>8.4f}{m[1]:>13.1f}{m[2]:>14.1f}'
          f'{m[3]:>12.1f}{m[4]:>11.4f}{m[5]:>10.4f}{r:>11.3f}', flush=True)
