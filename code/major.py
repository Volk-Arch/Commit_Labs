"""Что скажет действие о правиле БОЛЬШИНСТВА вместо единогласия.

Теорема: при m < k появляются циклы, но лишь выше критического размера
N_крит = k^(1/k)*(k!)^((k-1)/k). При k=7,m=6 это 1969, значит ниже мир
АЦИКЛИЧЕН и правило большинства там — законный частичный порядок.

Новый различитель: <S>/N обязано падать как N^(-1/4). Конус даёт -0.222,
ящик (единогласие k=4) +0.414. Вопрос: что даёт большинство?

Попутно проверяется, не зависит ли сам показатель от ПЛОТНОСТИ — иначе
сравнение нечестно. Для этого единогласие берётся при k=3,4,5, у которых
доля пар сильно разная.
"""
import math, numpy as np
from core import frac_pairs
from weight import w_of

def smeared_S(R, eps):
    A=R.astype(np.float32); btw=np.rint(A@A).astype(np.int32)
    wt=w_of(np.arange(len(R)+2), eps)
    return float(eps*len(R)+wt[np.clip(btw,0,len(R)+1)][R].sum())

def thresh_rel(V, m):
    """R[i,j] = v(j) < v(i) хотя бы по m шкалам из k."""
    N, k = V.shape
    C = np.zeros((N,N), np.int16)
    for s in range(k):
        C += (V[None,:,s] < V[:,None,s]).astype(np.int16)
    R = C >= m
    np.fill_diagonal(R, False)
    return np.asarray(R, bool)

def has_cycle(R):
    """Топологическая сортировка (Кан): есть ли цикл."""
    N=len(R); indeg = R.sum(1).astype(np.int64)   # сколько предшественников
    left = N; q=[int(i) for i in np.flatnonzero(indeg==0)]
    seen = np.zeros(N, bool)
    for i in q: seen[i]=True
    cnt=0
    while q:
        i=q.pop(); cnt+=1
        succ=np.flatnonzero(R[:,i])
        for j in succ:
            indeg[j]-=1
            if indeg[j]==0 and not seen[j]:
                seen[j]=True; q.append(int(j))
    return cnt < N

EPS, S = 0.2, 4
NS = (200, 400, 800, 1600)
print(f'eps={EPS}, засевов {S}')
print('эталоны показателя: конус -0.222 (к нулю), ящик k=4 +0.414')
print()
print(f"{'правило':>18}{'N':>7}{'доля':>9}{'циклы':>8}{'<S>/N':>12}"
      f"{'ошибка':>10}")
for lbl, k, m in (('единогласие k=3', 3, 3), ('единогласие k=4', 4, 4),
                  ('единогласие k=5', 5, 5), ('большинство 6/7', 7, 6),
                  ('большинство 4/5', 5, 4)):
    vals=[]
    for N in NS:
        acc=[]; cyc=0
        for t in range(S):
            rs=np.random.default_rng(801000+977*t)
            R=thresh_rel(rs.random((N,k)), m)
            cyc += int(has_cycle(R))
            acc.append((frac_pairs(R), smeared_S(R,EPS)/N)); del R
        A=np.array(acc,float); mm=A.mean(0); e=A.std(0)/math.sqrt(S)
        vals.append(mm[1])
        print(f'{lbl:>18}{N:>7}{mm[0]:>9.4f}{f"{cyc}/{S}":>8}'
              f'{mm[1]:>12.5f}{e[1]:>10.5f}', flush=True)
    p=np.polyfit(np.log(NS), np.log(np.abs(vals)),1)[0]
    print(f'{"показатель:":>26} {p:+.3f}')
    print()
