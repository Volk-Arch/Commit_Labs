"""Действие как фильтр на выросший порядок: падает или растёт.

Новый критерий: <S>/N обязано падать как N^(-1/4) (конус даёт -0.222).
Ящик даёт +0.414. Выросшие порядки по спектру сидели МЕЖДУ ними —
посмотрим, что скажет действие, которое различает их в 10 раз.
"""
import math, numpy as np
from core import frac_pairs
from weight import w_of
from refgrow import grow_ref
import growth2
from growth2 import grow

def smeared_S(R, eps):
    A=R.astype(np.float32); btw=np.rint(A@A).astype(np.int32)
    wt=w_of(np.arange(len(R)+2), eps)
    return float(eps*len(R)+wt[np.clip(btw,0,len(R)+1)][R].sum())

EPS, S = 0.2, 5
NS = (400, 800, 1600, 3200)
print(f'eps={EPS}, засевов {S}')
print('эталоны: конус -0.222, ящик +0.414')
print()
print(f"{'правило':>16}{'N':>7}{'доля':>9}{'<S>/N':>12}{'ошибка':>10}")
def scan(lbl, fn):
    vals=[]
    for N in NS:
        acc=[]
        for t in range(S):
            rs=np.random.default_rng(701000+977*t)
            R=fn(N, rs)
            acc.append((frac_pairs(R), smeared_S(R,EPS)/N)); del R
        A=np.array(acc,float); m=A.mean(0); e=A.std(0)/math.sqrt(S)
        vals.append(m[1])
        print(f'{lbl:>16}{N:>7}{m[0]:>9.4f}{m[1]:>12.5f}{e[1]:>10.5f}',
              flush=True)
    p=np.polyfit(np.log(NS), np.log(np.abs(vals)),1)[0]
    print(f'{"показатель:":>24} {p:+.3f}')
    print()

scan('refgrow k=6', lambda N, rs: grow_ref(N, 6, rs, 1.0, 0.0))
def descg(N, rs):
    growth2._ALPHA=1.0
    return grow(N, 'alpha', 0.05, 0.0, 0.2, 8, rs)[0]
scan('рост по потомкам', descg)
