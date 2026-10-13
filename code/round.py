"""Гранёный против круглого: растёт ли k при закреплённой плотности.

Правило проекта есть AND по k шкалам — конъюнкция, и единогласие
единственный работающий порог (теорема о циклах). Ящик это AND по
КОНЕЧНОМУ списку направлений, световой конус — AND по КОНТИНУУМУ. Оба
выпуклые конусы; разница в числе граней.

Если спектральный различитель мерит именно «круглость», то при
ЗАКРЕПЛЁННОЙ плотности рост k должен вести спектр к конусному +0.0145.
Прежний скан шёл до k=6 и с плавающей плотностью, так что не проверено.

Плотность держится подбором spread по бисекции.
"""
import math, numpy as np
from core import mm_dim, height, spectral, frac_pairs, links_of
from refgrow import grow_ref, jacc_inc

TARGET, N = 0.10, 800

def dens(kref, sp, seeds=2):
    v=[]
    for t in range(seeds):
        rs=np.random.default_rng(181000+977*t)
        R=grow_ref(N,kref,rs,sp,0.0); v.append(frac_pairs(R)); del R
    return float(np.mean(v))

def calib(kref):
    lo, hi = 0.3, 6.0
    for _ in range(14):
        mid=0.5*(lo+hi)
        if dens(kref, mid) < TARGET: lo=mid
        else: hi=mid
    return 0.5*(lo+hi)

S=10
print(f'N={N}, плотность подогнана к {TARGET} подбором spread, '
      f'засевов {S}')
print('ЦЕЛЬ конуса при N=800: спектр +0.0145 ± 0.0026, разм. 4.03, '
      'высота ~9.5, J 0.169')
print()
print(f"{'k':>4}{'spread':>8}{'доля':>9}{'разм.':>7}{'высота':>7}"
      f"{'<спектр>':>11}{'с.к.о.':>9}{'связ':>7}{'J':>8}")
for kref in (4, 6, 8, 12, 18, 26):
    sp = calib(kref)
    v,f,d,h,l,j=[],[],[],[],[],[]
    for t in range(S):
        rs=np.random.default_rng(191000+977*t)
        R=grow_ref(N,kref,rs,sp,0.0); ff=frac_pairs(R)
        v.append(spectral(R)); f.append(ff); d.append(mm_dim(ff))
        h.append(height(R)[0]); l.append(links_of(R).sum()/N)
        j.append(jacc_inc(R)); del R
    v=np.array(v)
    print(f'{kref:>4}{sp:>8.3f}{np.mean(f):>9.4f}{np.nanmean(d):>7.2f}'
          f'{np.mean(h):>7.1f}{v.mean():>+11.4f}{v.std():>9.4f}'
          f'{np.mean(l):>7.1f}{np.nanmean(j):>8.4f}', flush=True)
