"""Атом с более лёгкой частицей: m = 1/τ при ρ = 1 — по точному самоподобию это то же, что m = 1 при ρ = τ⁴ ≈ 6.85.
Кристалл против посыпки против непрерывной теории тем же прибором: уходит ли «тугая масса» (гребёнка) для лёгкой частицы?"""
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import fuzzy as Fz
TAU=(1+5**.5)/2; m=float(sys.argv[1]) if len(sys.argv)>1 else 1/TAU; rho=1.0; R=float(sys.argv[2]) if len(sys.argv)>2 else 11.0; H=float(sys.argv[3]) if len(sys.argv)>3 else 16.0
m2s=(m*m, 0.81*m*m)
src=open('instr_cont.py').read().split('E0=peak')[0].replace('ball=r<7','ball=r<%g'%R).replace('T=19.8','T=%g'%(H-0.2)).replace('ts<19.7','ts<%g'%(H-0.3)).replace('om=np.linspace(0.6,1.3,1401)','om=np.linspace(0.3,1.3,2001)')
g={}; exec(src,g)
E1=g['peak'](*g['evolve'](m2=m2s[0])); E2=g['peak'](*g['evolve'](m2=m2s[1])); Es=g['peak'](*g['evolve'](m2=m2s[0],cz=0.12/(2/np.e),Lz=2.0))
REF=dict(E=E1,slope=(E1-E2)/(m2s[0]-m2s[1]),stark=Es-E1,spin=float('nan'))
print('непрерывная тем же прибором: m=%.3f: 1s %.4f, наклон %.3f, второй порядок %+.5f'%(m,E1,REF['slope'],REF['stark']),flush=True)
mb=(Fz.bare_mass2(rho,0.115,m),Fz.bare_mass2(rho,0.115,0.9*m))
for kind in ('кристалл','посыпка'):
    r=Fz.measure(rho,H,R,50.0,REF,beta=0.115,mb2=mb,shift='poisson' if kind=='посыпка' else None,label=None)
    # measure считает наклон по разности 0.19 в m²; пересчёт на нашу разность масс
    sl=r['slope']*0.19/(m2s[0]-m2s[1])
    print('%s: 1s %.4f (%.4f), рост %+.3f, наклон %.3f (%.3f), второй порядок %+.5f (%+.5f), спин %+.5f'%(kind,r['E'],REF['E'],r['growth'],sl,REF['slope'],r['stark'],REF['stark'],r['spin']),flush=True)
