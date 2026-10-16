import sys, itertools, math, numpy as np
sys.path.insert(0,'/home/claude/code'); sys.argv=sys.argv[:1]
from renou import BELL, local_search, Z, A_opt, C_opt, phip, score
from scipy.optimize import minimize
r2=math.sqrt(2)
LO={'p+':BELL['p+'],'p-':BELL['p-'],'00':np.array([[1,0],[0,0]],complex),'11':np.array([[0,0],[0,1]],complex)}
# плотный набор наблюдаемых на сфере Блоха
X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]])
obs=[]
for th in np.linspace(0,math.pi,25):
    for ph in np.linspace(0,2*math.pi,48,endpoint=False):
        n=(math.sin(th)*math.cos(ph),math.sin(th)*math.sin(ph),math.cos(th)); obs.append(n[0]*X+n[1]*Y+n[2]*Z)
def bestT(bobset, psi=phip, phi=phip, sets=(obs,obs)):
    return max(local_search(psi,phi,[bobset[k] for k in pm],sets[0],sets[1],restarts=6)[0] for pm in itertools.permutations(bobset))
print('идеальный Белл, плотные настройки:', round(bestT(BELL),3))
print('линейная оптика (ψ+, ψ−, 00, 11), Φ+ на обоих источниках:', round(bestT(LO),3))
# оптимизация по состояниям тоже: локальные унитарные на Бобовой стороне каждой пары
rs=np.random.default_rng(0); best=0
def U(v):
    a,b,c=v; return np.array([[math.cos(a)*np.exp(1j*b), -math.sin(a)*np.exp(-1j*c)],[math.sin(a)*np.exp(1j*c), math.cos(a)*np.exp(-1j*b)]])
for t in range(40):
    v=rs.uniform(0,2*math.pi,6); psi=phip@U(v[:3]).T; phi=U(v[3:]).T@phip if False else U(v[3:])@phip
    best=max(best,bestT(LO,psi,phi))
print('линейная оптика, + случайные локальные повороты состояний (40):', round(best,3))
