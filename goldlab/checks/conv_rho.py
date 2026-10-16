"""Сходимость по плотности при ФИКСИРОВАННОМ физическом атоме (M = 1/τ² в единицах ρ = 1, α = 0.6, R = 11, окно 20):
посыпка при ρ = 1, 2, 4 и кристалл при ρ = 2, 4. Уровень и спиновый сдвиг; эталон 0.298 / +0.0144."""
import sys, time, json, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
Af.AL=0.6; M=0.381966; R=11.0; H=20.0; rho=float(sys.argv[1]); kind=sys.argv[2]
T0=time.time(); X=Af.make_world(rho,H,R,poisson_seed=1 if kind=='poisson' else None); at=Af.Atom(X,rho,M,H,R); e=np.array([1,0])
s1=at.respond(at.B(M))@e; E1=at.peak(s1); Ed=at.peak(at.respond(at.B(M,pauli=1.0))@e)
res=dict(kind=kind,rho=rho,n=at.n,E=E1,spin=Ed-E1,growth=at.growth(s1),sec=time.time()-T0); print(json.dumps(res),flush=True)
open('conv_rho.jsonl','a').write(json.dumps(res)+'\n')
