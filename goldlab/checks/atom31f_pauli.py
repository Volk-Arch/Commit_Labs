"""Паули и «гелий» в атоме 3+1 (m = 1/τ², α = 0.6): кристалл без сдвига. J — со сдвигом уровня от облака: точное поле и поле
ИЗ ПОРЯДКА тем же размытым конусом (фотон = тот же конус)."""
import sys, json, time, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
Af.AL=0.6; M=0.381966; R=11.0; H=20.0; T0=time.time()
X=Af.make_world(1.0,H,R); at=Af.Atom(X,1.0,M,H,R)
pa=at.pauli(); print('Паули:',json.dumps(pa),'%.0f c'%(time.time()-T0),flush=True)
F,rr,Uc=Af.cloud_field_from_order(M,1.0); print('поле облака из порядка / точное:',' '.join('r=%.1f: %.3f'%(a,b/d) for a,b,_,d in F),'%.0f c'%(time.time()-T0),flush=True)
last=F[-1,1]/F[-1,3]; Uord=lambda q: np.where(q<=F[-1,0],np.interp(q,F[:,0],F[:,1]),np.interp(q,rr,Uc)*last)
e=np.array([1,0]); B=at.B(M)
E0=at.peak(at.respond(B)@e); Ee=at.peak(at.respond(B,Ug=at.Unuc+np.interp(at.rg,rr,Uc))@e); Eo=at.peak(at.respond(B,Ug=at.Unuc+Uord(at.rg))@e)
res=dict(pauli=pa,field_ratio=[float(b/d) for a,b,_,d in F],E0=E0,J_exact_field=Ee-E0,J_order_field=Eo-E0)
print(json.dumps(res),'%.0f c'%(time.time()-T0),flush=True); json.dump(res,open('atom31f_pauli_al0.6.json','w'))
