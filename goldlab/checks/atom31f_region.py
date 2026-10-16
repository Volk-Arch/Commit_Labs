import sys, json, time, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
Af.AL=0.6; M=0.381966; H=20.0; R=float(sys.argv[1]); kind=sys.argv[2]
if kind=='ref':
    E1=Af._radial(M,R,H); Ed=Af._radial(M,R,H,pauli=1.0); Et=Af._radial(M,R,H,trunc=True); Edt=Af._radial(M,R,H,pauli=1.0,trunc=True)
    print('эталон R=%g: E %.4f спин %+.5f | с обрезкой: E %.4f спин %+.5f'%(R,E1,Ed-E1,Et,Edt-Et),flush=True)
else:
    T0=time.time(); X=Af.make_world(1.0,H,R,poisson_seed=1 if kind=='poisson' else None); at=Af.Atom(X,1.0,M,H,R); e=np.array([1,0])
    E1=at.peak(at.respond(at.B(M))@e); Ed=at.peak(at.respond(at.B(M,pauli=1.0))@e)
    print('%s R=%g (n=%d): E %.4f спин %+.5f  %.0f c'%(kind,R,at.n,E1,Ed-E1,time.time()-T0),flush=True)
