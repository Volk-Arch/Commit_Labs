import sys, json, time; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
Af.AL=float(sys.argv[1]); Af.A=float(sys.argv[2]); Af.SHAPE=int(sys.argv[3]); Af.BETA=float(sys.argv[4]); kind=sys.argv[5]; M=0.381966; R=11.0; H=20.0
if kind=='ref':
    r=Af.cont_refs(M,R,H); print('эталон AL=%g a=%g:'%(Af.AL,Af.A),json.dumps({k:round(v,5) for k,v in r.items()}),flush=True)
else:
    T0=time.time(); X=Af.make_world(1.0,H,R,poisson_seed=1 if kind=='poisson' else None); at=Af.Atom(X,1.0,M,H,R)
    print('%s AL=%g a=%g форма %d β=%g:'%(kind,Af.AL,Af.A,Af.SHAPE,Af.BETA),json.dumps({k:round(v,5) for k,v in at.measure().items()}),'%.0f c'%(time.time()-T0),flush=True)
