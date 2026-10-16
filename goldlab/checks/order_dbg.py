import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab'); sys.path.insert(0,'/home/claude/code/goldlab_checks')
from goldlab import atom31f as Af, hinv as Hv, fuzzy as Fz
Af.AL=0.6; R,H=6.0,14.0; X=Af.make_world(1.0,H,R); X=X[np.argsort(X[:,0])]; N=len(X)
names=['S1','OSc','OSo']; K={k:np.load('/home/claude/scratch_kernels/gold_6_14_%d.npy'%i) for i,k in enumerate(names)}
Un=-Af.AL/np.sqrt(np.linalg.norm(X[:,1:],axis=1)**2+Af.A**2)
for M in [float(x) for x in sys.argv[1].split(',')]:
    at=Af.Atom(X,1.0,M,H,R); f=at.src()[:,0]
    for nm in names:
        KT=K[nm]; mb=at.mb2(M) if nm=='S1' else {0.382:0.1391,0.7:0.414}[M]
        for lab,b in (('свободно',-mb*np.ones(N)),('атом E=0.85M',-(mb+2*0.85*M*Un-Un*Un))):
            phi=np.zeros(N); s=f.real.copy()
            for j in range(N):
                phi[j]=KT[j,:j]@s[:j]; s[j]+=b[j]*phi[j]
            if not np.all(np.isfinite(phi)): print(M,nm,lab,'nan'); continue
            Y,_=Af.shell_channels(at,phi.astype(complex)); o,g,w=Hv.pencil(Y,0.5,8)
            amp=[np.max(np.abs(phi[(at.t>q)&(at.t<q+1)]),initial=0) for q in range(1,13,2)]
            print('M=%.3f %-4s %-13s моды:'%(M,nm,lab),' '.join('%.3f/%+.3f'%(a,b) for a,b,c in zip(o,g,w) if a>0 and c>0.02*w[0])[:90],' |φ|max:',' '.join('%.1e'%a for a in amp),flush=True)
    if M==0.382 or True:
        # S1 через respond_fast для сверки
        at.idx=Fz.build_index(at.X,2.0); B=np.ascontiguousarray((-at.mb2(M)*at.eye)/1.0,dtype=np.complex128)
        ph=Fz.respond_fast(at.X,at.idx,at.tc2,at.alpha,at.coef,at.rg,at.Unuc,at.src(),B,Af.BETA,Af.SHAPE,use_phase=0)[:,0]
        Y,_=Af.shell_channels(at,ph); o,g,w=Hv.pencil(Y,0.5,8); print('   сверка respond_fast S1 свободно:',' '.join('%.3f/%+.3f'%(a,b) for a,b,c in zip(o,g,w) if a>0 and c>0.02*w[0])[:90])
