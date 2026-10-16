import sys; sys.path.insert(0,'/home/claude/code/goldlab_checks')
import numpy as np
from bd_mean import *
oms=np.array([0.1,0.2,0.382,0.5,0.7,1.0])
def report(nm,c,P):
    F0,Fs=Bhat(c,P,np.r_[0.05,oms],T=80,N=8000); B=Fs-F0; norm=-(B[0].real)/0.05**2; B=B[1:]/norm
    print('%-28s'%nm,' '.join('%.2f:%+.3f%+.3fi'%(w,(b.real+w*w)/(w*w),b.imag/(w*w)) for w,b in zip(oms,B)),flush=True)
P=[Pk_poisson(k) for k in range(8)]
print('столбцы: ω: относит. ошибка Re (B̂+ω²)/ω², Im/ω² (Im<0 — затухание)')
for nk in (4,5,6,7):
    c=design(P,nk); report('Пуассон, %d слоёв c=%s'%(nk,np.round(c,1)),c,P)
