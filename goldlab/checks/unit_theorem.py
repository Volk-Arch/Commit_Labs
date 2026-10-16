# Проверка: ε = τ + √τ — единица нормы 1 поля Q(√τ); образы: буст, обратное сжатие, поворот (|·| = 1). Нет ли меньшей такой единицы?
import numpy as np, itertools
u=np.roots([1,0,-1,0,-1])                          # корни u⁴ − u² − 1
re=sorted([x.real for x in u if abs(x.imag)<1e-12],reverse=True); cx=[x for x in u if x.imag>1e-12][0]
emb=[re[0],re[1],cx]                               # два вещественных места и одно комплексное
f=lambda c,x: c[0]+c[1]*x+c[2]*x**2+c[3]*x**3
eps=(0,1,1,0); im=[f(eps,x) for x in emb]
print('ε образы: %.10f, %.10f, комплексный %s (|·| = %.12f, угол %.6f°)'%(im[0].real,im[1].real,np.round(im[2],6),abs(im[2]),np.degrees(np.angle(im[2]))))
print('норма N(ε) = %.12f; ch η = %.10f (τ = %.10f)'%((im[0]*im[1]*abs(im[2])**2).real,(im[0]+1/im[0]).real/2,(1+5**.5)/2))
best=[]
R=range(-12,13)
for c in itertools.product(R,R,R,R):
    v=[f(c,x) for x in emb]; N=(v[0]*v[1]*abs(v[2])**2).real
    if abs(abs(N)-1)<1e-9 and abs(abs(v[2])-1)<1e-9 and 1+1e-9<abs(v[0].real)<im[0].real*24.2:
        best.append((abs(v[0].real),c,round(np.degrees(np.angle(v[2])),3)))
best=sorted(set(best))[:6]; print('единицы с |комплексный образ| = 1 и 1 < e^η (коэфф. до ±12):',best)
