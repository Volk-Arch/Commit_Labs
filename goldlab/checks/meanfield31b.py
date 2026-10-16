"""Слои по числу событий в интервале: Φ(x,y) = c·Σ w_n [n_xy = n]. Средняя по посыпке: h(s) = Σ w_n sⁿe^{−s}/n!, s = ρV.
Условия (выведены): Σw_n Γ(n+½)/n! = √π (нормировка), Σw_n = 0 (нет хвоста 1/t ⇒ нет логарифма ⇒ нет мнимой части на массовой
оболочке), Σw_n Γ(n+3/2)/n! = 0 (следующий порядок). Связи: (1). Два слоя: (2, −2). Три слоя: (3, −7, 4)."""
import numpy as np, math
from meanfield31 import delta, root
for name,w in (('связи (1)',[1]),('два слоя (2, −2)',[2,-2]),('три слоя (3, −7, 4)',[3,-7,4])):
    h=lambda s,w=w: sum(wn*s**n*np.exp(-s)/math.factorial(n) for n,wn in enumerate(w))
    for rho in (1.0,4.0):
        t,d=delta(h,rho,T=12.0,N=3000); tail=d[-200:].mean()*t[-200:].mean()
        s='%s ρ=%g: хвост t·δ(t) = %+.3f; '%(name,rho,tail)
        for m in (0.5,1.0,1.41):
            ww=root(t,d,m); s+='m=%.2f: ω=%.3f%+.3fi; '%(m,ww.real,ww.imag)
        print(s,flush=True)
