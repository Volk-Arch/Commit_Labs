"""Массовый сектор размытого конуса S1 в средней теории: F(ω) = −1/ω² + G(ω) при k = 0; полюс 1 + m_b²F = 0.
Голая масса, дающая физическую m: m_b² = m²/(1 − m²·Re G(m)). Наклон: dω/dm² у кристалла и непрерывный."""
import numpy as np, math
from meanfield_sym import delta_sym
from meanfield31 import root
for rho in (1.0,2.0,4.0):
    t,d=delta_sym(rho); G=lambda w: np.trapezoid(np.exp(1j*w*t)*d,t)
    w1=root(t,d,1.0); w09=root(t,d,0.9); 
    mb2=lambda m: m*m/(1-m*m*G(m).real)
    print('ρ=%g: ω(m=1) = %.4f%+.4fi, ω(m=0.9) = %.4f; наклон (ω(1)−ω(0.9))/0.19 = %.3f (непр. %.3f); G(1) = %.4f%+.4fi'%(rho,w1.real,w1.imag,w09.real,(w1.real-w09.real)/0.19,(1-0.9)/0.19,G(1.0).real,G(1.0).imag))
    for m in (0.9,1.0):
        mb=math.sqrt(mb2(m)); w=root(t,d,mb); print('     m=%.1f: голая m_b² = %.4f → ω = %.4f%+.4fi'%(m,mb*mb,w.real,w.imag))
    print('     Z = dm_b²/dm² ≈ %.3f'%((mb2(1.0)-mb2(0.9))/0.19))
