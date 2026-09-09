import numpy as np

def frac_overlap(fr, R=1.0):
    """Доля общего объёма двух шаров радиуса R с центрами на расстоянии D = fr*R."""
    D = fr*R
    if D >= 2*R: return 0.0
    Vint = np.pi/12*(2*R - D)**2*(D + 4*R)
    Vball = 4/3*np.pi*R**3
    return Vint/Vball

print('ПЕРЕСЕЧЕНИЕ ДВУХ ПРОШЛЫХ КОНУСОВ')
print(f"{'расстояние':>18}{'доля общего':>14}")
for fr in (0.0, 0.2, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0):
    print(f'{f"{fr:.1f} R":>18}{frac_overlap(fr):>14.3f}')
print()
print('НАШИ ЗНАЧЕНИЯ РАЗЛИЧИМОСТИ И СООТВЕТСТВУЮЩЕЕ РАССТОЯНИЕ:')
for nm, val in (('d=2: 0.77', 0.77), ('d=3: 0.59', 0.59), ('d=4: 0.48', 0.48)):
    lo, hi = 0.0, 2.0
    for _ in range(60):
        m = (lo+hi)/2
        if frac_overlap(m) > val: lo = m
        else: hi = m
    print(f'  {nm}  ->  расстояние {lo:.3f} радиуса')
