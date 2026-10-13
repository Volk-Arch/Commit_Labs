"""Два вопроса: покрытие цепью, и действие по модулю 8."""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from frames import diamond, rel_of, observer, radar

print('=== ВОПРОС 1: сколько мира покрывает цепь ===')
print('радарные координаты есть у события, если у него ЕСТЬ и прошлое,')
print('и будущее на цепи. Покрытие = доля таких событий.')
print()
N, T = 3000, 1.0
print(f"{'длина цепи (доля T)':>22}{'событий цепи':>14}{'покрытие':>11}"
      f"{'ожидание':>26}")
for frac in (0.25, 0.5, 0.75, 0.95, 1.0):
    cov = []
    for s in range(3):
        rs = np.random.default_rng(800+17*s)
        bg = diamond(N, T, rs)
        half = (1.0 - frac) / 2.0 * T + 0.02
        dens = max(8, int(120 * frac))
        Z = observer(dens, T, 0.0, 0.0, half=half)
        X = np.vstack([bg, Z])
        _, _, ok = radar(X, N, Z)
        cov.append(ok.mean())
    note = 'цепь во весь мир' if frac >= 1.0 else ''
    print(f'{frac:>22.2f}{dens:>14}{np.mean(cov):>11.4f}{note:>26}')
print()
print('  покрытие < 1 даже при цепи во весь мир => система отсчёта')
print('  накрывает лишь причинный ромб СВОЕЙ цепи, а не весь мир')

print()
print('=== ВОПРОС 2: действие по модулю 8 ===')
print('B = N - n0 + 9n1 - 16n2 + 8n3, и mod 8: 9=1, 16=0, 8=0')
print('значит B = N - n0 + n1 (mod 8) ТОЧНО. Сверяю численно.')
print()
W = (-1.0, 9.0, -16.0, 8.0)
bad = 0; vals = []
for s in range(40):
    n = int(np.random.default_rng(s).integers(120, 420))
    X = diamond(n, 1.0, np.random.default_rng(2000+s))
    R = rel_of(X); A = R.astype(np.float32)
    c = np.bincount((A@A)[R].astype(np.int64), minlength=4)
    B = n + sum(int(W[m])*int(c[m]) for m in range(4))
    cong = (n - int(c[0]) + int(c[1])) % 8
    if B % 8 != cong: bad += 1
    vals.append(B % 8)
print(f'  сверено 40 миров, расхождений с B = N - n0 + n1 (mod 8): {bad}')
h = np.bincount(np.array(vals), minlength=8)
print(f'  распределение остатка B mod 8: {list(h)}')
chi = ((h - len(vals)/8.0)**2 / (len(vals)/8.0)).sum()
print(f'  хи-квадрат против равномерного: {chi:.1f} при 7 степенях '
      f'(порог 5% = 14.1)')
print()
print('  остаток ТОЧЕН и не требует точности в S => амплитуда вида')
print('  exp(2*pi*i*B/8) считается без шума, и стена §15 её не касается')
