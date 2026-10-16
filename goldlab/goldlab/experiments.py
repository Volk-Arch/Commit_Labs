"""РЕЕСТР ОПЫТОВ. Каждый опыт — запись: блок, название, «что это» простыми словами, параметры (с пресетами «быстро» и
«как в документе»), эталон со ссылкой на документ и допуском, функция run(p, region) → результат:
  numbers: [{key, label, value, expected, tol, status}]   series: [{title, xlabel, lines: [{name, x, y, style}]}]
  overlay: {t, x, v, label} (раскраска событий мира 1+1)   notes: [строки]
Статус числа: 'ok' (в допуске), 'near' (в тройном допуске), 'off'. Эталоны — из документов (docs/…)."""
import math, time
TAU_G = (1 + 5 ** 0.5) / 2
import numpy as np
from . import worlds as Wd, causal as Cs, propagate as Pr, fields as Fl, observe as Ob

REG = {}
def experiment(**meta):
    def deco(fn): REG[meta['id']] = {**meta, 'run': fn}; return fn
    return deco

def num(key, label, value, expected=None, tol=None, rel=False, fmt='{:.4f}'):
    st = None
    if expected is not None and tol is not None and value is not None and np.isfinite(value):
        dev = abs(value - expected) / (abs(expected) if rel else 1.0); st = 'ok' if dev <= tol else ('near' if dev <= 3 * tol else 'off')
    return {'key': key, 'label': label, 'value': None if value is None or not np.isfinite(value) else float(value),
            'expected': expected, 'tol': tol, 'rel': rel, 'status': st, 'fmt': fmt}

def overlay(world, v, label):
    v = np.asarray(v, float); return {'t': world.t.round(4).tolist(), 'x': world.x.round(4).tolist(), 'v': np.nan_to_num(v).round(5).tolist(), 'label': label}

def P(name, label, default, lo=None, hi=None, step=None, choices=None):
    return {'name': name, 'label': label, 'default': default, 'min': lo, 'max': hi, 'step': step, 'choices': choices}

# ======================= МИР, СВЯЗИ, СВЕТ, МАССА =======================
@experiment(id='world', block='Мир, свет, масса', title='Мир и его размерность',
    what='Строит кристалл (или посыпку) и считает события, связи и размерность Майрхейма–Мейера по доле упорядоченных пар в выделенном причинном интервале.',
    params=[P('kind', 'Мир', 'gold11', choices=['gold11', 'poisson11']), P('rho', 'Плотность ρ', 4.0, 1, 8, 1), P('L', 'Размер ромба L', 12.0, 6, 24, 1)],
    region='interval', presets={'быстро': {}, 'как в документе': {'L': 16}}, ref='golden-doc; ARTICLE (размерность 2)', cost='секунды')
def exp_world(p, region=None):
    w = Wd.make(p['kind'], rho=p['rho'], L=p['L']); I, J = Cs.links_dense(w.X)
    t0, t1, xc = (region['t0'], region['t1'], 0.5 * (region['x0'] + region['x1'])) if region else (-0.8 * p['L'], 0.8 * p['L'], 0.0)
    a, b = np.array([t0, xc]), np.array([t1, xc]); dA = w.X - a; dB = b - w.X
    ins = (dA[:, 0] > np.abs(dA[:, 1])) & (dB[:, 0] > np.abs(dB[:, 1]))
    d, f = Cs.mm_dimension(w.X[ins]) if ins.sum() > 10 else (float('nan'), float('nan'))
    deg = np.bincount(J, minlength=w.n)
    return {'numbers': [num('n', 'событий', w.n, fmt='{:.0f}'), num('links', 'связей в прошлое на событие (внутри)', deg[np.abs(w.t) < 0.5 * p['L']].mean(), fmt='{:.2f}'),
                        num('nint', 'событий в интервале', int(ins.sum()), fmt='{:.0f}'), num('dim', 'размерность Майрхейма–Мейера', d, 2.0, 0.05),
                        num('f', 'доля упорядоченных пар (2D: 0.25)', f, 0.25, 0.01)],
            'overlay': overlay(w, ins.astype(float), 'интервал'), 'series': [],
            'notes': ['Интервал — от нижней до верхней точки выделения на одной вертикали. Для 1+1 ожидается 2.']}

@experiment(id='dispersion', block='Мир, свет, масса', title='Закон дисперсии: инерция = масса',
    what='Широкий источник e^{ikx} возбуждает одну волну; частота отклика (спектроскопия) даёт E(k). Если √(E² − k²) одинаково при всех k — инертная масса равна массе покоя.',
    params=[P('m', 'Масса m', 1.0, 0.3, 1.5, 0.1), P('rho', 'ρ', 4.0, 2, 8, 1), P('L', 'Ромб L', 24.0, 16, 40, 2)],
    presets={'быстро': {}, 'как в документе': {'L': 36}}, ref='maxwell-doc §14', cost='~1 мин (L = 24)')
def exp_dispersion(p, region=None):
    w = Wd.gold11(p['rho'], p['L']); t, x = w.t, w.x; ts = -p['L'] + 12; zero = lambda s: 0 * s; m = p['m']
    om = np.linspace(0.2, 3.0, 2801); rows = []; lines = []
    for k in (0.0, 0.5, 1.0, 1.5):
        g = lambda s, k=k: np.exp(-s * s / 72) * np.exp(1j * k * s)
        phi = Pr.corner_respond(w.X, zero, zero, m, p['rho'], np.where(np.abs(t - ts) < 0.45, g(x), 0))
        sel = (t > ts + 2) & (t < p['L'] - 10) & (np.abs(x) < 10); S = Ob.spectrum(t[sel], np.conj(g(x[sel])) * phi[sel], om, 'hann')
        pk = max(Ob.peaks(om, S), key=lambda z: z[1]); rows.append((k, pk[0])); lines.append({'name': f'k = {k}', 'x': om[::4].tolist(), 'y': (S / S.max())[::4].tolist()})
    mass = [math.sqrt(max(E * E - k * k, 0)) for k, E in rows]
    nums = [num(f'E{k}', f'E при k = {k} (КГ {math.sqrt(m * m + k * k):.3f})', E, math.sqrt(m * m + k * k), 0.04, rel=True) for k, E in rows]
    nums += [num('mspread', '√(E² − k²): разброс по k (отн.)', np.std(mass) / np.mean(mass), 0.0, 0.04), num('mrest', '√(E² − k²), среднее', np.mean(mass), m, 0.05, rel=True)]
    return {'numbers': nums, 'series': [{'title': 'Спектры отклика', 'xlabel': 'E', 'lines': lines}], 'notes': ['Эталон (m = 1, L = 36): 1.041 / 1.029 / 1.046 / 0.955.']}

@experiment(id='clock', block='Мир, свет, масса', title='Часы массивной частицы',
    what='Источник «покоя» — полоса внизу выделенной области; поле в её центре колеблется с частотой ≈ m (пропагатор Джонстона).',
    params=[P('m', 'Масса m', 1.0, 0.3, 1.5, 0.1), P('rho', 'ρ', 4.0, 2, 8, 1), P('L', 'Ромб L', 12.0, 8, 18, 1)],
    region='interval', presets={'быстро': {}, 'как в документе': {}}, ref='mass-doc §2 (золото: 1.078 при m = 1)', cost='секунды')
def exp_clock(p, region=None):
    w = Wd.gold11(p['rho'], p['L']); t, x = w.t, w.x; m = p['m']
    t0, t1, xc = (region['t0'], region['t1'], 0.5 * (region['x0'] + region['x1'])) if region else (-0.67 * p['L'], 0.5 * p['L'], 0.0)
    f = ((np.abs(t - t0) < 0.45) & (np.abs(x - xc) < 0.4 * p['L'])).astype(float)
    phi = Pr.johnston11(w.X, p['rho'], m).T @ f
    c = (np.abs(x - xc) < 1.0) & (t > t0 + 0.6) & (t < t1); tt, yy = t[c] - t0, phi[c]
    best = (0, 0)
    for om in np.arange(0.05, 3.0, 0.005):
        A = np.c_[np.sin(om * tt), np.cos(om * tt), np.ones_like(tt)]; r = yy - A @ np.linalg.lstsq(A, yy, rcond=None)[0]
        ex = 1 - (r @ r) / ((yy - yy.mean()) @ (yy - yy.mean()))
        if ex > best[1]: best = (om, ex)
    o = np.argsort(tt)
    return {'numbers': [num('freq', 'частота часов', best[0], m, 0.1, rel=True), num('expl', 'объяснено синусом', best[1], fmt='{:.3f}')],
            'series': [{'title': 'Поле в центре против времени', 'xlabel': 'время от источника', 'lines': [{'name': 'φ', 'x': tt[o].tolist(), 'y': yy[o].tolist(), 'style': 'dots'}]}],
            'overlay': overlay(w, phi / max(np.abs(phi).max(), 1e-12), 'поле φ'), 'notes': ['Частота завышена на ~8 % из-за ширины источника (mass-doc).']}

@experiment(id='light31', block='Мир, свет, масса', title='Свет и масса в 3+1 (вакуум SJ)',
    what='Причинный ромб 3+1, пропагатор по связям, вакуум SJ. Медленные моды подгоняются плоской волной: ω = |k| у света и √(m² + k²) у массы.',
    params=[P('m', 'Масса m', 0.0, 0, 1.5, 0.5), P('rho', 'ρ', 4.0, 1, 4, 1), P('H', 'Высота ромба H', 8.0, 6, 10, 1)],
    presets={'быстро': {}, 'как в документе': {'H': 10}}, ref='mass-doc §9 (83 % / 92 %)', cost='~1 мин (H = 8), ~4 мин (H = 10)')
def exp_light31(p, region=None):
    w = Wd.gold31(p['rho'], 'diamond', H=p['H']); X = w.X; n = w.n; I, J = Cs.links_dense(X)
    L = np.zeros((n, n)); L[I, J] = 1; Phi = (math.sqrt(p['rho']) / (2 * math.pi * math.sqrt(6))) * L; b = -p['m'] ** 2 / p['rho']
    K = np.linalg.solve((np.eye(n) - b * Phi).T, Phi.T).T; lam, V = np.linalg.eigh(1j * (K - K.T)); o = np.argsort(-lam); lam, V = lam[o], V[:, o]
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]; kmag = np.linspace(0, 2.4, 25); oms = np.linspace(-3, 3, 241); Et = np.exp(1j * np.outer(oms, X[:, 0]))
    rows = []
    for i in range(12):
        v = V[:, i]; best = (0, 0, 0)
        for di, dv in enumerate(dirs):
            pr = X[:, 1:] @ dv
            for km in (kmag if di == 0 else kmag[1:]):
                z = np.abs(Et.conj() @ (v * np.exp(-1j * km * pr))); j = int(np.argmax(z))
                if z[j] > best[0]: best = (z[j], oms[j], km)
        rows.append((best[1], best[2]))
    ok = np.mean([abs(abs(om) - math.sqrt(p['m'] ** 2 + k * k)) < 0.2 for om, k in rows]); same = max(np.mean([om > 0 for om, _ in rows]), np.mean([om < 0 for om, _ in rows]))
    return {'numbers': [num('n', 'событий', n, fmt='{:.0f}'), num('disp', 'доля мод с |ω| = √(m² + k²) (±0.2)', ok, 0.85, 0.1), num('sign', 'доля мод одного знака частоты', same, 1.0, 0.1)],
            'series': [{'title': 'Медленные моды: |ω| против |k|', 'xlabel': '|k|', 'lines': [{'name': 'кристалл', 'x': [k for _, k in rows], 'y': [abs(om) for om, _ in rows], 'style': 'dots'},
                        {'name': 'теория √(m² + k²)', 'x': kmag.tolist(), 'y': np.sqrt(p['m'] ** 2 + kmag ** 2).tolist()}]}],
            'notes': ['Эталон (H = 10): свет 83 %, масса 1 — 92 %, один знак частоты.']}

# ======================= ПОЛЕ, АТОМ, ВОДОРОД =======================
@experiment(id='coulomb31', block='Поле, атом, водород', title='Закон Кулона из связей 3+1',
    what='Неподвижный заряд — трубка событий вдоль времени. Потенциал = (функция Грина Джонстона) × (число связей точки с трубкой); континуум требует r·n(r) = const.',
    params=[P('rho', 'ρ', 1.5, 1, 2, 0.5), P('a', 'Радиус трубки', 1.2, 0.8, 1.6, 0.1), P('nf', 'Точек поля на радиус', 30, 10, 80, 10)],
    presets={'быстро': {}, 'как в документе': {'nf': 60}}, ref='maxwell-doc §2 (отношение 1.07, наклон −1.04)', cost='~1–3 мин')
def exp_coulomb31(p, region=None):
    RS = (2.5, 4.0, 6.0, 8.0); rho, a = p['rho'], p['a']; TL = -(max(RS) + 4.5); S = max(RS) + 2.5
    G = Wd.points_in_box31([(TL, 0.6), (-S, S), (-S, S), (-S, S)], Wd.r_for_density31(rho)); vol = (0.6 - TL) * (2 * S) ** 3; dens = len(G) / vol
    tube = G[np.linalg.norm(G[:, 1:], axis=1) < a]; rng = np.random.default_rng(1); rr = np.linalg.norm(G[:, 1:], axis=1); rows = []
    for R in RS:
        idx = np.nonzero((np.abs(G[:, 0]) < 0.3) & (np.abs(rr - R) < 0.25))[0]; F = G[rng.choice(idx, min(int(p['nf']), len(idx)), replace=False)]; cnt = []
        for xx in F:
            d = xx - G; Px = G[d[:, 0] > np.linalg.norm(d[:, 1:], axis=1) + 1e-12]; dy = xx - tube
            cand = tube[dy[:, 0] > np.linalg.norm(dy[:, 1:], axis=1) + 1e-12]; nl = 0
            for y in cand:
                e = Px - y; nl += not (e[:, 0] > np.linalg.norm(e[:, 1:], axis=1) + 1e-12).any()
            cnt.append(nl)
        rows.append((np.linalg.norm(F[:, 1:], axis=1).mean(), np.mean(cnt), np.std(cnt) / math.sqrt(len(cnt))))
    formula = 4 / 3 * math.pi * a ** 3 * math.sqrt(6 * dens) / 2; ratio = np.mean([r * n / formula for r, n, _ in rows])
    slope = np.polyfit(np.log([r for r, _, _ in rows]), np.log([n for _, n, _ in rows]), 1)[0]
    return {'numbers': [num('ratio', 'r·n(r) / формула (среднее)', ratio, 1.07, 0.15), num('slope', 'наклон log n / log r', slope, -1.0, 0.15), num('dens', 'плотность', dens, fmt='{:.3f}')],
            'series': [{'title': 'r·n(r) / формула', 'xlabel': 'r', 'lines': [{'name': 'кристалл', 'x': [r for r, _, _ in rows], 'y': [r * n / formula for r, n, _ in rows], 'style': 'dots'},
                        {'name': 'Кулон', 'x': [2, 8.5], 'y': [1, 1]}]}], 'notes': ['Ошибка одной точки ~10 %; трубка одна, её флуктуации общие.']}

def _packet_force(w, Pfun, dUfun, m, rho, x0, t0=-6.0, steps=(2.0, 4.0, 6.0, 8.0)):
    Phi, B = Pr.corner_dense(w.X, Pfun, dUfun, m, rho); K = Pr.johnston11(w.X, rho, m, Phi, B); Wp, _, _ = Pr.sj(K)
    f = np.where(np.abs(w.t - t0) < 0.45, np.exp(-(w.x - x0) ** 2 / (2 * 1.2 ** 2)), 0).astype(complex); psi = Wp.T @ f
    return Ob.centers(w.t, w.x, psi, [t0 + s for s in steps]), psi

@experiment(id='charge_force', block='Поле, атом, водород', title='Сила между зарядами',
    what='Заряд-источник в Xc (поле в 1+1 — V-образный потенциал), пробный массивный квант заряда ±q (пакет SJ, связь — световой уголок). Одноимённые расходятся, разноимённые сходятся.',
    params=[P('qE', 'Сила qE', 0.042, 0.01, 0.1, 0.002), P('Xc', 'Положение заряда', -3.0, -6, 6, 0.5), P('x0', 'Старт пакета', 1.0, -6, 6, 0.5), P('L', 'Ромб L', 16.0, 12, 20, 2)],
    presets={'быстро': {'L': 14}, 'как в документе': {'L': 16}}, ref='maxwell-doc §9 (нечётная 0.834, чётная 0.017)', cost='~1 мин')
def exp_charge_force(p, region=None):
    w = Wd.gold11(4.0, p['L']); steps = (2.0, 4.0, 6.0, 8.0); res = {}
    for sgn in (+1, 0, -1):
        Pf, dU, _ = Fl.vshape(-sgn * p['qE'], p['Xc'])          # sgn = +1: одноимённые (U убывает от заряда — отталкивание)
        res[sgn], psi = _packet_force(w, Pf, dU, 1.0, 4.0, p['x0'])
        if sgn == 0: ov = np.abs(psi)
    away = np.sign(p['x0'] - p['Xc']); like = (res[1] - res[0]) * away; unlike = (res[-1] - res[0]) * away
    odd = (like - unlike) / 2; even = (like + unlike) / 2
    return {'numbers': [num('like', 'одноимённые: сдвиг от заряда к t = 8', like[-1]), num('unlike', 'разноимённые: сдвиг от заряда', unlike[-1]),
                        num('odd', 'нечётная часть (сила)', odd[-1], 0.834, 0.1, rel=True), num('even', 'чётная часть (должна быть ≈ 0)', even[-1], 0.0, 0.03)],
            'series': [{'title': 'Сдвиг центра от заряда', 'xlabel': 'время от старта', 'lines': [{'name': 'одноимённые', 'x': list(steps), 'y': like.tolist()}, {'name': 'разноимённые', 'x': list(steps), 'y': unlike.tolist()}]}],
            'overlay': overlay(w, ov / ov.max(), '|ψ| свободного пакета'), 'notes': ['Знак «одноимённые» задан энергией: E = m + qV (maxwell-doc §5).']}

def _spectro(w, Pf, dU, U, m, rho, parity, sig, om, ts, Lw, kind='odd'):
    g = (lambda s: s * np.exp(-s * s / (2 * sig * sig))) if parity == 'odd' else (lambda s: np.exp(-s * s / (2 * sig * sig)))
    phi = Pr.corner_respond(w.X, Pf, dU, m, rho, np.where(np.abs(w.t - ts) < 0.45, g(w.x), 0))
    sel = (w.t > ts + 1.5) & (w.t < Lw - 8) & (np.abs(w.x) < 8); S = Ob.spectrum(w.t[sel], g(w.x[sel]) * phi[sel], om)
    return S / S.max(), phi

@experiment(id='atom11', block='Поле, атом, водород', title='Спектр атома 1+1',
    what='Тяжёлое ядро, потенциал U = F|x| (поле ядра в 1+1). Удар у ядра — отклик звенит на уровнях; сравнение с уравнением Клейна–Гордона.',
    params=[P('F', 'Сила F', 0.117, 0.03, 0.3, 0.01), P('parity', 'Удар', 'odd', choices=['odd', 'even']), P('L', 'Ромб L', 30.0, 20, 36, 2)],
    presets={'быстро': {'L': 24}, 'как в документе': {'L': 36}}, ref='maxwell-doc §10 (до 1–3 %)', cost='~1–2 мин')
def exp_atom11(p, region=None):
    w = Wd.gold11(4.0, p['L']); Pf, dU, U = Fl.vshape(p['F']); om = np.linspace(0.5, 2.6, 1051)
    S, phi = _spectro(w, Pf, dU, U, 1.0, 4.0, p['parity'], 1.0, om, -p['L'] + 10, p['L'])
    pk = [e for e, h in Ob.peaks(om, S) if h > 0.1]; kg = Ob.kg_levels_linear(p['F'], 1.0, p['parity'], 2.6)
    nums = [num(f'E{i}', f'уровень {i + 1} (КГ {kg[i]:.3f})', min(pk, key=lambda e: abs(e - kg[i])) if pk else None, kg[i], 0.03, rel=True) for i in range(min(2, len(kg)))]
    return {'numbers': nums, 'series': [{'title': 'Спектр отклика', 'xlabel': 'E', 'lines': [{'name': 'кристалл', 'x': om.tolist(), 'y': S.tolist()}], 'marks': kg}],
            'overlay': overlay(w, np.abs(phi) / np.abs(phi).max(), '|φ|'), 'notes': [f'Пики: {", ".join(f"{e:.3f}" for e in pk)}']}

@experiment(id='hydrogen11', block='Поле, атом, водород', title='Водород, s-уровни (радиальная задача)',
    what='Нечётный сектор одномерной задачи с U = −α/√(x² + a²) — ровно s-волна водорода. Связь — световой уголок (точно при сильном поле).',
    params=[P('alpha', 'α', 0.45, 0.2, 0.49, 0.01), P('a', 'Размер ядра a', 0.1, 0.05, 1.5, 0.05), P('L', 'Ромб L', 44.0, 30, 56, 2)],
    presets={'быстро': {'L': 36}, 'как в документе': {'L': 56}}, ref='maxwell-doc §13 (1s 0.878 / 0.871 против КГ 0.8755)', cost='~1–3 мин')
def exp_hydrogen11(p, region=None):
    w = Wd.gold11(4.0, p['L']); Pf, dU, U = Fl.coulomb_soft(p['alpha'], p['a']); a0 = 1 / p['alpha']; om = np.linspace(0.55, 1.2, 1301)
    g = lambda s: s * np.exp(-np.abs(s) / a0); ts = -p['L'] + 8
    phi = Pr.corner_respond(w.X, Pf, dU, 1.0, 4.0, np.where(np.abs(w.t - ts) < 0.45, g(w.x), 0))
    sel = (w.t > ts + 1.5) & (w.t < p['L'] - 8) & (np.abs(w.x) < 8); S = Ob.spectrum(w.t[sel], g(w.x[sel]) * phi[sel], om); S /= S.max()
    kg = Ob.kg_levels(U, 1.0, 'odd', 0.55, 0.9995)
    pk = [e for e, h in Ob.peaks(om, S, 0.03) if e < 0.995]; best = min(pk, key=lambda e: abs(e - kg[0])) if pk else None
    return {'numbers': [num('E1s', f'1s на кристалле (КГ {kg[0]:.4f})', best, kg[0], 0.01, rel=True)],
            'series': [{'title': 'Спектр отклика (ниже 1 — связанные)', 'xlabel': 'E', 'lines': [{'name': 'кристалл', 'x': om.tolist(), 'y': S.tolist()}], 'marks': kg[:3]}],
            'notes': [f'Пики ниже 1: {", ".join(f"{e:.4f}" for e in pk)}; ядро острее зерна (a ≲ 0.05) не разрешается.']}

@experiment(id='hydrogen31', block='Поле, атом, водород', title='Водород в 3+1',
    what='Трёхмерный кристалл, пропагатор по связям, прямая нить Вильсона; удар формой 1s. Уровень 1s против Клейна–Гордона (s-волна, тот же потенциал).',
    params=[P('alpha', 'α', 0.45, 0.2, 0.49, 0.05), P('H', 'Окно H', 20.0, 12, 20, 2), P('Rc', 'Радиус цилиндра', 7.0, 5, 7, 1)],
    presets={'быстро': {'H': 12, 'Rc': 6}, 'как в документе': {'H': 20, 'Rc': 7}}, ref='maxwell-doc §16 (0.9044 против 0.9041 — ↺ понижено 16 окт.: при ρ = 1 отклик растёт, уровень гуляет ±5 % по окну и плотности; см. spin31)', cost='~2 мин (быстро), ~6 мин (документ)')
def exp_hydrogen31(p, region=None):
    w = Wd.gold31(1.0, 'cylinder', H=p['H'], Rc=p['Rc'], Rs=3.0); I, J = Cs.links_sparse(w.X, 1.0, 1.6); X = w.X; a, al = 0.5, p['alpha']; a0 = 1 / al
    f = np.where(X[:, 0] < 0.45, np.exp(-np.linalg.norm(X[:, 1:], axis=1) / a0), 0)
    def level(phase):
        phi = Pr.johnston31_respond(w.n, I, J, 1.0, 1.0, f, phase)
        sel = (X[:, 0] > 1.5) & (X[:, 0] < p['H'] - 0.3); tt = X[sel, 0]; g = np.exp(-np.linalg.norm(X[sel, 1:], axis=1) / a0)
        om = np.linspace(0.5, 1.6, 2201); S = Ob.spectrum(tt, g * phi[sel], om, 'hann'); return Ob.peaks(om, S, 0.0)[np.argmax([h for _, h in Ob.peaks(om, S, 0.0)])][0], om, S / S.max()
    E0, om, S0 = level(None); E1, _, S1 = level(Pr.wilson31(X, I, J, lambda r: -al / np.sqrt(r * r + a * a))); E2, _, S2 = level(Pr.wilson31(X, I, J, lambda r: al / np.sqrt(r * r + a * a)))
    kg = Ob.kg_levels(lambda s: -al / math.sqrt(s * s + a * a), 1.0, 'odd', 0.5, 0.9995)
    return {'numbers': [num('n', 'событий / связей', w.n, fmt='{:.0f}'), num('E1s', f'1s на кристалле (КГ {kg[0]:.4f}; систематика ±5 %)', E1, kg[0], 0.05, rel=True),
                        num('free', 'без ядра (кинетика источника, выше 1)', E0, fmt='{:.4f}'), num('rep', 'отталкивающее ядро (выше свободного)', E2, fmt='{:.4f}')],
            'series': [{'title': 'Спектр (ветвь частицы)', 'xlabel': 'E', 'lines': [{'name': 'притяжение', 'x': om.tolist(), 'y': S1.tolist()}, {'name': 'без ядра', 'x': om.tolist(), 'y': S0.tolist()}, {'name': 'отталкивание', 'x': om.tolist(), 'y': S2.tolist()}], 'marks': kg[:1]}],
            'notes': [f'Связей {len(I)}. Мера — абсолютный уровень (сдвиг от свободного пика — не энергия связи).', '↺ 16 окт.: при ρ = 1 масса m = 1 равна масштабу зерна — отклик растёт со временем, уровень почти не слушает массу (dE/dm² 0.13 против 0.49). Совпадение до 0.03 % — удача точки m = 1; честная систематика ±5 % (dirac-doc §20).']}

@experiment(id='geom_charge', block='Поле, атом, водород', title='Заряд из геометрии',
    what='Квант с целым числом оборотов m фазы вокруг окна в поле закрученного среза (связность нормалей). Без «q»: ускорение ∝ m. Контроль — тот же опыт во внешнем однородном поле той же силы: отношение должно быть 1.',
    params=[P('kappa', 'Закрутка κ', 0.055, 0.03, 0.06, 0.005), P('L', 'Ромб L', 16.0, 12, 16, 2)],
    region='point', presets={'быстро': {'L': 12}, 'как в документе': {'L': 16}}, ref='maxwell-doc §22 (поправка 15 окт.: срез / внешнее = 1.00 ± 0.03 на оси)', cost='~3–8 мин')
def exp_geom_charge(p, region=None):
    from scipy.interpolate import RegularGridInterpolator
    from .core import layers
    w = Wd.gold11(4.0, p['L']); t, x, X = w.t, w.x, w.X; h = Fl.slice_twist(p['kappa']); L = p['L']
    g = np.linspace(-L - 1, L + 1, 161); AT = np.zeros((161, 161)); Ax = np.zeros((161, 161))
    for i, Tq in enumerate(g):
        for j, xq in enumerate(g):
            try: a_, b_ = Fl.connection(h, Tq, xq); AT[i, j], Ax[i, j] = a_[0, 1], b_[0, 1]
            except ValueError: pass
    fAT = RegularGridInterpolator((g, g), AT); fAx = RegularGridInterpolator((g, g), Ax); dg = g[1] - g[0]
    fF = RegularGridInterpolator((g, g), np.gradient(Ax, dg, axis=0) - np.gradient(AT, dg, axis=1))
    gx, gw = np.polynomial.legendre.leggauss(6)
    def seg(t0_, x0_, t1_, x1_):
        tot = 0
        for xg, wg in zip(gx, gw):
            s = 0.5 * (xg + 1); pts = np.stack([t0_ + s * (t1_ - t0_), x0_ + s * (x1_ - x0_)], -1); tot = tot + 0.5 * wg * (fAT(pts) * (t1_ - t0_) + fAx(pts) * (x1_ - x0_))
        return tot
    C = Cs.causal(X); I, J = np.nonzero(C); ty, xy, tx, xx = t[I], x[I], t[J], x[J]; tc = 0.5 * ((ty - xy) + (tx + xx)); xc = 0.5 * ((ty + xy) - (tx - xx))
    W1 = np.empty(len(I))
    for s0 in range(0, len(I), 300000):
        sl = slice(s0, s0 + 300000); W1[sl] = seg(ty[sl], xy[sl], tc[sl], xc[sl]) + seg(tc[sl], xc[sl], tx[sl], xx[sl])
    Ev = fF(np.stack([t, x], -1))
    t0, x0 = ((region['t0'] + region['t1']) / 2, (region['x0'] + region['x1']) / 2) if region else (-0.375 * L, 0.0)
    span = min(8.0, L - 3 - t0); LT = np.array([q for q, _ in layers(X)]); tl = LT[(LT > t0 + 1.0) & (LT < t0 + span)]; s_ = tl - t0
    Fpath = float(np.mean(fF(np.stack([np.linspace(t0, t0 + span, 40), np.full(40, x0)], -1))))
    def cent(Wp, A0, mq):                                                   # центр |ψ|² на точных слоях времени
        f = np.where(np.abs(t - t0) < 0.45, np.exp(-(x - x0) ** 2 / (2 * 1.2 ** 2)) * np.exp(1j * mq * A0 * (x - x0)), 0); psi = Wp.T @ f; out = []
        for q in tl:
            S = np.abs(t - q) < 1e-9; pp = np.abs(psi[S]) ** 2; out.append((pp * x[S]).sum() / pp.sum())
        return np.array(out)
    def geom(mq):
        Phi = np.zeros((w.n, w.n), complex); Phi[I, J] = 0.5 * np.exp(1j * mq * W1); B = -(1 - 1j * mq * (-Ev)) / 4.0
        return cent(Pr.sj(Pr.johnston11(X, 4.0, 1.0, Phi, B))[0], fAx(np.array([[t0, x0]]))[0], mq)
    def ext(mq):                                                            # внешнее однородное поле той же силы (F на пути), световой уголок
        P_, dU, _ = Fl.uniform(mq * abs(Fpath)); Phi, B = Pr.corner_dense(X, P_, dU, 1.0, 4.0); return cent(Pr.sj(Pr.johnston11(X, 4.0, 1.0, Phi, B))[0], 0.0, mq)
    acc = lambda y: np.linalg.lstsq(np.c_[0.5 * s_ * s_, s_], y, rcond=None)[0][0]
    cg = {mq: geom(mq) for mq in (0, 1, -1, 2, -2, 3, -3)}; odd = {mq: (cg[mq] - cg[-mq]) / 2 for mq in (1, 2, 3)}; a = {mq: acc(odd[mq]) for mq in (1, 2, 3)}
    ce = {mq: ext(mq) for mq in (1, -1, 2, -2, 3, -3)}; oe = {mq: (ce[mq] - ce[-mq]) / 2 for mq in (1, 2, 3)}; ae = {mq: acc(oe[mq]) for mq in (1, 2, 3)}
    Fc = Fl.curvature(h, 0.3, 0.2)[0, 1]
    return {'numbers': [num('F', 'кривизна нормальной связности у центра', Fc, -2 * p['kappa'] ** 2, 1e-3, rel=True, fmt='{:.6f}'),
                        num('Fpath', 'поле на пути пакета (вдали от центра ромба оно сильнее)', Fpath, fmt='{:.5f}'),
                        num('ratio', 'срез / внешнее поле, m = 1', abs(a[1]) / abs(ae[1]), 1.0, 0.05, rel=True),
                        num('ratio2', 'срез / внешнее поле, m = 2 (внешнее — заряд 2)', abs(a[2]) / abs(ae[2]), 1.0, 0.05, rel=True),
                        num('ratio3', 'срез / внешнее поле, m = 3', abs(a[3]) / abs(ae[3]), 1.0, 0.05, rel=True),
                        num('r2', f'ускорение m = 2 / m = 1 (во внешнем поле {ae[2] / ae[1]:.2f})', a[2] / a[1], fmt='{:.2f}'), num('r3', f'ускорение m = 3 / m = 1 (во внешнем {ae[3] / ae[1]:.2f})', a[3] / a[1], fmt='{:.2f}'),
                        num('probe', 'зонд-пакет: ускорение / F (метод, не физика)', abs(ae[1]) / abs(Fpath), fmt='{:.2f}')],
            'series': [{'title': 'Сдвиг (нечётная часть) по слоям времени', 'xlabel': 'время от старта', 'lines': [{'name': f'срез, m = {mq}', 'x': s_.tolist(), 'y': np.abs(odd[mq]).tolist(), 'style': 'dots'} for mq in (1, 2, 3)] +
                        [{'name': 'внешнее поле, m = 1', 'x': s_.tolist(), 'y': np.abs(oe[1]).tolist(), 'style': 'dots'}]}],
            'notes': [f'Старт (t, x) = ({t0:.1f}, {x0:.1f}); точку можно выделить на мире. Ускорение — подгонка a·s²/2 + v·s по {len(tl)} слоям.',
                      'Центр пакета SJ — шумный зонд силы: даже во внешнем однородном поле его ускорение зависит от места старта (0.4–0.9 от F). Поэтому физика — отношение к внешнему полю при том же старте.',
                      'Вне оси (x₀ = ±1) отношение уходит в 0.8 / 1.2 (в среднем 1.0) — не объяснено.']}

# ======================= ПАРЫ И БЕЛЛ =======================
@experiment(id='light_point', block='Пары и Белл', title='Квант из точки (полярное событие)',
    what='Свет из события, ближайшего к центру выделения: каждое событие переводит амплитуды входящих связей в выходящие ближайшей изометрией; лишнее уходит в тень. Мир + тень + край = 1 точно.',
    params=[P('kind', 'Мир', 'gold11', choices=['gold11', 'poisson11']), P('rho', 'ρ', 4.0, 2, 8, 1), P('L', 'Ромб L', 10.0, 8, 14, 1), P('m0', 'Масштаб ядра m₀', 2.0, 0.5, 4, 0.5)],
    region='point', presets={'быстро': {}, 'как в документе': {}}, ref='closedsym / песочница (норма точная)', cost='секунды')
def exp_light_point(p, region=None):
    w = Wd.make(p['kind'], rho=p['rho'], L=p['L']); la, lb = Cs.staircase11(w.X); PL = Ob.PolarLight(w.X, la, lb, p['m0'])
    tc, xc = ((region['t0'] + region['t1']) / 2, (region['x0'] + region['x1']) / 2) if region else (-0.72 * p['L'], 0.0)
    src = int(np.argmin(np.abs(w.t - tc) + np.abs(w.x - xc))); amp, shadow = PL.run(src)
    ev = np.zeros(w.n); np.add.at(ev, lb, amp ** 2)
    edge = sum((amp[PL.inL[k]] ** 2).sum() for k in range(w.n) if not PL.outL[k])
    return {'numbers': [num('total', 'мир + тень + край (всё учтено)', shadow + edge, 1.0, 1e-9, fmt='{:.9f}'), num('shadow', 'ушло в тень', shadow), num('edge', 'ушло за край', edge)],
            'overlay': overlay(w, np.sqrt(ev / max(ev.max(), 1e-30)), '|амплитуда| на входе события'), 'series': [], 'notes': [f'Источник: событие ({w.t[src]:.2f}, {w.x[src]:.2f}).']}

@experiment(id='bell', block='Пары и Белл', title='Алиса и Боб: синглет и CHSH',
    what='Пара рождается в неизвестном месте полосы накачки (сумма по местам), кванты летят полярным светом к портам Алисы и их зеркалам у Боба. Нечётная накачка + точное зеркало золота ⇒ синглет.',
    params=[P('kind', 'Мир', 'gold11', choices=['gold11', 'poisson11']), P('pump', 'Накачка', 'odd', choices=['odd', 'even']), P('rho', 'ρ', 4.0, 2, 8, 1), P('L', 'Ромб L', 10.0, 8, 14, 1)],
    presets={'быстро': {}, 'как в документе': {}}, ref='песочница v0.3 (1.0000, 2.828); bell-chain-doc', cost='секунды')
def exp_bell(p, region=None):
    w = Wd.make(p['kind'], rho=p['rho'], L=p['L']); t, x = w.t, w.x; L = p['L']; la, lb = Cs.staircase11(w.X); PL = Ob.PolarLight(w.X, la, lb, 2.0)
    strip = np.nonzero((np.abs(t + 0.6 * L) < 0.45) & (np.abs(x) <= 3) & (np.abs(x) > 1e-9))[0]
    near = lambda tt, xx: int(np.argmin(np.where(np.abs(t - tt) < 0.6, np.abs(x - xx) + 0.3 * np.abs(t - tt), np.inf)))
    A1 = near(0.3 * L, -0.3 * L); A2 = int(np.argmin(np.where((np.abs(t - t[A1]) < 0.6) & (np.arange(w.n) != A1), np.abs(x - (x[A1] - 1.3)), np.inf)))
    def mirror(k):
        m = np.nonzero((np.abs(t - t[k]) < 1e-9) & (np.abs(x + x[k]) < 1e-9))[0]
        return (int(m[0]), True) if len(m) else (near(t[k], -x[k]), False)
    (B1, e1), (B2, e2) = mirror(A1), mirror(A2); M = np.zeros((2, 2)); field = np.zeros(PL.E)
    for s in strip:
        pv = np.sign(x[s]) if p['pump'] == 'odd' else 1.0; amp, _ = PL.run(s); a = [PL.arrival(amp, A1), PL.arrival(amp, A2)]; b = [PL.arrival(amp, B1), PL.arrival(amp, B2)]
        M += pv * np.outer(a, b); field += pv * amp
    psi = (M / np.linalg.norm(M)).ravel(); singlet = (psi[1] - psi[2]) ** 2 / 2; S = Ob.chsh_max(psi)
    betas = np.arange(0, 181, 2.5); ev = np.zeros(w.n); np.add.at(ev, lb, np.abs(field))
    exp_s = (1.0 if (p['pump'] == 'odd' and p['kind'] == 'gold11') else None)
    return {'numbers': [num('singlet', 'доля синглета', singlet, exp_s, 1e-3), num('S', 'CHSH, наибольшее S (классика ≤ 2)', S, 2 * math.sqrt(2) if exp_s else None, 0.01), num('mirror', 'порты Боба — точные зеркала', float(e1 and e2), fmt='{:.0f}')],
            'series': [{'title': 'Корреляция E(0°, β)', 'xlabel': 'угол Боба β (°)', 'lines': [{'name': 'кристалл', 'x': betas.tolist(), 'y': [Ob.correlation(psi, 0, b) for b in betas]}, {'name': 'синглет −cos 2β', 'x': betas.tolist(), 'y': (-np.cos(2 * np.radians(betas))).tolist()}]}],
            'overlay': overlay(w, ev / max(ev.max(), 1e-30), 'поле пары'), 'notes': ['Измерение — идеальный поворот базиса (в полной модели — делитель-событие).']}

@experiment(id='pair', block='Пары и Белл', title='Пара частица + античастица (пропагатор Фейнмана)',
    what='Амплитуда пары на «одновременных» парах событий, на каждой линии — пропагатор Фейнмана (частица вперёд, античастица назад, из вакуума SJ), мгновенная вершина F|x₁ − x₂|. Решение — GMRES.',
    params=[P('m1', 'm₁', 1.0, 0.5, 1.4, 0.1), P('m2', 'm₂', 1.0, 0.5, 1.4, 0.1), P('parity', 'Канал', 'odd', choices=['odd', 'even']), P('L', 'Ромб L', 16.0, 12, 20, 2),
            P('measure', 'Мера вершины', 'strip', choices=['strip', 'level'])],
    presets={'быстро': {'L': 14}, 'как в документе': {'L': 20}}, ref='maxwell-doc §21, §25 (точный c, 15 окт.: 0.389 против 0.425; нормировка открыта)', cost='~5–25 мин')
def exp_pair(p, region=None):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import LinearOperator, gmres
    w = Wd.gold11(4.0, p['L']); t, x, n = w.t, w.x, w.n; rho, dl, F = 4.0, 0.3, 0.08
    I1, I2 = np.nonzero((np.abs(t[:, None] - t[None, :]) < dl) & (np.abs(x[:, None] - x[None, :]) < 10.0))
    tb = 0.5 * (t[I1] + t[I2]); o = np.argsort(tb, kind='stable'); I1, I2, tb = I1[o], I2[o], tb[o]; M = len(I1)
    G1, c1, r1 = Pr.feynman(Pr.johnston11(w.X, rho, p['m1']), t); G2, c2, r2 = (G1, c1, r1) if p['m2'] == p['m1'] else Pr.feynman(Pr.johnston11(w.X, rho, p['m2']), t)
    def A(v):
        Zt = csr_matrix((v, (I1, I2)), shape=(n, n)).T @ G1; out = np.empty(M, complex)
        for s in range(0, M, 6000): sl = slice(s, s + 6000); out[sl] = (Zt[:, I1[sl]] * G2[:, I2[sl]]).sum(0)
        return out
    r = x[I1] - x[I2]; R = 0.5 * (x[I1] + x[I2]); ts = -p['L'] + 7; h = (r if p['parity'] == 'odd' else 1.0) * np.exp(-r * r / 4.5)
    f = np.where((np.abs(t[I1] - ts) < 0.45) & (np.abs(t[I2] - ts) < 0.45), h * np.exp(-R * R / 18), 0).astype(complex); b = A(f)
    meas = rho * Cs.time_levels(t, rho)[I1] if p['measure'] == 'level' else 2 * dl * rho ** 2
    wv = -1j * 4 * p['m1'] * p['m2'] * F * np.abs(r) / meas
    op = LinearOperator((M, M), matvec=lambda v: v - A(wv * v), dtype=complex); it = [0]
    psi, info = gmres(op, b, rtol=1e-4, restart=250 if p['measure'] == 'level' else 60, maxiter=3, callback=lambda rk: it.__setitem__(0, it[0] + 1), callback_type='pr_norm')
    res = np.linalg.norm(op @ psi - b) / np.linalg.norm(b)
    sel = (tb > ts + 1.5) & (tb < p['L'] - 5); om = np.linspace(p['m1'] + p['m2'] - 0.4, p['m1'] + p['m2'] + 1.2, 1601)
    S = Ob.spectrum(tb[sel], (h * np.exp(-R * R / 18))[sel] * psi[sel], om); S /= S.max(); pk = Ob.peaks(om - p['m1'] - p['m2'], S, 0.1)
    sal = Ob.salpeter_levels(p['m1'], p['m2'], F, p['parity']); main = max(pk, key=lambda z: z[1])[0] if pk else None
    return {'numbers': [num('c', '|c| в K_вперёд = c(W₊ − W₋) — тождество (K запаздывающий), проверка счёта', abs(c1), 1.0, 1e-6), num('resid', 'невязка того же тождества (не физика)', r1, 0.0, 1e-9, fmt='{:.1e}'),
                        num('gmres', f'невязка GMRES ({it[0]} итераций)', res, 0.0, 1e-3, fmt='{:.1e}'), num('E', f'главный уровень E − m₁ − m₂ (Солпитер {sal[0]:.3f}; измеритель ±0.1, §26)', main, sal[0], 0.25, rel=True)],
            'series': [{'title': 'Спектр пары', 'xlabel': 'E − m₁ − m₂', 'lines': [{'name': 'кристалл', 'x': (om - p['m1'] - p['m2']).tolist(), 'y': S.tolist()}], 'marks': sal}],
            'notes': ['Нормировка вершины не установлена (maxwell-doc §25): мера «полоса» даёт устойчиво −8.5…−10 % (мир 16: 0.381, 20: 0.389); мера «уровень» дрейфует с размером (16: 0.425, 18: 0.525, 20: не сходится).',
                      'Мера «полоса», ромб 20: 0.389. Прежние 0.422 (§21) — от подгонки множителя c по спектру (c = 0.076 − 1.029i вместо точного −i): с той же подгонкой воспроизводится 0.422 ровно (maxwell-doc §21, поправка 15 окт.).',
                      'Измеритель уровня пары ненадёжен: даже у свободной пары пик прыгает 0.07…0.29 с размером мира (maxwell-doc §26) — числа сравнивать с теорией пока нельзя.']}

# ======================= ГРАВИТАЦИЯ СРЕЗА И ЦВЕТ =======================
@experiment(id='slice_rt', block='Гравитация среза и цвет', title='Уравнения среза (Редже–Тейтельбойм)',
    what='Лист мира гнётся в тень; закон «экстремум действия по форме» даёт Эйнштейна плюс «материю вложения». Отклонение от Шварцшильда на δA даёт добавочное притяжение, растущее как r^{3/2}.',
    params=[P('dA', 'Отклонение δA/A', 0.01, -0.05, 0.05, 0.005), P('w', 'Скорость поворота в тени ω', 1.0, 0.5, 2, 0.5), P('r1', 'До радиуса', 200.0, 60, 300, 20)],
    presets={'быстро': {'r1': 120}, 'как в документе': {'r1': 290}}, ref='slice-doc §5–6 (q = 1/4; наклон 1.45 на r = 30…160, растёт к 3/2 — поправка 15 окт.)', cost='~10–30 с')
def exp_slice_rt(p, region=None):
    o = Fl.rt_integrate(1.0, p['w'], 10.0, p['r1'], dA=p['dA'], n=800); r = o[:, 0]; v2 = o[:, 3]; ex = v2 * r - r / (r - 2)
    rhi = min(160.0, r[-1]); band = (r >= 30) & (r <= rhi)                  # r = 30…160: здесь решение устойчиво к шагу (slice-doc §5, поправка 15 окт.)
    ok = band & np.isfinite(ex) & (np.abs(ex) > 1e-6); slope = np.polyfit(np.log(r[ok]), np.log(np.abs(ex[ok])), 1)[0] if ok.sum() > 5 else float('nan')
    qv = o[band, 4]; qv = qv[np.isfinite(qv)]
    return {'numbers': [num('q', f'p_t / p_r, медиана на r = 30…{rhi:.0f}', float(np.median(qv)) if len(qv) else None, 0.25, 0.01), num('slope', f'наклон log M_доп / log r на r = 30…{rhi:.0f} (асимптота 3/2)', slope, 1.45, 0.03), num('rmax', 'досчитано до r (зависит от шага — не физика)', r[-1], fmt='{:.0f}')],
            'series': [{'title': 'Скорость на орбите v²', 'xlabel': 'r', 'lines': [{'name': 'срез (Р–Т)', 'x': r.tolist(), 'y': v2.tolist()}, {'name': 'Кеплер (Шварцшильд)', 'x': r.tolist(), 'y': (1 / (r - 2)).tolist()}]}],
            'notes': ['Плоские кривые вращения требуют q → 0; в винтовом классе q → 1/4 притягивает — плоских кривых нет (slice-doc §6).', 'M_доп = v²r − r/(r − 2): добавка к кеплеровой (шварцшильдовой) скорости. Где решение «рвётся», зависит от шага (161…300+) — это не физический результат.']}

@experiment(id='nonabelian', block='Гравитация среза и цвет', title='Неабелево поле из формы среза',
    what='Срез 1+1 в тени R⁴. Связность нормалей — so(4). Если плоскости изгиба поворачиваются от места к месту, обходы двух петель в разном порядке не коммутируют (признак глюонного поля); контроль — неподвижные плоскости.',
    params=[P('k', 'Изгиб κ', 0.15, 0.05, 0.2, 0.01), P('om', 'Скорость поворота плоскостей', 1.0, 0, 1.5, 0.1)],
    presets={'быстро': {}, 'как в документе': {}}, ref='gauge-doc (1.5·10⁻³…10⁻² против 10⁻¹⁴)', cost='~1 мин')
def exp_nonabelian(p, region=None):
    from scipy.linalg import logm
    out = {}
    for name, h in (('rot', Fl.slice_rotating(p['k'], p['om'])), ('fix', Fl.slice_fixed4(p['k']))):
        F0, R0 = Fl.curvature(h, 0.0, 0.0), Fl.ricci(h, 0.0, 0.0)
        UA = Fl.holonomy(h, Fl.lasso((1.2, 1.2)), 30); UB = Fl.holonomy(h, Fl.lasso((-1.2, -1.2)), 30)
        LA, LB = np.real(logm(UA)), np.real(logm(UB))
        out[name] = (np.abs(F0 + R0).max() / max(np.abs(R0).max(), 1e-12), np.abs(UA @ UB - UB @ UA).max(), (LA * LB).sum() / math.sqrt((LA * LA).sum() * (LB * LB).sum()))
    return {'numbers': [num('ricci', 'уравнение Риччи |F + R|/|R|', out['rot'][0], 0.0, 1e-3, fmt='{:.1e}'), num('comm', 'коммутатор голономий (поворот плоскостей)', out['rot'][1], fmt='{:.1e}'),
                        num('commfix', 'то же, контроль (плоскости неподвижны)', out['fix'][1], 0.0, 1e-10, fmt='{:.1e}'), num('cos', 'cos между log U_A и log U_B (контроль: +1)', out['rot'][2], fmt='{:+.2f}')],
            'series': [], 'notes': ['Сравнивать F в разных точках нельзя — разные калибровки рамки; честный признак — голономия.']}

@experiment(id='color', block='Гравитация среза и цвет', title='Квант с цветом',
    what='Амплитуда — 4-вектор «цвета». Перенос вдоль светового уголка — упорядоченная экспонента so(4), вставка — матрица поля. Цвета (e₁ ± ie₂)/√2 — заряды ∓, e₃ нейтрален; в неабелевом поле «заряд по полю» не сохраняется.',
    params=[P('case', 'Срез', 'fix', choices=['fix', 'rot']), P('L', 'Ромб L', 12.0, 10, 14, 1)],
    presets={'быстро': {'L': 10}, 'как в документе': {'L': 14}}, ref='gauge-doc (абелево: q = ±1.000; неабелево: −0.913…−0.984)', cost='~2–16 мин')
def exp_color(p, region=None):
    from scipy.interpolate import RegularGridInterpolator
    from scipy.linalg import expm
    w = Wd.gold11(4.0, p['L']); t, x, n = w.t, w.x, w.n; L = p['L']
    h = Fl.slice_fixed4(0.05) if p['case'] == 'fix' else Fl.slice_rotating(0.04, 0.12); N = 61; g = np.linspace(-L - 0.5, L + 0.5, N)
    AT = np.zeros((N, N, 4, 4)); Ax = np.zeros((N, N, 4, 4))
    for i, Tq in enumerate(g):
        for j, xq in enumerate(g):
            if abs(Tq) + abs(xq) <= L + 1.5: AT[i, j], Ax[i, j] = Fl.connection(h, Tq, xq)
    dg = g[1] - g[0]; Fg = np.gradient(Ax, dg, axis=0) - np.gradient(AT, dg, axis=1) + np.einsum('ijab,ijbc->ijac', AT, Ax) - np.einsum('ijab,ijbc->ijac', Ax, AT)
    fAT, fAx, fF = (RegularGridInterpolator((g, g), Q) for Q in (AT, Ax, Fg))
    C = Cs.causal(w.X); I, J = np.nonzero(C); Ut = np.empty((len(I), 4, 4))
    for s0 in range(0, len(I), 200000):
        sl = slice(s0, s0 + 200000); ty, xy, tx, xx = t[I[sl]], x[I[sl]], t[J[sl]], x[J[sl]]
        tc = 0.5 * ((ty - xy) + (tx + xx)); xc = 0.5 * ((ty + xy) - (tx - xx)); U = np.broadcast_to(np.eye(4), (len(ty), 4, 4)).copy()
        for (a0, b0, a1, b1) in ((ty, xy, tc, xc), (tc, xc, tx, xx)):
            for k in range(8):
                sm = (k + 0.5) / 8; pts = np.stack([a0 + sm * (a1 - a0), b0 + sm * (b1 - b0)], -1)
                Mx = -(fAT(pts) * ((a1 - a0) / 8)[:, None, None] + fAx(pts) * ((b1 - b0) / 8)[:, None, None]); U = (np.eye(4) + Mx + 0.5 * Mx @ Mx + Mx @ Mx @ Mx / 6) @ U
        Ut[sl] = U
    Phi = np.zeros((4 * n, 4 * n))
    for a in range(4):
        for b in range(4): Phi[4 * I + a, 4 * J + b] = 0.5 * Ut[:, b, a]
    Fz = fF(np.stack([t, x], -1)); Bz = np.zeros((4 * n, 4 * n))
    for z in range(n): Bz[4 * z:4 * z + 4, 4 * z:4 * z + 4] = (-(np.eye(4) - Fz[z]) / 4.0).T
    K = np.linalg.solve((np.eye(4 * n) - Phi @ Bz).T, Phi.T).T; Wp, _, _ = Pr.sj(K)
    t0, x0 = -0.36 * L, 0.0; steps = [1.5, 3.0, 4.5, 6.0]; res = {}
    for nm, c in (('+', np.array([1, 1j, 0, 0]) / math.sqrt(2)), ('−', np.array([1, -1j, 0, 0]) / math.sqrt(2)), ('0', np.array([0, 0, 1.0, 0]))):
        A0 = fAx(np.array([[t0, x0]]))[0]; f = np.zeros(4 * n, complex)
        for z in np.nonzero(np.abs(t - t0) < 0.45)[0]: f[4 * z:4 * z + 4] = math.exp(-(x[z] - x0) ** 2 / 2.88) * (expm(-A0 * (x[z] - x0)) @ c)
        psi = (Wp.T @ f).reshape(n, 4); qs, cs = [], []
        for s in steps:
            S = np.nonzero(np.abs(t - (t0 + s)) < 0.5)[0]; Pp = (np.abs(psi[S]) ** 2).sum(1); Fn = fF(np.stack([t[S], x[S]], -1))
            Fn = Fn / np.maximum(np.abs(Fn).max(axis=(1, 2)), 1e-12)[:, None, None]; qs.append(np.einsum('za,zab,zb->z', psi[S].conj(), Fn, psi[S]).imag.sum() / Pp.sum()); cs.append((Pp * x[S]).sum() / Pp.sum())
        res[nm] = (np.array(qs), np.array(cs))
    exq = -1.0 if p['case'] == 'fix' else None
    return {'numbers': [num('qplus', 'заряд по полю цвета (e₁ + ie₂)/√2, в конце', res['+'][0][-1], exq, 0.01), num('qzero', 'заряд по полю цвета e₃', res['0'][0][-1], 0.0, 0.02),
                        num('qvar', 'изменение q цвета + за время', np.ptp(res['+'][0]), 0.0 if p['case'] == 'fix' else None, 0.005, fmt='{:.3f}')],
            'series': [{'title': 'Заряд по полю q(t)', 'xlabel': 'время', 'lines': [{'name': f'цвет {nm}', 'x': steps, 'y': res[nm][0].tolist()} for nm in res]},
                       {'title': 'Сдвиг центра', 'xlabel': 'время', 'lines': [{'name': f'цвет {nm}', 'x': steps, 'y': (res[nm][1] - res['0'][1]).tolist()} for nm in ('+', '−')]}],
            'notes': ['Источник ковариантный; лист проверен на лоренцевость во всём ромбе (det g < 0).']}


# ======================= ФЕРМИОНЫ И АТОМЫ (dirac-doc, 15 окт.) =======================
@experiment(id='onebody_long', block='Фермионы и атомы', title='Измеритель: одно тело на длинном окне',
    what='Скаляр (Джонстон, световой уголок) в поле F|x|. На мирах ≥ 24 уровень сходится; отличие от КГ — перенормировка массы кристалла (1.041, по дисперсии).',
    params=[P('F', 'Сила F', 0.08, 0.04, 0.15, 0.01), P('L', 'Ромб L', 28.0, 20, 44, 4)],
    presets={'быстро': {'L': 24}, 'как в документе': {'L': 36}}, ref='maxwell-doc §26 (0.371 = КГ при m = 1.041)', cost='~0.5–2 мин')
def exp_onebody_long(p, region=None):
    L = p['L']; w = Wd.gold11(4.0, L); t, x = w.t, w.x; ts = -L + 7; Pf, dU, U = Fl.vshape(p['F'])
    g = x * np.exp(-x * x / 4.5); phi = Pr.corner_respond(w.X, Pf, dU, 1.0, 4.0, np.where(np.abs(t - ts) < 0.45, g, 0))
    sel = (t > ts + 1.5) & (t < L - 5); k = np.round(t[sel], 6); tu, inv = np.unique(k, return_inverse=True); sg = np.zeros(len(tu), complex); np.add.at(sg, inv, (g * phi)[sel])
    om = np.linspace(0.6, 2.2, 1601); S = Ob.spectrum(tu, sg, om); S /= S.max(); pk = Ob.peaks(om - 1, S, 0.1); E = max(pk, key=lambda z: z[1])[0]
    kg1 = Ob.kg_levels_linear(p['F'], 1.0, 'odd', 2.8)[0] - 1; kgm = Ob.kg_levels_linear(p['F'], 1.041, 'odd', 2.8)[0] - 1
    return {'numbers': [num('E', f'уровень E − 1 (КГ при m = 1: {kg1:.3f})', E, kgm, 0.03, rel=True), num('kg', 'КГ при массе кристалла 1.041', kgm, fmt='{:.4f}')],
            'series': [{'title': 'Спектр', 'xlabel': 'E − 1', 'lines': [{'name': 'кристалл', 'x': (om - 1).tolist(), 'y': S.tolist()}], 'marks': [kg1, kgm]}],
            'notes': ['На мирах ≤ 20 измеритель недосчитывает (окно короче расстояния между уровнями).']}

@experiment(id='dirac_walk', block='Фермионы и атомы', title='Частица со спином: блуждание по нитям',
    what='Каждое событие — на одной нити вправо и одной влево; в событии монета (смешение компонент). Проверки: норма сохраняется точно, две ветви ±m, обе с положительной нормой; у скаляра вторая ветвь — отрицательная норма. Отсюда (Паули, 1940): частица со спином — фермион.',
    params=[P('m', 'Масса m', 1.0, 0.5, 1.5, 0.1), P('L', 'Ромб L', 16.0, 12, 24, 2)],
    presets={'быстро': {'L': 12}, 'как в документе': {'L': 16}}, ref='dirac-doc §1–3', cost='~10–40 с')
def exp_dirac_walk(p, region=None):
    from . import fermions as Fe
    L = p['L']; m = p['m']; w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Fe.Walk(w)
    lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9; g = np.where(src, np.exp(-x * x / 50), 0)
    sel, tu, proj = Fe.readout(w, ts, L, 8.0); res = {}
    for nm, cr, cl in (('+', 1, 1), ('−', 1, -1)):
        aR, aL, leak = W.run(m, cr * g, cl * g); inj = 2 * (g ** 2).sum(); h = np.exp(-x * x / 50)
        sp = proj(h, aR, aL) + proj(h, aR, -aL) if False else None
        s_ = np.zeros(len(tu), complex); kk = np.round(t[sel], 6); _, inv = np.unique(kk, return_inverse=True)
        np.add.at(s_, inv, (h * (cr * aR + cl * aL))[sel]); om = np.linspace(-2.5, 2.5, 2501); S = Ob.spectrum(tu, s_, om); S /= S.max()
        res[nm] = (abs(inj - leak) / inj, max(Ob.peaks(om, S, 0.1), key=lambda z: z[1])[0], om, S)
    lam = np.linalg.eigvalsh(1j * (Pr.johnston11(w.X, 4.0, m) - Pr.johnston11(w.X, 4.0, m).T)) if w.n < 2500 else np.array([0.0])
    return {'numbers': [num('c', 'скорость безмассовой частицы на нитях (должна быть 1 — открыто)', W.c, 1.0, 0.05, rel=True, fmt='{:.3f}'),
                        num('full', 'доля событий с полным набором нитей', W.full, 1.0, 0.03, fmt='{:.3f}'),
                        num('norm', 'норма: |вход − выход| / вход', max(res['+'][0], res['−'][0]), 0.0, 1e-12, fmt='{:.1e}'),
                        num('wp', 'частота ветви (1, 1) — частица (ожидание −m)', res['+'][1], -m, 0.03, rel=True, fmt='{:+.3f}'),
                        num('wm', 'частота ветви (1, −1) — античастица (ожидание +m)', res['−'][1], m, 0.03, rel=True, fmt='{:+.3f}'),
                        num('neg', 'скаляр: доля отрицательных собственных iΔ (бозоны: 0.5)', float((lam < -1e-9).sum() / max((np.abs(lam) > 1e-9).sum(), 1)), 0.5, 0.01, fmt='{:.3f}')],
            'series': [{'title': 'Спектры ветвей (норма обеих положительна)', 'xlabel': 'ω', 'lines': [{'name': 'ветвь (1, 1)', 'x': res['+'][2].tolist(), 'y': res['+'][3].tolist()}, {'name': 'ветвь (1, −1)', 'x': res['−'][2].tolist(), 'y': res['−'][3].tolist()}]}],
            'notes': ['Положительная норма обеих ветвей ⇒ согласованно только антикоммутатор ⇒ фермионы ⇒ Паули (фаза π за пересечение историй).', 'Скорость нитей ≈ 0.76: нити выбраны в системе кристалла (dirac-doc §8, §10).']}

@experiment(id='field11', block='Фермионы и атомы', title='Поле заряда из порядка (1+1)',
    what='Неподвижный заряд — трубка событий. Потенциал = ½ за каждое её событие в прошлом конусе (пропагатор света Джонстона). В 1+1 — линейный по |x| с наклоном −q/2: сила между зарядами из одного порядка.',
    params=[P('L', 'Ромб L', 24.0, 12, 24, 4), P('a', 'Радиус трубки', 1.0, 0.5, 2, 0.5)],
    presets={'быстро': {'L': 16}, 'как в документе': {'L': 24}}, ref='dirac-doc §7 (−0.49 против −0.5)', cost='секунды')
def exp_field11(p, region=None):
    L, a = p['L'], p['a']; w = Wd.gold11(4.0, L); X, t, x = w.X, w.t, w.x; tube = np.nonzero(np.abs(x) < a)[0]
    Np = Cs.causal(X)[tube].sum(0); lam = len(tube) / (2 * L - 2 * a) / (2 * a); phi = 0.5 * Np / (lam * 2 * a)
    S = (np.abs(t) < 0.5) & (np.abs(x) > a + 0.5) & (np.abs(x) < L - 2); k, b = np.polyfit(np.abs(x[S]), phi[S] - t[S], 1)
    o = np.argsort(x[S])
    return {'numbers': [num('slope', 'наклон потенциала по |x| (континуум −0.5)', k, -0.5, 0.05, rel=True)],
            'series': [{'title': 'Потенциал на слое t ≈ 0', 'xlabel': 'x', 'lines': [{'name': 'кристалл', 'x': x[S][o].tolist(), 'y': (phi[S] - t[S])[o].tolist(), 'style': 'dots'}]}],
            'overlay': overlay(w, phi - t, 'потенциал заряда'), 'notes': ['Руками остаётся только величина заряда.']}

@experiment(id='pauli_atom', block='Фермионы и атомы', title='Паули в атоме: два фермиона',
    what='Две одинаковые частицы со спином в поле ядра −α/√(x² + a²), без взаимного отталкивания. Фермионы: «обе внизу» — ноль, низшая пара E₁ + E₂; бозоны — 2E₁.',
    params=[P('alpha', 'α', 0.5, 0.3, 0.7, 0.05), P('L', 'Ромб L', 30.0, 20, 30, 2)],
    presets={'быстро': {'L': 24}, 'как в документе': {'L': 30}}, ref='dirac-doc §4', cost='~0.5 мин')
def exp_pauli_atom(p, region=None):
    from . import fermions as Fe
    L, al = p['L'], p['alpha']; w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Fe.Walk(w); V = lambda xx: -al / np.sqrt(xx * xx + 1.0)
    ph = W.phases(Fe.step_integral(V)); lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9
    sel, tu, proj = Fe.readout(w, ts, L, 8.0); he, ho = np.exp(-x * x / 4), x * np.exp(-x * x / 4); A = {}
    for k, x0 in ((1, -1.5), (2, 1.5)):
        g = np.where(src, np.exp(-(x - x0) ** 2 / 2), 0); aR, aL, _ = W.run(1.0, g, g, ph); A[k] = (proj(he, aR, aL), proj(ho, aR, aL))
    top = lambda s_: max(Fe.energies(tu, s_, 0.3, 1.0), key=lambda z: z[1])[0]
    E1 = top(A[1][0] + A[2][0]); E2 = top(A[1][1] + A[2][1])
    Fee = A[1][0] * A[2][0] - A[2][0] * A[1][0]; Bee = 2 * A[1][0] * A[2][0]; Feo = A[1][0] * A[2][1] - A[2][0] * A[1][1]
    xg, Ec, _, _, _ = Fe.dirac_orbitals(V, W.c, nlev=2)
    return {'numbers': [num('E1', f'E₁ (непрерывный Дирак {Ec[0]:.4f})', E1, Ec[0], 0.05, rel=True), num('E2', f'E₂ (непрерывный {Ec[1]:.4f})', E2, Ec[1], 0.05, rel=True),
                        num('zero', 'фермионы «обе внизу»: |амплитуда|', float(np.abs(Fee).max()), 0.0, 1e-12, fmt='{:.1e}'),
                        num('Bee', 'бозоны «обе внизу»: энергия пары (ждём 2E₁)', top(Bee) if False else max(Fe.energies(tu, Bee, 0.6, 2.2), key=lambda z: z[1])[0], 2 * E1, 0.01, rel=True),
                        num('Feo', 'фермионы, низшая пара (ждём E₁ + E₂)', max(Fe.energies(tu, Feo, 0.6, 2.2), key=lambda z: z[1])[0], E1 + E2, 0.015, rel=True)],
            'series': [], 'notes': ['Взаимное отталкивание не включено (см. «гелий»).', 'Нижний уровень чётный, следующий нечётный: чтение формами e^{−x²/4} и x·e^{−x²/4}.']}

@experiment(id='helium_field', block='Фермионы и атомы', title='«Гелий»: отталкивание из поля зарядов',
    what='Ядро +2 и два электрона действуют друг на друга только линейным полем из порядка: ядро (Ze²/2)|x|, электроны −(e²/2)|Δx|. Двухчастичное блуждание по нитям; сдвиг энергии пары от отталкивания против первого порядка J − K непрерывного Дирака.',
    params=[P('e2', 'e²', 0.1, 0.05, 0.2, 0.01), P('L', 'Ромб L', 24.0, 16, 30, 2)],
    presets={'быстро': {'L': 16}, 'как в документе': {'L': 24}}, ref='dirac-doc §7 (−0.141 / −0.122 против −0.142)', cost='~0.5–2 мин')
def exp_helium_field(p, region=None):
    from . import fermions as Fe
    L, e2 = p['L'], p['e2']; Zn = 2.0; al = Zn * e2 / 2; be = e2 / 2; w = Wd.gold11(4.0, L); W = Fe.Walk(w); V = lambda xx: al * np.abs(xx)
    ph = W.phases(Fe.step_integral(V)); lv = np.unique(np.round(w.t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]
    xg, Ec, shapes, dens, vec = Fe.dirac_orbitals(V, W.c, nlev=2, locmax=30.0); h = xg[1] - xg[0]
    he = lambda xx: np.interp(xx, xg, shapes[0]); ho = lambda xx: np.interp(xx, xg, shapes[1])
    Ux = -be * np.abs(xg[:, None] - xg[None, :]); N = len(xg); p1, p2 = vec[:, 0], vec[:, 1]
    dd = lambda u, v: (np.conj(u[:N]) * v[:N] + np.conj(u[N:]) * v[N:]) / h
    J = float(np.real(dd(p1, p1) @ Ux @ dd(p2, p2)) * h * h); K = float(np.real(dd(p1, p2) @ Ux @ dd(p2, p1)) * h * h)
    out = {}
    for nm, U in (('0', None), ('U', lambda d: -be * np.abs(d))):
        tt, s1, s2 = Fe.pair_run(W, ph, 1.0, ts, L, U, he, ho); out[nm] = (max(Fe.energies(tt, s1, 1.8, 3.6, npts=3601), key=lambda z: z[1])[0], float(np.abs(s2).max() / np.abs(s1).max()))
    return {'numbers': [num('E0', f'пара без отталкивания (непрерывный {Ec[0] + Ec[1]:.4f})', out['0'][0], Ec[0] + Ec[1], 0.03, rel=True),
                        num('shift', f'сдвиг от отталкивания (первый порядок J − K = {J - K:+.4f})', out['U'][0] - out['0'][0], J - K, 0.15, rel=True),
                        num('zero', '«обе внизу» / основной сигнал', out['U'][1], 0.0, 1e-6, fmt='{:.1e}')],
            'series': [], 'notes': [f'J = {J:+.4f} (прямое), K = {K:+.4f} (обмен). Чтение — формой непрерывных орбиталей (фильтр, частоту не задаёт).']}

@experiment(id='lithium_scf', block='Фермионы и атомы', title='«Литий»: самосогласованное поле',
    what='Ядро +3, три электрона, поле только из порядка. Метод Хартри: каждый электрон в среднем поле остальных; уровни — из блуждания, орбитали — Фурье отклика. Опора — тот же метод в непрерывном Дираке.',
    params=[P('NIT', 'Итераций', 10, 4, 16, 1), P('L', 'Ромб L', 30.0, 24, 30, 2)],
    presets={'быстро': {'NIT': 6}, 'как в документе': {'NIT': 14}}, ref='dirac-doc §9 (3.7441 против 3.7927)', cost='~3–8 мин')
def exp_lithium_scf(p, region=None):
    from . import fermions as Fe
    L = p['L']; NIT = int(p['NIT']); e2, Zn, Ne, m = 0.1, 3.0, 3, 1.0; w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Fe.Walk(w)
    xb = np.linspace(-40, 40, 641); dxb = xb[1] - xb[0]; VH_of = lambda rho: -(e2 / 2) * (np.abs(xb[:, None] - xb[None, :]) @ rho) * dxb
    lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9
    sel = (t > ts + 2) & (t < L - 4) & (np.abs(x) < 12); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
    win = np.sin(np.pi * (t[sel] - t[sel].min()) / np.ptp(t[sel])) ** 2
    def cont(Vtab):
        xg, Ec, shapes, dens, _ = Fe.dirac_orbitals(lambda xx: np.interp(xx, xb, Vtab), W.c, nlev=3, N=384, Lg=48.0)
        return xg, Ec, shapes, [np.interp(xb, xg, d) for d in dens]
    def cryst(Vtab):
        ph = W.phases(Fe.step_integral(lambda xx: np.interp(xx, xb, Vtab), k=6)); xg, Ec, shapes, _ = cont(Vtab); Es, dens = [], []
        for k in range(3):
            g = np.interp(x, xg, shapes[k]); gs = np.where(src, g, 0); aR, aL, _ = W.run(m, gs, gs, ph); psi = (aR + aL)[sel]
            s_ = np.zeros(len(tu), complex); np.add.at(s_, inv, g[sel] * psi); pk = Fe.energies(tu, s_, 0.6, 2.2)
            E = min(pk, key=lambda z: abs(z[0] - Ec[k]))[0]; Es.append(E)
            b = np.clip(np.round((x[sel] - xb[0]) / dxb).astype(int), 0, len(xb) - 1); prof = np.zeros(len(xb), complex); cnt = np.zeros(len(xb))
            np.add.at(prof, b, win * psi * np.exp(1j * E * t[sel])); np.add.at(cnt, b, win); d = np.convolve(np.abs(prof / np.maximum(cnt, 1e-9)) ** 2, np.ones(5) / 5, 'same'); dens.append(d / (d.sum() * dxb))
        return Es, dens
    Vnuc = (Zn * e2 / 2) * np.abs(xb); out = {}
    for nm, solver in (('кристалл', cryst), ('непрерывный', lambda V_: cont(V_)[1:4:2])):
        Veff = Vnuc.copy(); hist = []
        for it in range(NIT):
            Es, dens = solver(Veff); rho = sum(dens); VH = VH_of(rho) * (Ne - 1) / Ne; hist.append(sum(Es) - 0.5 * (rho @ VH) * dxb); Veff = 0.5 * Veff + 0.5 * (Vnuc + VH)
        out[nm] = (hist, Es)
    return {'numbers': [num('Ec', f'полная энергия, кристалл (непрерывный {out["непрерывный"][0][-1]:.4f})', out['кристалл'][0][-1], out['непрерывный'][0][-1], 0.03, rel=True),
                        num('conv', 'сходимость: последний шаг', abs(out['кристалл'][0][-1] - out['кристалл'][0][-2]), 0.0, 2e-3, fmt='{:.1e}')],
            'series': [{'title': 'Итерации самосогласования', 'xlabel': 'итерация', 'lines': [{'name': nm, 'x': list(range(1, NIT + 1)), 'y': out[nm][0]} for nm in out]}],
            'notes': [f'Уровни в конце: кристалл {", ".join(f"{e:.4f}" for e in out["кристалл"][1])}; непрерывный {", ".join(f"{e:.4f}" for e in out["непрерывный"][1])}.', 'Хартри (без обмена); вклад отталкивания различается сильнее, чем полная энергия (−0.51 против −0.39) — открыто.']}

@experiment(id='onerule', block='Фермионы и атомы', title='Одно правило на разных уровнях',
    what='Подкристалл «тень в r/τ» — тот же кристалл ×τ внутри мелкого. Амплитуды мелкого уровня между событиями крупного: снова ли правило Джонстона? Без массы — точно; с массой — нет: масса ломает самоподобие. Контроль — случайное прореживание.',
    params=[P('m', 'Масса m', 0.5, 0.0, 1.0, 0.25), P('L', 'Ромб L', 12.0, 8, 12, 2)],
    presets={'быстро': {'L': 10}, 'как в документе': {'L': 12}}, ref='git-dictionary-doc (без массы 10⁻¹⁵; m = 0.5: остаток 0.25)', cost='~10–30 с')
def exp_onerule(p, region=None):
    from scipy.optimize import least_squares
    from . import core
    L, rho, m = p['L'], 4.0, p['m']; r = core.r_for_density(rho, 2)
    c, cp = core.numbers(-L / core.ST, L / core.ST, -r / core.SI, r / core.SI); d, dp = core.numbers(-L, L, -r, r)
    i, j = np.nonzero((cp * core.SI)[:, None] ** 2 + dp[None, :] ** 2 <= r * r); X = np.stack([(c * core.ST)[i], d[j]], 1); sh = (cp * core.SI)[i] ** 2 + dp[j] ** 2
    keep = np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L; X, sh = X[keep], sh[keep]; o = np.argsort(X[:, 0], kind='stable'); X, sh = X[o], sh[o]
    coarse = sh <= (r / core.TAU) ** 2 + 1e-12; Phi = 0.5 * Cs.causal(X).astype(float); n = len(X); b = -m * m / rho
    K = np.linalg.solve((np.eye(n) - b * Phi).T, Phi.T).T
    def fit(idx):
        Ks = K[np.ix_(idx, idx)]; Pc = 0.5 * Cs.causal(X[idx]).astype(float); iu = np.triu_indices(len(idx), 1)
        f = lambda q: (Ks - q[0] * Pc - q[1] * (Pc @ Ks))[iu]; sol = least_squares(f, [1.0, -0.1]); return sol.x, np.linalg.norm(f(sol.x)) / np.linalg.norm(Ks[iu])
    (z, bb), res = fit(np.nonzero(coarse)[0]); (zr, br), resr = fit(np.sort(np.random.default_rng(5).choice(n, coarse.sum(), replace=False)))
    rho_c = rho / core.TAU ** 2
    return {'numbers': [num('ratio', 'событий мелкий / крупный (τ² = 2.618)', n / coarse.sum(), core.TAU ** 2, 0.02, rel=True),
                        num('res', 'остаток правила на крупном уровне (золото)', res, 0.0 if m == 0 else None, 1e-10, fmt='{:.2e}'),
                        num('resr', 'то же, случайное прореживание', resr, fmt='{:.2e}'),
                        num('mprime', 'масса на крупном уровне m′', float(np.sqrt(max(-bb * rho_c, 0))), m if m > 0 else None, 0.05, rel=True)],
            'series': [], 'notes': ['Без массы правило «½ за отношение» одинаково на любом уровне (и любом подмножестве); масса — то, что не повторяется.']}


@experiment(id='whytau', block='Мир, свет, масса', title='Почему именно золотое сечение',
    what='То же правило «тень в шаре», но с другим числом вместо τ: дроби Фибоначчи (периодический мир) или серебряное/бронзовое. Буст η = arcch θ переводит мир в себя точно только при τ: лишь тогда он вращает тень (условие — само уравнение τ² = τ + 1), и круглое окно сохраняется.',
    params=[P('number', 'Число', 'τ', choices=['τ', '3/2', '5/3', '8/5', '13/8', '1+√2', '(3+√13)/2']), P('L', 'Ромб L', 12.0, 8, 16, 2)],
    presets={'быстро': {}, 'как в документе': {}}, ref='golden-doc (добавление 15 окт.): τ 0.99, серебро 0.59, дроби и бронза 0.01', cost='секунды')
def exp_whytau(p, region=None):
    th, thc = {'τ': ((1 + 5 ** .5) / 2, -2 / (1 + 5 ** .5)), '3/2': (1.5, -2 / 3), '5/3': (5 / 3, -0.6), '8/5': (1.6, -0.625), '13/8': (13 / 8, -8 / 13),
               '1+√2': (1 + 2 ** .5, 1 - 2 ** .5), '(3+√13)/2': ((3 + 13 ** .5) / 2, (3 - 13 ** .5) / 2)}[p['number']]
    w, sh = Wd.unit_crystal(th, thc, 4.0, p['L']); X = w.X; eta = math.acosh(th); ch, sn = math.cosh(eta), math.sinh(eta)
    inner = np.abs(X[:, 0]) + np.abs(X[:, 1]) < 0.35 * p['L']; P_ = X[inner]; Q = np.c_[P_[:, 0] * ch - P_[:, 1] * sn, -P_[:, 0] * sn + P_[:, 1] * ch]
    S = {tuple(q) for q in np.round(X, 6)}; frac = float(np.mean([tuple(q) in S for q in np.round(Q, 6)]))
    I, J = Cs.links_dense(X); a, b = np.array([-0.8 * p['L'], 0.0]), np.array([0.8 * p['L'], 0.0]); dA = X - a; dB = b - X
    ins = (dA[:, 0] > np.abs(dA[:, 1])) & (dB[:, 0] > np.abs(dB[:, 1])); d, f = Cs.mm_dimension(X[ins])
    gold = p['number'] == 'τ'
    return {'numbers': [num('n', 'событий', w.n, fmt='{:.0f}'), num('dim', 'размерность Майрхейма–Мейера', d, 2.0, 0.05),
                        num('boost', 'доля событий, которые буст η = arcch θ переводит в события', frac, 1.0 if gold else None, 0.02, fmt='{:.2f}'),
                        num('cond', 'условие вращения тени: 1/θ + 1/θ² (= 1 только у τ)', 1 / th + 1 / th ** 2, 1.0 if gold else None, 1e-9, fmt='{:.4f}')],
            'overlay': overlay(w, (sh <= (w.meta['window_r'] / th) ** 2).astype(float), 'крупный уровень (тень в r/θ)'), 'series': [],
            'notes': ['Синглет Белла и размерность 2 — у всех чисел; скорость нитей ≈ 0.75 и сдвиг массы кристалла от числа не зависят (golden-doc, 15 окт.).']}

@experiment(id='nonull', block='Мир, свет, масса', title='Световых пар нет: наименьший интервал',
    what='Если интервал пары равен нулю, то по Галуа равна нулю и её «тень» — а тень у разности двух событий нулём быть не может. Значит, в золотом кристалле нет ни одной точно световой пары (ни в 1+1, ни в 3+1), а интервалы принимают дискретный набор значений с наименьшим ≥ 1/(4r²).',
    params=[P('rho', 'ρ', 4.0, 2, 8, 1), P('L', 'Ромб L', 12.0, 8, 14, 2)],
    presets={'быстро': {'L': 10}, 'как в документе': {'L': 12}}, ref='dirac-doc §12 (ρ = 4: 0 пар, min q = τ⁻⁵ = 0.0902, оценка 0.039)', cost='секунды')
def exp_nonull(p, region=None):
    from .core import r_for_density
    w = Wd.gold11(p['rho'], p['L']); X = w.X; r = r_for_density(p['rho'], 2); d = X[None] - X[:, None]; q = d[..., 0] ** 2 - d[..., 1] ** 2
    q = q[np.triu_indices(w.n, 1)]; tl = q[q > 1e-12]; vals = np.unique(np.round(tl[tl < 2], 6))
    return {'numbers': [num('zero', 'пар с нулевым интервалом', float(np.sum(np.abs(q) <= 1e-9)), 0.0, 0.5, fmt='{:.0f}'),
                        num('qmin', 'наименьший времениподобный интервал² q', tl.min(), fmt='{:.4f}'), num('bound', 'оценка снизу по норме Галуа 1/(4r²)', 1 / (4 * r * r), fmt='{:.4f}'),
                        num('ok', 'q_min ≥ оценки (1 = да)', float(tl.min() >= 1 / (4 * r * r) - 1e-12), 1.0, 0.5, fmt='{:.0f}'),
                        num('nvals', 'разных значений q в (0, 2) — дискретный набор', float(len(vals)), fmt='{:.0f}')],
            'series': [{'title': 'Значения интервала² (первые) — набор не зависит от размера окна', 'xlabel': 'номер', 'lines': [{'name': 'q', 'x': list(range(len(vals))), 'y': vals.tolist()}]}],
            'notes': ['Вывод: q ∈ Z[τ], N(q) = q·q′ — целое ≠ 0, |q′| ≤ 4r² ⇒ |q| ≥ 1/(4r²). Поэтому «нити» по световым лучам невозможны в принципе.',
                      'Наименьшие значения — степени τ: τ⁻⁵ = 0.0902, 2τ⁻⁵, τ⁻³, 2τ⁻³, τ⁻¹ … (при ρ = 4).']}

@experiment(id='lightspin', block='Фермионы и атомы', title='Спин без нитей: компоненты Дирака из светового уголка',
    what='Компонента Дирака подчиняется уравнению 4D_uD_v + m² — у него функция Грина ровно «световой уголок» Джонстона без вставки напряжённости. Скаляр КГ — тот же уголок плюс вставка −iU′. Зеркальный уголок (сначала вправо, потом влево) — другая компонента. Проверка: в атоме U = F|x| уровни компонент сдвинуты от КГ ровно на спиновую поправку Дирака; безмассовый фронт идёт со скоростью 1 (у нитей 0.76).',
    params=[P('F', 'Сила F', 0.4, 0.2, 0.6, 0.1), P('L', 'Ромб L', 30.0, 26, 34, 2)],
    presets={'быстро': {'L': 26}, 'как в документе': {'L': 30}}, ref='dirac-doc §12 (F = 0.4: сдвиг 0.033–0.048 против 0.040)', cost='~10–20 с')
def exp_lightspin(p, region=None):
    from . import fermions as Fe
    F, L, m = p['F'], p['L'], 1.0; w = Wd.gold11(4.0, L); t, x = w.t, w.x; Pf, dU, U = Fl.vshape(F); ts = -L + 10
    g = lambda s: (1 + s) * np.exp(-s * s / 2); om = np.linspace(m - 0.1, m + 1.2, 2601); E, lines = [], []
    for nm, side, ins in (('КГ: уголок + вставка −iU′', 1, 1.0), ('Дирак: уголок', 1, 0.0), ('Дирак: зеркальный уголок', -1, 0.0)):
        phi = Pr.corner_respond(w.X, Pf, dU, m, 4.0, np.where(np.abs(t - ts) < 0.45, g(x), 0), side, ins)
        sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); S = Ob.spectrum(t[sel], g(x[sel]) * phi[sel], om); S /= S.max()
        E.append(max(Ob.peaks(om, S, 0.05), key=lambda z: z[1])[0]); lines.append({'name': nm, 'x': om[::4].tolist(), 'y': S[::4].tolist()})
    kg = sorted(Ob.kg_levels_linear(F, m, 'even', m + 0.9, n=300) + Ob.kg_levels_linear(F, m, 'odd', m + 0.9, n=300))[0]
    _, ev, _, _, _ = Fe.dirac_orbitals(lambda s: F * np.abs(s), 1.0, m, nlev=1, N=512, Lg=80, locmax=60); dth = ev[0] - kg
    zero = lambda s: 0 * s; f0 = np.where((np.abs(t - ts) < 1e-6) | ((np.abs(t - ts) < 0.45) & (np.abs(x) < 0.6)), np.abs(x) < 0.6, 0).astype(float)
    ph0 = np.abs(Pr.corner_respond(w.X, zero, zero, 0.0, 4.0, f0)); ss, xs = [], []
    for s_ in (4.0, 7.0, 10.0, 13.0):
        b = np.abs(t - (ts + s_)) < 0.3
        if b.any() and ph0[b].max() > 0: on = b & (ph0 > 0.5 * ph0[b].max()); ss.append(s_); xs.append(0.5 * (x[on].max() - x[on].min()))
    c0 = float(np.polyfit(ss, xs, 1)[0])
    return {'numbers': [num('kg', f'основной уровень КГ на кристалле (непрерывный {kg:.4f})', E[0], kg, 0.03, rel=True),
                        num('dR', f'компонента «уголок» (непрерывный Дирак {ev[0]:.4f})', E[1], ev[0], 0.03, rel=True),
                        num('dL', 'компонента «зеркальный уголок»', E[2], ev[0], 0.03, rel=True),
                        num('shift', f'спиновый сдвиг: Дирак − КГ на кристалле (непрерывный {dth:+.4f})', 0.5 * (E[1] + E[2]) - E[0], dth, 0.25, rel=True, fmt='{:+.4f}'),
                        num('lr', 'уголок − зеркальный (обе — один Дирак: 0)', E[1] - E[2], 0.0, 0.01, fmt='{:+.4f}'),
                        num('c0', 'скорость безмассового фронта (нити: 0.76)', c0, 1.0, 0.05, rel=True, fmt='{:.3f}')],
            'series': [{'title': 'Спектры отклика: скаляр и две компоненты Дирака', 'xlabel': 'E', 'lines': lines, 'marks': [kg, ev[0]]}],
            'notes': ['Нитей и монеты нет: обе компоненты — сумма по цепочкам (½ на шаг, −m²/ρ на остановку); спин входит только как ориентация уголка (какой световой луч первый), т.е. через напряжённость поля.',
                      'Связь компонент (ψ_L из ψ_R) — производная, нужна только при чтении (рамка наблюдателя). Разброс уголок/зеркальный ±0.005–0.02 — точность измерителя.']}

@experiment(id='orderatom', block='Фермионы и атомы', title='Атом Дирака целиком из порядка',
    what='Ядро — трубка событий. Его поле в каждой точке — ½ за каждое событие ядра в прошлом конусе (то же правило, что даёт Кулона). Электрон — две компоненты Дирака: сумма по цепочкам с фазой этого поля вдоль светового уголка, без нитей и без монеты. Сравнение с непрерывным Дираком при измеренном наклоне поля; спиновый сдвиг — от скаляра КГ в том же поле.',
    params=[P('F', 'Целевая сила (задаёт заряд g = 2F)', 0.4, 0.2, 0.6, 0.1), P('L', 'Ромб L', 30.0, 26, 34, 2)],
    presets={'быстро': {'L': 26}, 'как в документе': {'L': 30}}, ref='dirac-doc §13 (F = 0.389; сдвиг +0.030…+0.044 против +0.038)', cost='~40–60 с')
def exp_orderatom(p, region=None):
    from . import fermions as Fe
    L, m = p['L'], 1.0; w = Wd.gold11(4.0, L); t, x = w.t, w.x; ts = -L + 10
    Uf, dUev, Uev, (al, be, F) = Fl.order_field11(w, 1.0, 2 * p['F'], (ts, L - 4))
    g = lambda s: (1 + s) * np.exp(-s * s / 2); om = np.linspace(m - 0.1, m + 1.2, 2601); E, lines = [], []
    for nm, side, ins in (('КГ: уголок + вставка −iU′', 1, 1.0), ('Дирак: уголок', 1, 0.0), ('Дирак: зеркальный уголок', -1, 0.0)):
        phi = Pr.corner_respond_general(w.X, Uf, dUev, m, 4.0, np.where(np.abs(t - ts) < 0.45, g(x), 0), side, ins)
        sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); S = Ob.spectrum(t[sel], g(x[sel]) * phi[sel], om); S /= S.max()
        E.append(max(Ob.peaks(om, S, 0.05), key=lambda z: z[1])[0]); lines.append({'name': nm, 'x': om[::4].tolist(), 'y': S[::4].tolist()})
    kg = sorted(Ob.kg_levels_linear(F, m, 'even', m + 0.9, n=300) + Ob.kg_levels_linear(F, m, 'odd', m + 0.9, n=300))[0]
    _, ev, _, _, _ = Fe.dirac_orbitals(lambda s: F * np.abs(s), 1.0, m, nlev=1, N=512, Lg=80, locmax=60); dth = ev[0] - kg
    o = np.argsort(x); band = np.abs(t) < 0.5
    return {'numbers': [num('F', f'наклон поля ядра из порядка (цель {p["F"]})', F, p['F'], 0.05, rel=True),
                        num('al', 'однородная часть α (калибровка; ожидание −F)', al, -F, 0.02, rel=True, fmt='{:+.4f}'),
                        num('kg', f'основной уровень КГ (непрерывный при этом F: {kg:.4f})', E[0], kg, 0.03, rel=True),
                        num('dR', f'Дирак, уголок (непрерывный {ev[0]:.4f})', E[1], ev[0], 0.03, rel=True),
                        num('dL', 'Дирак, зеркальный уголок', E[2], ev[0], 0.03, rel=True),
                        num('shift', f'спиновый сдвиг Дирак − КГ (непрерывный {dth:+.4f})', 0.5 * (E[1] + E[2]) - E[0], dth, 0.25, rel=True, fmt='{:+.4f}')],
            'series': [{'title': 'Спектры: скаляр и две компоненты Дирака в поле из порядка', 'xlabel': 'E', 'lines': lines, 'marks': [kg, ev[0]]},
                       {'title': 'Поле ядра на слое t ≈ 0 (после калибровки)', 'xlabel': 'x', 'lines': [{'name': 'из порядка', 'x': x[o][band[o]].tolist(), 'y': Uev[o][band[o]].tolist(), 'style': 'dots'}]}],
            'overlay': overlay(w, Uev, 'поле ядра из порядка'),
            'notes': ['Поле не вставлено готовым: оно посчитано подсчётом событий ядра в прошлых конусах; зёрна поля остаются в фазе.',
                      'Вычтена только однородная часть α·t + β — функция одного времени, т.е. калибровка (выбор нуля энергии).']}

@experiment(id='heliumorder', block='Фермионы и атомы', title='«Гелий» без нитей: Паули и обмен, всё поле из порядка',
    what='Ядро +2 и два электрона. Поле ядра и поле облака каждого электрона — ½ за событие в прошлом конусе (облако — с весом своей плотности). Электроны — компоненты Дирака по световому уголку. Орбитали вынимаются из самого кристалла. Хартри — замер: уровень одного электрона в поле облака другого. Обмен K — по орбиталям кристалла. Пара фермионов E₀ + E₁ + J − K, бозонов в основном 2E₀ + J₀₀: Паули поднимает пару.',
    params=[P('e2', 'e²', 0.2, 0.1, 0.3, 0.05), P('L', 'Ромб L', 30.0, 26, 34, 4)],
    presets={'быстро': {'L': 26}, 'как в документе': {'L': 30}}, ref='dirac-doc §14 (пара фермионов 2.648 / 2.590 / 2.621 против 2.619)', cost='~2–5 мин')
def exp_heliumorder(p, region=None):
    from . import fermions as Fe
    w = Wd.gold11(4.0, p['L']); r = Fe.helium_order(w, p['e2'], log=lambda *a: None); E0, E1 = r['E']; Ec = r['Ec']
    Jd = 0.5 * ((r['Eh'][0] - E0) + (r['Eh'][1] - E1)); EF = E0 + E1 + Jd - r['K']; EFc = Ec[0] + Ec[1] + r['Jc'] - r['Kc']
    EB = 2 * E0 + r['J00']; EBc = 2 * Ec[0] + r['J00c']
    return {'numbers': [num('E0', f'уровень 0 в поле ядра (непр. {Ec[0]:.4f})', E0, Ec[0], 0.03, rel=True), num('E1', f'уровень 1 (непр. {Ec[1]:.4f})', E1, Ec[1], 0.03, rel=True),
                        num('fid', 'совпадение плотностей орбиталей кристалла с непрерывными (хуже из двух)', min(r['fid']), 1.0, 0.02),
                        num('cloud', 'сила поля облака из порядка (e²/2)', float(np.mean(r['cs'])), p['e2'] / 2, 0.08, rel=True),
                        num('H0', f'Хартри: уровень 0 в поле облака 1 (непр. {r["hc"][0]:.4f})', r['Eh'][0], r['hc'][0], 0.03, rel=True),
                        num('H1', f'Хартри: уровень 1 в поле облака 0 (непр. {r["hc"][1]:.4f})', r['Eh'][1], r['hc'][1], 0.03, rel=True),
                        num('J', f'прямое J по орбиталям кристалла (непр. {r["Jc"]:+.4f})', r['J'], r['Jc'], 0.15, rel=True, fmt='{:+.4f}'),
                        num('K', f'обмен K по орбиталям кристалла (непр. {r["Kc"]:+.4f})', r['K'], r['Kc'], 0.2, rel=True, fmt='{:+.4f}'),
                        num('EF', f'пара фермионов E₀ + E₁ + J(дин.) − K (непр. {EFc:.4f})', EF, EFc, 0.02, rel=True),
                        num('EB', f'бозоны в основном 2E₀ + J₀₀ — запрещено Паули (непр. {EBc:.4f})', EB, EBc, 0.02, rel=True),
                        num('pauli', f'подъём пары Паули: фермионы − бозоны (непр. {EFc - EBc:.4f})', EF - EB, EFc - EBc, 0.15, rel=True)],
            'series': [{'title': 'Спектры: в поле ядра; уровень 0 в поле облака 1; уровень 1 в поле облака 0', 'xlabel': 'E',
                        'lines': [{'name': nm, 'x': r['om'][::4].tolist(), 'y': S[::4].tolist()} for nm, S in zip(('ядро', 'ядро + облако 1', 'ядро + облако 0'), r['spec'])], 'marks': [Ec[0], Ec[1], r['hc'][0], r['hc'][1]]}],
            'notes': ['Замер: уровни и сдвиги Хартри (динамика на кристалле). Счёт по ингредиентам кристалла: J, K, J₀₀ (орбитали кристалла × ядро −(e²/2)|Δx|).',
                      'Обмен как динамика (двухчастичная сумма по цепочкам) пока не сделан — K берётся по орбиталям кристалла, первый порядок.']}

@experiment(id='pairexchange', block='Фермионы и атомы', title='Обмен как динамика: пара с взаимодействием, все порядки',
    what='Два электрона в поле ядра (из порядка), без нитей. Двухчастичная амплитуда на парах событий с вершиной i·λ·[события несравнимы] (ковариантная, из порядка) — все порядки прямой развёрткой по слоям, без итераций. Сдвиг пика пары при ±λ в трёх каналах: фермионы, бозоны в тех же орбиталях, бозоны оба внизу. Нормировка вершины неизвестна — сравниваются отношения: фермионы/бозоны = (J − K)/(J + K), оба внизу/бозоны = J₀₀/(J + K).',
    params=[P('L', 'Ромб L', 24.0, 24, 28, 2)],
    presets={'быстро': {'L': 24}, 'как в документе': {'L': 26}}, ref='dirac-doc §15 (Ф/Б 1.64 / 1.73 / 1.39; ББ/Б 0.84 / 0.88 / 0.93; теория 1.747 и 0.897)', cost='~4–8 мин')
def exp_pairexchange(p, region=None):
    from . import fermions as Fe
    w = Wd.gold11(4.0, p['L']); r = Fe.pair_exchange(w, log=lambda *a: None); J, K, J00, Kr = r['J'], r['K'], r['J00'], r['Kret']
    rFB = r['F'][1] / r['B'][1]; rBB = r['BB'][1] / r['B'][1]; Kfb = -J * (rFB - 1) / (rFB + 1); Kbb = J00 / rBB - J
    return {'numbers': [num('pF', f'пик пары фермионов без взаимодействия (E₀ + E₁ = {r["F"][2]:.4f})', r['F'][0], r['F'][2], 0.01, rel=True),
                        num('pBB', f'пик пары «оба внизу» (2E₀ = {r["BB"][2]:.4f})', r['BB'][0], r['BB'][2], 0.015, rel=True),
                        num('rFB', f'фермионы/бозоны: сдвиги (теория мгновенная {(J - K) / (J + K):.3f}, с запаздыванием {(J - Kr) / (J + Kr):.3f})', rFB, (J - K) / (J + K), 0.12, rel=True),
                        num('rBB', f'оба внизу/бозоны (теория {J00 / (J + K):.3f})', rBB, J00 / (J + K), 0.08, rel=True),
                        num('Kfb', f'обмен K из Ф/Б (непр. {K:+.4f}; с запаздыванием {Kr:+.4f})', Kfb, K, 0.3, rel=True, fmt='{:+.4f}'),
                        num('Kbb', 'обмен K из ББ/Б', Kbb, K, 0.3, rel=True, fmt='{:+.4f}'),
                        num('sB', 'наклон dE/dλ, бозоны (нормировка вершины не выведена)', r['B'][1], fmt='{:+.2f}')],
            'series': [], 'notes': ['Знак обмена верный во всех окнах: фермионная пара сдвигается сильнее бозонной в тех же орбиталях (K > 0).',
                                    'Открыто: абсолютная нормировка вершины; мгновенная или запаздывающая (разница 15 % — на пределе точности); разброс между окнами ±10–15 %.']}

@experiment(id='pairnorm', block='Фермионы и атомы', title='Нормировка вершины сшивкой и «гелий» всеми порядками',
    what='Вес вершины пары не подгоняется, а сшивается с одночастичным счётом: той же машиной мерится «вес» каждой частицы (во сколько раз постоянная вставка сдвигает её уровень), и λ = (e²/2)/(2·w_a·w_b). Затем пара считается двигателем эволюции всеми порядками и сравнивается с непрерывной задачей двух тел (конфигурационное взаимодействие).',
    params=[P('L', 'Ромб L', 24.0, 24, 26, 2)],
    presets={'быстро': {'L': 24}, 'как в документе': {'L': 24}}, ref='dirac-doc §16 (фермионы 2.640 / 2.616 против 2.582; бозоны 2.382 / 2.402 против 2.405)', cost='~3–5 мин')
def exp_pairnorm(p, region=None):
    from . import fermions as Fe
    w = Wd.gold11(4.0, p['L']); r = Fe.pair_physical(w, log=lambda *a: None); E0c, E1c, EFc, EBc, EF1, EB1 = r['ci']
    return {'numbers': [num('wa', 'вес частицы a (dE/dδb)', r['w'][0], fmt='{:+.3f}'), num('wb', 'вес частицы b', r['w'][1], fmt='{:+.3f}'),
                        num('EF', f'пара фермионов всеми порядками (непрерывная задача двух тел {EFc:.4f}; первый порядок {EF1:.4f})', r['F'][1], EFc, 0.03, rel=True),
                        num('dF', f'сдвиг пары фермионов (непр. {EFc - (E0c + E1c):+.4f})', r['F'][1] - r['F'][0], EFc - (E0c + E1c), 0.2, rel=True, fmt='{:+.4f}'),
                        num('EB', f'бозоны оба внизу всеми порядками (непр. {EBc:.4f})', r['BB'][1], EBc, 0.03, rel=True),
                        num('dB', f'сдвиг бозонов (непр. {EBc - 2 * E0c:+.4f})', r['BB'][1] - r['BB'][0], EBc - 2 * E0c, 0.2, rel=True, fmt='{:+.4f}'),
                        num('pauli', f'подъём Паули: фермионы − бозоны (непр. {EFc - EBc:.4f})', r['F'][1] - r['BB'][1], EFc - EBc, 0.15, rel=True)],
            'series': [], 'notes': ['Нормировка выведена сшивкой двух счётов (вставка на остановках у одной частицы и пары несравнимых событий у двух) — не подгонкой под ответ.',
                                    'Остаётся: вес зависит от пары состояний (w_a w_b); вершина, одинаковая для всех состояний, должна прийти из калибровочной связи (закон площади).']}

@experiment(id='vectorvertex', block='Фермионы и атомы', title='Одна вершина для всех состояний: векторная связь',
    what='Вес частицы оказался w = −ρ/(2(E − U)) — множитель «энергия минус потенциал», как у заряда, связанного с полем. Отсюда одна вершина пары без измеренных чисел: W = i·(e²/ρ²)·(E − U(x₁))(E − U(x₂))·[несравнимы]. Три канала (фермионы, бозоны в тех же орбиталях, бозоны оба внизу) одной формулой против непрерывной задачи двух тел.',
    params=[P('L', 'Ромб L', 24.0, 24, 26, 2)],
    presets={'быстро': {'L': 24}, 'как в документе': {'L': 24}}, ref='dirac-doc §17 (сдвиги 0.84 / 0.89 / 0.78 и 0.89 / 0.99 / 0.82 от непрерывных)', cost='~3–5 мин')
def exp_vectorvertex(p, region=None):
    from . import fermions as Fe
    w = Wd.gold11(4.0, p['L']); r = Fe.pair_vector(w, log=lambda *a: None); nums = []
    for key, lab in (('F', 'фермионы a∧b'), ('B', 'бозоны a∨b'), ('BB', 'бозоны оба внизу')):
        e0, e1, ref, base = r[key]
        nums += [num(f'E{key}', f'{lab}: энергия пары (непрерывная задача двух тел {ref:.4f})', e1, ref, 0.03, rel=True),
                 num(f'r{key}', f'{lab}: сдвиг / непрерывный сдвиг', (e1 - e0) / (ref - base), 1.0, 0.2, fmt='{:.2f}')]
    return {'numbers': nums, 'series': [], 'notes': ['Вершина одна для всех каналов, нормировка из e² и ρ — без измеренных весов (ср. `pairnorm`).',
                                                    'Приближение «полюсное»: вместо оператора i∂_t берётся энергия канала. Производная по столбцам кристалла не работает (шаги до 3.3).',
                                                    'Недобор 10–15 % у каналов с двумя орбиталями не уходит с плотностью (`pairensemble`); прежнее объяснение «перенормировкой веса» отозвано.']}

@experiment(id='lithiumnew', block='Фермионы и атомы', title='«Литий» на новой основе: три электрона без нитей',
    what='Ядро +3, поле из порядка; три электрона — компоненты Дирака по световому уголку. Энергия = сумма трёх уровней кристалла + сдвиги трёх пар, каждая посчитана двигателем эволюции с векторной вершиной (все порядки внутри пары). Эталон — непрерывная задача трёх тел без пар (конфигурационное взаимодействие) и её парное приближение.',
    params=[P('L', 'Ромб L', 24.0, 24, 26, 2)],
    presets={'быстро': {'L': 24}, 'как в документе': {'L': 24}}, ref='dirac-doc §18 (4.500 / 4.234 против 4.246 трёх тел и 4.337 парного)', cost='~4–6 мин')
def exp_lithiumnew(p, region=None):
    from . import fermions as Fe
    w = Wd.gold11(4.0, p['L']); r = Fe.lithium_pairs(w, log=lambda *a: None)
    nums = [num(f'E{k}', f'уровень {k} кристалла (непр. {r["E"][k]:.4f})', r['Ec'][k], r['E'][k], 0.03, rel=True) for k in range(3)]
    nums += [num(f'p{i}{j}', f'сдвиг пары {i}{j} (непр. {r["pc"][(i, j)]:+.4f})', r['sh'][(i, j)], r['pc'][(i, j)], 0.3, rel=True, fmt='{:+.4f}') for (i, j) in ((0, 1), (0, 2), (1, 2))]
    nums += [num('Etot', f'«литий»: полная энергия (непр. трёх тел {r["E3"]:.4f}; парное приближение {r["Epair"]:.4f})', r['Etot'], r['E3'], 0.05, rel=True)]
    return {'numbers': nums, 'series': [], 'notes': ['Парное приближение: эффекты трёх тел (перестройка орбиталей третьим электроном) в непрерывном −0.09 — не учтены.',
                                                    'Разброс сдвигов пар между окнами ±20 % — предел нынешнего измерителя пары.']}

@experiment(id='eos', block='Гравитация среза и цвет', title='Этапы вселенной: уравнение состояния из порядка',
    what='Посыпки в расширяющейся вселенной 3+1 с разными этапами: излучение (w = 1/3), жёсткая материя (w = 1), экспонента (w = −1/3), тёмная энергия — де Ситтер (w = −1), и плоский контроль. Из порядка мерятся только счёты в причинных ромбах (событий, связанных пар, троек) и сравниваются с точным откликом; подгоняется одно число w.',
    params=[P('arm', 'Этап', 'все', choices=['все', 'lin', 'root', 'exp', 'ds', 'flat']), P('n', 'Точек квадратуры (тыс.)', 300.0, 50, 600, 50)],
    presets={'быстро': {'n': 150.0}, 'как в документе': {'n': 600.0}}, ref='weos-doc, добавление 16 окт. (опора 600 тыс.: +0.371 / +1.18 / −0.330 / −1.07; плоское χ² 6.1)', cost='~8–15 мин')
def exp_eos(p, region=None):
    from . import eos as Eo
    arms = Eo.load_arms(); todo = list(arms) if p['arm'] == 'все' else [p['arm']]; nums, lines = [], []
    for fam in todo:
        r = Eo.measure(fam, arms, n=int(p['n'] * 1000))
        if r['wtrue'] is None:
            nums.append(num(f'chi_{fam}', f'{Eo.NAMES[fam]}: χ²/степень лучшей подгонки (> 3 — отвергнуто)', r['chi2'] / r['dof'], fmt='{:.2f}'))
        else:
            nums.append(num(f'w_{fam}', f'{Eo.NAMES[fam]}: w из порядка (± {r["werr"]:.3f}, {r["worlds"]} миров)', r['w'], r['wtrue'], max(2 * r['werr'], 0.05), fmt='{:+.3f}'))
            nums.append(num(f'chi_{fam}', f'{Eo.NAMES[fam]}: χ²/степень', r['chi2'] / r['dof'], 1.0, 1.5, fmt='{:.2f}'))
        lines.append({'name': Eo.NAMES[fam], 'x': r['grid'], 'y': [v / r['dof'] for v in r['curve']]})
    return {'numbers': nums, 'series': [{'title': 'χ²/степень по сетке параметра семейства (минимум — измеренное значение)', 'xlabel': 'параметр (p или w)', 'lines': lines}],
            'notes': ['Допущения: постоянное w на этапе, известный темп расширения h в середине, конформно-плоская вселенная (как в weos-doc).',
                      'Плечи 1/3, 1, −1/3 — наборы 5–9 окт.; де Ситтер — новый набор (200 миров).']}

@experiment(id='pairensemble', block='Фермионы и атомы', title='Точность измерителя пары: ансамбль сдвигов фазона',
    what='Тот же закон мира, но шар тени сдвинут («сдвиг фазона») — другие события, честный ансамбль. На каждом мире пара в поле ядра считается двигателем эволюции с векторной вершиной; сдвиги трёх каналов усредняются. Ошибка среднего падает как 1/√(число миров).',
    params=[P('nw', 'Миров в ансамбле', 6.0, 3, 8, 1), P('rho', 'Плотность ρ', 4.0, 4, 6, 2)],
    presets={'быстро': {'nw': 3.0}, 'как в документе': {'nw': 6.0}}, ref='dirac-doc §19 (ρ = 4, 6 миров: Ф −0.265 ± 0.004, Б −0.177 ± 0.008, ББ −0.150 ± 0.009; Ф/Б 1.505 ± 0.050)', cost='~8–20 мин')
def exp_pairensemble(p, region=None):
    from . import fermions as Fe
    r = Fe.pair_ensemble(p['rho'], 24.0, int(p['nw']), log=lambda *a: None); nums = []
    for k, lab in (('F', 'фермионы a∧b'), ('B', 'бозоны a∨b'), ('BB', 'бозоны оба внизу')):
        m_, e_, ref = r[k]
        nums += [num(f'd{k}', f'{lab}: средний сдвиг (± {e_:.4f}; непрерывная задача двух тел {ref:+.4f})', m_, ref, 0.15, rel=True, fmt='{:+.4f}'),
                 num(f'r{k}', f'{lab}: среднее / непрерывное', m_ / ref, 1.0, 0.15, fmt='{:.3f}')]
    nums.append(num('FB', f'фермионы / бозоны (± {r["FB"][1]:.3f}; непр. все порядки 1.577, первый порядок с запаздыванием 1.464)', r['FB'][0], 1.577, 0.08, rel=True, fmt='{:.3f}'))
    return {'numbers': nums, 'series': [], 'notes': ['Недобор живёт в каналах с двумя разными орбиталями и не уходит с плотностью (ρ = 4 → 6) — это не зерно кристалла, а приближение вершины.']}

@experiment(id='spin31', block='Поле, атом, водород', title='Атом со спином в 3+1 и предел зерна',
    what='Спинор Вейля в 3+1 подчиняется Клейну–Гордону с членом Паули: B = −(m² ∓ iσ·∇U)/ρ на каждом событии, Джонстон по связям, нить Вильсона. '
         'При нескольких плотностях: темп роста отклика (неустойчивость при m ~ ρ^¼), наклон уровня по массе dE/dm², знак второго порядка (вставка z·e^{−r/2}) '
         'и спиновый сдвиг 1s Дирак − КГ. Эталон — непрерывная теория ТЕМ ЖЕ прибором (окно, Ханн, вес e^{−αr}).',
    params=[P('rmax', 'Наибольшая плотность ρ', 4.0, 1, 4, 1), P('H', 'Окно H', 12.0, 12, 12, 1)],
    presets={'быстро': {'rmax': 2.0}, 'как в документе': {'rmax': 4.0}}, ref='dirac-doc §20 (ρ = 1 / 2 / 4: рост +0.22 / +0.06 / 0.00; спин −0.0004 / +0.0042 / +0.0039; непрерывная тем же прибором +0.0068)', cost='~1 мин (быстро), ~4 мин (документ)')
def exp_spin31(p, region=None):
    H, Rc, al, a, m = p['H'], 5.0, 0.45, 0.5, 1.0
    REF = {'E': 0.89899, 'slope': 0.511, 'stark': -0.00259, 'spin': +0.00682}          # непрерывная КГ / Дирак тем же прибором, H = 12, шар 5
    sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex); e = np.array([1, 0], complex)
    nums, rows = [], []
    for rho in [r_ for r_ in (1.0, 2.0, 4.0, 8.0) if r_ <= p['rmax']]:
        w = Wd.gold31(rho, 'cylinder', H=H, Rc=Rc, Rs=3.0); X = w.X; I, J = Cs.links_sparse(X, rho, 1.6)
        r = np.linalg.norm(X[:, 1:], axis=1); t = X[:, 0]; prof = np.exp(-al * r); src = np.where((t < 0.45)[:, None], prof[:, None] * e[None, :], 0)
        ph = Pr.wilson31(X, I, J, lambda q: -al / np.sqrt(q * q + a * a)); sel = (t > 1.5) & (t < H - 0.3); om = np.linspace(0.5, 1.3, 1601)
        def go(Bx):
            sig = Pr.johnston31_respond_spinor(w.n, I, J, rho, m, src, ph, Bx) @ e
            S = Ob.spectrum(t[sel], prof[sel] * sig[sel], om, 'hann'); pk = Ob.peaks(om, S, 0.0); return pk[int(np.argmax([h for _, h in pk]))][0], sig
        eye = np.eye(2)[None]; E1, sig = go(-m * m / rho * eye + 0 * r[:, None, None]); E2, _ = go(-0.81 / rho * eye + 0 * r[:, None, None])
        ed = np.arange(0, H + 1e-9, 1.0); k = np.digitize(t, ed) - 1; ok = (k >= 0) & (k < len(ed) - 1)
        amp = np.sqrt(np.bincount(k[ok], np.abs(prof * sig)[ok] ** 2, minlength=len(ed) - 1) / np.maximum(np.bincount(k[ok], minlength=len(ed) - 1), 1))
        tc = 0.5 * (ed[1:] + ed[:-1]); s_ = (tc > 2) & (tc < H - 1); gr = np.polyfit(tc[s_], np.log(amp[s_] + 1e-300), 1)[0]
        g = X[:, 3] * np.exp(-r / 2.0); c = 0.12 / np.abs(g).max(); Es, _ = go((-m * m + c * g)[:, None, None] * eye / rho)
        rh = np.where(r[:, None] > 1e-9, X[:, 1:] / np.maximum(r, 1e-9)[:, None], 0); sr = rh[:, 0, None, None] * sx + rh[:, 1, None, None] * sy + rh[:, 2, None, None] * sz
        dU = al * r / (r * r + a * a) ** 1.5; Ed, _ = go((-m * m * eye + 1j * dU[:, None, None] * sr) / rho)
        sl = (E1 - E2) / 0.19; rows.append((rho, E1, gr, sl, Es - E1, Ed - E1)); tag = f'ρ = {rho:g}'
        nums += [num(f'E{rho:g}', f'{tag}: 1s КГ (непрерывная тем же прибором {REF["E"]:.4f})', E1, REF['E'], 0.04, rel=True),
                 num(f'g{rho:g}', f'{tag}: темп роста отклика (0 — устойчиво)', gr, 0.0, 0.05, fmt='{:+.3f}'),
                 num(f'sl{rho:g}', f'{tag}: наклон dE/dm² (тем же прибором {REF["slope"]:.3f})', sl, REF['slope'], 0.3, rel=True, fmt='{:.3f}'),
                 num(f'st{rho:g}', f'{tag}: второй порядок, вставка z (тем же прибором {REF["stark"]:+.5f}; знак обязан быть минус)', Es - E1, REF['stark'], 0.6, rel=True, fmt='{:+.5f}'),
                 num(f'sp{rho:g}', f'{tag}: спиновый сдвиг Дирак − КГ (тем же прибором {REF["spin"]:+.5f})', Ed - E1, REF['spin'], 0.5, rel=True, fmt='{:+.5f}')]
    xs = [r_[0] for r_ in rows]
    return {'numbers': nums, 'series': [{'title': 'Спиновый сдвиг 1s (Дирак − КГ) по плотности; непрерывная тем же прибором +0.0068', 'xlabel': 'ρ', 'lines': [
                {'name': 'кристалл', 'x': xs, 'y': [r_[5] for r_ in rows]}, {'name': 'непрерывная', 'x': xs, 'y': [REF['spin']] * len(xs)}]}],
            'notes': ['При ρ = 1 масса m = 1 равна масштабу зерна (m ≈ ρ^¼): отклик растёт со временем, уровень почти не слушает массу, второй порядок с обратным знаком — спиновый сдвиг не разрешён.',
                      'С плотностью знаки встают на место: спиновый сдвиг +0.0038…+0.0043 при ρ = 2, 4, 8 — но это миры с шаром тени в нуле (особые); на сдвинутых мирах разброс ±0.003–0.005 (опыт ens31, dirac-doc §21).',
                      'Структура спина (две киральности, спин ↑/↓) совпадает при любой плотности. Тонкая структура 2p (~0.0015) ниже нынешней точности.']}

@experiment(id='ens31', block='Поле, атом, водород', title='Ансамбль сдвигов фазона в 3+1: атом со спином',
    what='Тот же закон мира 3+1, но шар тени сдвинут на случайный вектор — другие события. На каждом мире: уровень 1s, темп роста отклика, наклон по массе, '
         'второй порядок и спиновый сдвиг Дирак − КГ. Средние по ансамблю и экстраполяция к нулевому росту (линейно по темпу роста — приближение).',
    params=[P('nw', 'Миров', 6.0, 3, 8, 1), P('rho', 'Плотность ρ', 4.0, 2, 4, 2)],
    presets={'быстро': {'nw': 3.0, 'rho': 2.0}, 'как в документе': {'nw': 6.0, 'rho': 4.0}},
    ref='dirac-doc §21 (ρ = 4, 6 миров: 1s 0.892 ± 0.021, спин +0.0057 ± 0.0019; все 18 миров при нулевом росте: спин +0.0060 ± 0.0014 против +0.0068)', cost='~1 мин (быстро), ~10 мин (документ)')
def exp_ens31(p, region=None):
    from . import atom31 as At
    rng = np.random.default_rng(int(p['rho'] * 100 + 12)); R = []
    for k in range(int(p['nw'])):
        R.append(At.Atom(p['rho'], shift=At.random_shift(p['rho'], rng)).measure())
    REF = At.CONT_SAME_INSTRUMENT; g = np.array([r['growth'] for r in R]); nums = []
    for key, lab, tol in (('E', '1s КГ', 0.03), ('slope', 'наклон dE/dm²', 0.3), ('spin', 'спиновый сдвиг Дирак − КГ', 0.5), ('stark', 'второй порядок (вставка z)', 1.0)):
        y = np.array([r[key] for r in R]); m_, se = y.mean(), y.std(ddof=1) / math.sqrt(len(y))
        nums.append(num(key, f'{lab}: среднее ± {se:.4f} (разброс по мирам {y.std(ddof=1):.4f}; непрерывная тем же прибором {REF[key]:+.4f})', m_, REF[key], tol, rel=True, fmt='{:+.4f}'))
        if len(y) >= 3:
            c = np.polyfit(g, y, 1); nums.append(num(key + '0', f'{lab}: при нулевом росте (экстраполяция)', c[1], REF[key], tol, rel=True, fmt='{:+.4f}'))
    nums.append(num('growth', f'темп роста отклика: среднее (разброс {g.std(ddof=1):.3f})', g.mean(), 0.0, 0.03, fmt='{:+.3f}'))
    return {'numbers': nums, 'series': [{'title': 'Спиновый сдвиг против темпа роста (мир = точка)', 'xlabel': 'темп роста', 'lines': [
                {'name': 'миры', 'x': g.tolist(), 'y': [r['spin'] for r in R], 'style': 'dots'}, {'name': 'непрерывная', 'x': [float(g.min()), float(g.max())], 'y': [REF['spin']] * 2}]}],
            'notes': ['Миры с центром шара тени в нуле особые (лишняя симметрия): их совпадения не представительны.',
                      'Рост («разгон» суммы по цепочкам) не исчезает до ρ = 8 и тянет уровень вверх, а спиновый сдвиг вниз; уровень растёт с плотностью (+0.08 на удвоение ρ) — не сошёлся. Второй порядок (вставка z) не воспроизводится.']}

@experiment(id='pauli31', block='Поле, атом, водород', title='Паули в 3+1 и «гелий» в первом порядке',
    what='Два электрона со спином в поле ядра. (1) Без взаимодействия: амплитуда двух фермионов — антисимметризованное произведение откликов кристалла; '
         'проекция на 1s ⊗ 1s: ↑↑ — тождественный ноль, ↑↓ — частота 2·E₁ₛ. (2) «Гелий» 1s² (синглет): сдвиг уровня второго электрона в поле облака первого — J; '
         'поле облака точное или ИЗ ПОРЯДКА (связи, как Кулон 3+1).',
    params=[P('rho', 'Плотность атома ρ', 4.0, 2, 4, 2), P('frho', 'Плотность мира поля', 2.0, 2, 4, 2)],
    presets={'быстро': {'rho': 2.0}, 'как в документе': {'rho': 4.0, 'frho': 4.0}},
    ref='dirac-doc §22 (↑↑ ≤ 10⁻¹⁶; ↑↓ 1.790 = 2E; J: кристалл с точным полем 0.0687, с полем из порядка 0.0573 (ρ поля 4), непрерывная тем же прибором 0.0690, точно 0.0593)', cost='~3 мин (быстро), ~25 мин (документ)')
def exp_pauli31(p, region=None):
    from . import atom31 as At
    at = At.Atom(p['rho']); Bx = at.B(pauli=1.0); X = at.X
    def src(x0, spin):
        rr = np.linalg.norm(X[:, 1:] - np.array([x0, 0, 0]), axis=1); return np.where(at.src0[:, None], np.exp(-At.AL * rr)[:, None] * np.eye(2)[spin][None, :], 0)
    A_ = at.respond(Bx, src(+1.5, 0)); Bu = at.respond(Bx, src(-1.5, 0)); Bd = at.respond(Bx, src(-1.5, 1))
    ed = np.arange(1.5, at.H - 0.3 + 1e-9, 0.25); k = np.digitize(at.t, ed) - 1; ok = (k >= 0) & (k < len(ed) - 1); tc = 0.5 * (ed[1:] + ed[:-1])
    sl_ = lambda ps, c: np.bincount(k[ok], (at.prof * ps[:, c])[ok].real, minlength=len(tc)) + 1j * np.bincount(k[ok], (at.prof * ps[:, c])[ok].imag, minlength=len(tc))
    def pk(sig, om):
        w_ = np.sin(np.pi * (tc - tc.min()) / np.ptp(tc)) ** 2; S = np.abs(np.exp(-1j * np.outer(om, tc)) @ (w_ * sig)) ** 2; return om[int(np.argmax(S))]
    a0, a1, bu0, bd0, bd1 = sl_(A_, 0), sl_(A_, 1), sl_(Bu, 0), sl_(Bd, 0), sl_(Bd, 1)
    Fuu = a0 * bu0 - bu0 * a0; Buu = a0 * bu0 + bu0 * a0; Fud = a0 * bd1 - bd0 * a1; om2 = np.linspace(1.4, 2.2, 1601)
    E1 = pk(a0, np.linspace(0.6, 1.3, 1401))
    r, sg, Uc, Ec, Jc = At.cloud(); F = At.field_from_order(r, sg, rho=p['frho'])
    last = F[-1, 1] / np.interp(F[-1, 0], r, Uc); Uord = lambda q: np.where(q <= F[-1, 0], np.interp(q, F[:, 0], F[:, 1]), np.interp(q, r, Uc) * last)
    f = np.where(at.src0, at.prof, 0); om = np.linspace(0.6, 1.3, 1401)
    def level(Uf):
        phi = Pr.johnston31_respond(at.n, at.I, at.J, at.re, At.M, f, Pr.wilson31(X, at.I, at.J, Uf)); return at.peak(phi, om)
    E0 = level(At.U_NUC); Je = level(lambda q: At.U_NUC(q) + np.interp(q, r, Uc)) - E0; Jo = level(lambda q: At.U_NUC(q) + Uord(q)) - E0
    REF = At.CONT_SAME_INSTRUMENT['J']; ratio = F[:, 1] / np.interp(F[:, 0], r, Uc)
    return {'numbers': [num('uu', 'фермионы ↑↑ в 1s ⊗ 1s: max|Ψ| / бозоны (Паули: ноль)', np.abs(Fuu).max() / np.abs(Buu).max(), 0.0, 1e-12, fmt='{:.1e}'),
                        num('ud', f'фермионы ↑↓ в 1s ⊗ 1s: частота (2·E₁ₛ = {2 * E1:.4f})', pk(Fud, om2), 2 * E1, 0.02, rel=True),
                        num('bb', 'бозоны ↑↑: частота', pk(Buu, om2), 2 * E1, 0.02, rel=True),
                        num('Je', f'J на кристалле, точное поле облака (непрерывная тем же прибором {REF:.4f}; точно {Jc:.4f})', Je, REF, 0.05, rel=True),
                        num('Jo', f'J на кристалле, поле облака из порядка (ρ поля {p["frho"]:g})', Jo, REF, 0.2, rel=True),
                        num('field', 'поле облака из порядка / точное (среднее по радиусам)', float(ratio.mean()), 1.0, 0.1, rel=True, fmt='{:.3f}')],
            'series': [{'title': 'Поле облака электрона: из порядка против точного', 'xlabel': 'r', 'lines': [
                {'name': 'из порядка', 'x': F[:, 0].tolist(), 'y': F[:, 1].tolist(), 'style': 'dots'}, {'name': 'точное', 'x': r[r < 7].tolist(), 'y': Uc[r < 7].tolist()}]}],
            'notes': ['Без взаимодействия ↑↑ = 0 — алгебра антисимметрии; содержательно то, что на одной орбитали помещаются два электрона с разным спином (2·E₁ₛ).',
                      'Обмен (1s2s синглет/триплет) требует уровня 2s — не разрешён на нынешнем окне; полная пара в 3+1 (n² амплитуд) не помещается в память.',
                      'Поле из порядка внутри облака занижено (0.6–0.9 при ρ поля 2, 0.8–0.9 при 4): ближняя зона функции Грина по связям.']}

@experiment(id='grow31', block='Поле, атом, водород', title='Откуда «разгон» в 3+1: средняя связь и шум',
    what='(1) Средняя теория: среднее ядро связи c·e^{−ρV}, его хвост t·δ(t) → −0.78/√ρ и корень моды массы 1 + m²F(ω) = 0 (Im ω > 0 — рост). '
         'То же для выведенных слоёв (3, −7, 4). (2) Замер: рост отклика при m² = 1 на кристалле и на случайной посыпке той же плотности, связи против слоёв.',
    params=[P('rho', 'Плотность ρ', 1.0, 1, 2, 1)],
    presets={'быстро': {'rho': 1.0}, 'как в документе': {'rho': 1.0}},
    ref='dirac-doc §23 (средняя теория ρ = 1, m = 1: связи ω = 0.93 + 0.33i, слои 1.165 + 0.056i; замер: посыпка тоже растёт, слои хуже связей)', cost='~3 мин')
def exp_grow31(p, region=None):
    rho = p['rho']; c = math.sqrt(rho) / (2 * math.pi * math.sqrt(6)); nums = []
    def delta(h, T=8.0, N=1500):
        t = np.linspace(1e-6, T, N); A = np.array([np.trapezoid(4 * np.pi * np.linspace(0, tt, 1501) ** 2 * h(rho * np.pi * (tt * tt - np.linspace(0, tt, 1501) ** 2) ** 2 / 24), np.linspace(0, tt, 1501)) for tt in t])
        return t, c * A - t
    from scipy.optimize import fsolve
    def root(t, d, m):
        G = lambda w: np.trapezoid(np.exp(1j * w * t) * d, t); f = lambda v: (lambda w: [(1 + m * m * (-1 / w ** 2 + G(w))).real, (1 + m * m * (-1 / w ** 2 + G(w))).imag])(v[0] + 1j * v[1])
        v = fsolve(f, [m, 0.0]); return v[0] + 1j * v[1]
    for nm, w in (('связи', (1.0,)), ('слои (3, −7, 4)', (3.0, -7.0, 4.0))):
        h = lambda s_, w=w: sum(wn * s_ ** n * np.exp(-s_) / math.factorial(n) for n, wn in enumerate(w))
        t, d = delta(h); om = root(t, d, 1.0)
        nums += [num('tail' + nm[:2], f'{nm}: хвост t·δ(t) (связи — теория −0.783/√ρ)', float(np.mean((t * d)[-100:])), -0.783 / math.sqrt(rho) if len(w) == 1 else 0.0, 0.05, fmt='{:+.3f}'),
                 num('g' + nm[:2], f'{nm}: средняя теория, рост моды m = 1 (Im ω; Re ω = {om.real:.3f})', om.imag, fmt='{:+.3f}')]
    H, Rc, Rs = 12.0, 5.0, 3.0
    def worldX(kind):
        if kind == 'кристалл': return Wd.gold31(rho, 'cylinder', H=H, Rc=Rc, Rs=Rs).X
        rng = np.random.default_rng(1); n = rng.poisson(rho * (H + 0.7) * (2 * Rc) ** 3); P = np.c_[rng.uniform(-0.6, H + 0.1, n), rng.uniform(-Rc, Rc, (n, 3))]
        rr = np.linalg.norm(P[:, 1:], axis=1); P = P[rr <= np.minimum(Rc, Rs + np.maximum(P[:, 0], 0) + 0.6)]; return P[np.argsort(P[:, 0])]
    for kind in ('кристалл', 'посыпка'):
        X = worldX(kind); I, J, N = Cs.layers31(X, rho, 2, 3.0); n = len(X); r = np.linalg.norm(X[:, 1:], axis=1); t = X[:, 0]; prof = np.exp(-0.45 * r); f = np.where(t < 0.45, prof, 0)
        for nm, w in (('связи', (1.0,)), ('слои', (3.0, -7.0, 4.0))):
            Wt = np.zeros(len(I))
            for k, wk in enumerate(w): Wt[N == k] = wk
            keep = Wt != 0; phi = Pr.johnston31_respond(n, I[keep], J[keep], rho, 1.0, f, Wt[keep])
            ed = np.arange(0, H + 1e-9, 1.0); kk = np.digitize(t, ed) - 1; ok = (kk >= 0) & (kk < len(ed) - 1); cnt = np.bincount(kk[ok], minlength=len(ed) - 1)
            amp = np.sqrt(np.bincount(kk[ok], np.abs(prof * phi)[ok] ** 2, minlength=len(ed) - 1) / np.maximum(cnt, 1)); tc = 0.5 * (ed[1:] + ed[:-1]); s_ = (tc > 2) & (tc < H - 1) & (cnt > 0)
            nums.append(num(f'm_{kind[:3]}_{nm[:2]}', f'замер, {kind}, {nm}: рост отклика при m² = 1 (непрерывная тем же измерителем ≈ 0)', float(np.polyfit(tc[s_], np.log(amp[s_] + 1e-300), 1)[0]), fmt='{:+.3f}'))
    return {'numbers': nums, 'series': [], 'notes': ['Разгон есть и на случайной посыпке — это свойство суммы по цепочкам связей с массой в 3+1, а не кристалла.',
                                                    'Выведенные слои убирают хвост средней связи (средняя теория устойчива), но на настоящих мирах раздувают шум — рост больше, чем у связей.',
                                                    'В широкой области (R = 7) разгон сильнее: рост +0.15…+0.17 при ρ = 2 и 4; узкая область R = 5 его отчасти скрывала.']}

@experiment(id='fuzzy31', block='Поле, атом, водород', title='Симметрично размытый световой конус в 3+1',
    what='У кристалла нет световых пар, поэтому острый свет 3+1 можно только размыть на зерно. Связи размывают его с одной стороны (внутри конуса) — '
         'отсюда поправка δ′ и разгон. Здесь — в обе стороны: все почти световые пары, сравнимые и несравнимые, вес (c/2)·e^{−ρπτ⁴/24}·(1 ∓ β). '
         'β ≈ 0.115 (средняя теория) гасит остаток раскачки; голая масса выводится из того же ядра. Сравнение: кристалл и случайная посыпка.',
    params=[P('rho', 'Плотность ρ', 1.0, 1, 2, 1), P('R', 'Радиус области', 9.0, 5, 9, 4)],
    presets={'быстро': {'rho': 1.0, 'R': 5.0}, 'как в документе': {'rho': 1.0, 'R': 9.0}},
    ref='dirac-doc §24–25 (посыпка, β = 0.115 + голая масса: рост ≈ 0, наклон 0.514 против 0.534; кристалл: наклон 0.11 — гребёнка времён)', cost='~20 с (быстро), ~2.5 мин (документ)')
def exp_fuzzy31(p, region=None):
    from . import fuzzy as Fz
    R = p['R']; Rs = 3.0 if R <= 5 else 50.0; rho = p['rho']
    REF = {'E': 0.89899, 'slope': 0.511, 'stark': -0.00259, 'spin': 0.00682} if R <= 5 else {'E': 0.90078, 'slope': 0.534, 'stark': -0.00232, 'spin': 0.00528}
    mb = (Fz.bare_mass2(rho, 0.115, 1.0), Fz.bare_mass2(rho, 0.115, 0.9)); nums = []
    for lab, kw in (('кристалл, симметрично (β = 0)', dict(beta=0.0)), ('кристалл, β = 0.115 + голая масса', dict(beta=0.115, mb2=mb)),
                    ('посыпка, β = 0.115 + голая масса', dict(beta=0.115, mb2=mb, shift='poisson'))):
        r = Fz.measure(rho, 12.0, R, Rs, REF, label=None, **kw)
        nums += [num(f'g{len(nums)}', f'{lab}: темп роста (0 — устойчиво)', r['growth'], 0.0, 0.05, fmt='{:+.3f}'),
                 num(f'E{len(nums)}', f'{lab}: 1s ({REF["E"]:.4f})', r['E'], REF['E'], 0.05, rel=True),
                 num(f's{len(nums)}', f'{lab}: наклон dE/dm² ({REF["slope"]:.3f})', r['slope'], REF['slope'], 0.3, rel=True, fmt='{:.3f}'),
                 num(f'st{len(nums)}', f'{lab}: второй порядок, поле z ({REF["stark"]:+.5f})', r['stark'], REF['stark'], 0.5, rel=True, fmt='{:+.5f}'),
                 num(f'sp{len(nums)}', f'{lab}: спиновый сдвиг ({REF["spin"]:+.5f})', r['spin'], REF['spin'], 0.5, rel=True, fmt='{:+.5f}')]
    return {'numbers': nums, 'series': [], 'notes': [f'Голая масса из средней теории: m_b² = {mb[0]:.4f} (m = 1), {mb[1]:.4f} (m² = 0.81).',
                                                    'На посыпке рецепт работает целиком (рост 0, наклон по массе как в теории). На кристалле наклон сжат: почти световые пары идут «гребёнкой» по времени (разности времён кристалла — множество Мейера).',
                                                    'Цена: несравнимые почти световые пары берутся по координатному времени — нарушение причинности на масштабе зерна.']}

@experiment(id='ladder31', block='Поле, атом, водород', title='Лестница частот золотого кристалла и резонанс по массе',
    what='(1) Спектр «гребёнки»: почти световые пары кристалла приходят на избранных разностях времён; их преобразование Фурье даёт брэгговские частоты (у посыпки — нет). '
         '(2) Свободная частица (размытый конус, β = 0.115, голая масса из ядра) по массам: посыпка идёт за теорией, кристалл залипает на ступенях лестницы.',
    params=[P('R', 'Радиус области', 9.0, 7, 9, 2)],
    presets={'быстро': {'R': 7.0}, 'как в документе': {'R': 9.0}},
    ref='dirac-doc §26 (кристалл при m ≥ 1.1 залипает на ≈ 1.75 и ≈ 2.8; при m = 1/τ² кристалл ≈ посыпка: уровень +1.5 %, наклон +6 %)', cost='~1 мин (быстро), ~3 мин (документ)')
def exp_ladder31(p, region=None):
    from . import fuzzy as Fz
    rho, R, H, al, beta = 1.0, p['R'], 12.0, 0.45, 0.115
    ms = np.array([0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4]); t_, d_ = Fz.mean_kernel_delta(rho, beta)
    Gf = lambda w: np.trapezoid(np.exp(1j * w * t_) * d_, t_); mb2 = [m * m / (1 - m * m * Gf(m).real) for m in ms]
    def world(kind):
        if kind == 'кристалл': return np.ascontiguousarray(Wd.gold31(rho, 'cylinder', H=H, Rc=R, Rs=50.0).X)
        rng = np.random.default_rng(1); n = rng.poisson(rho * (H + 0.7) * (2 * R) ** 3); P_ = np.c_[rng.uniform(-0.6, H + 0.1, n), rng.uniform(-R, R, (n, 3))]
        P_ = P_[np.linalg.norm(P_[:, 1:], axis=1) <= R]; return np.ascontiguousarray(P_[np.argsort(P_[:, 0])])
    res = {}; om = np.linspace(0.05, 3.2, 3151)
    for kind in ('кристалл', 'посыпка'):
        X = world(kind); n = len(X); r = np.linalg.norm(X[:, 1:], axis=1); t = X[:, 0]; prof = np.exp(-al * r); sel = (t > 1.5) & (t < H - 0.3); e = np.array([1, 0], complex)
        f = np.ascontiguousarray(np.where((t < 0.45)[:, None], prof[:, None] * e[None, :], 0).astype(np.complex128))
        alpha = rho * math.pi / 24; coef = math.sqrt(rho) / (2 * math.pi * math.sqrt(6)); tc2 = (2.8 * rho ** -0.25) ** 2; out = []
        for m2 in mb2:
            B = np.ascontiguousarray(np.broadcast_to(np.eye(2) * (-m2 / rho), (n, 2, 2)), dtype=np.complex128)
            sg = Fz.respond_fly(X, tc2, alpha, coef, 0.0, 0.5, Fz.GX, Fz.GW, f, B, False, beta) @ e; S = Fz._spec(t[sel], prof[sel] * sg[sel], om)
            out.append(float(om[int(np.argmax(S * (om > 0.12)))]))
        res[kind] = np.array(out)
    nums = []
    for i, m in enumerate(ms):
        nums.append(num(f'c{i}', f'm = {m:.2f}: частота покоя, кристалл (посыпка {res["посыпка"][i]:.3f})', res['кристалл'][i], res['посыпка'][i], 0.1, rel=True, fmt='{:.3f}'))
    return {'numbers': nums, 'series': [{'title': 'Частота покоя свободной частицы против массы', 'xlabel': 'm', 'lines': [
                {'name': 'кристалл', 'x': ms.tolist(), 'y': res['кристалл'].tolist()}, {'name': 'посыпка', 'x': ms.tolist(), 'y': res['посыпка'].tolist()}]}],
            'notes': ['Залипание кристалла на ступенях лестницы — резонанс с брэгговскими частотами гребёнки; у посыпки его нет.',
                      'Точное самоподобие τ·Мир(r) = Мир(r/τ): масса m при ρ·τ⁴ ≡ масса m/τ при ρ. При m = 1/τ² (атом, dirac-doc §26) кристалл ведёт себя как посыпка.']}

@experiment(id='lightatom31', block='Поле, атом, водород', title='Атом 3+1 без гребёнки: лёгкая частица m = 1/τ²',
    what='Частица в τ² раз легче зерна (по самоподобию — то же, что m = 1 при ρ = τ⁸ ≈ 47): лестница частот кристалла её не трогает. '
         'Размытый конус (β = 0.115, голая масса из ядра), α = 0.6. Кристалл против посыпки против непрерывной теории тем же прибором; Паули — антисимметрия двух электронов.',
    params=[P('R', 'Радиус области', 11.0, 9, 13, 2)],
    presets={'быстро': {'R': 9.0}, 'как в документе': {'R': 11.0}},
    ref='dirac-doc §27 (R = 11: кристалл 1s 0.370 ± 0.001 (4 мира), посыпка 0.359, теория 0.298; спин 0.0039 / 0.0047 против 0.0144; Паули ↑↑ = 10⁻¹⁷, ↑↓ = 2E)', cost='~6 мин (быстро), ~15 мин (документ)')
def exp_lightatom31(p, region=None):
    from . import atom31f as Af
    Af.AL = 0.6; M = 1 / TAU_G ** 2; R = p['R']; H = 20.0
    ref = Af.cont_refs(M, R, H); nums = []
    for kind in ('кристалл', 'посыпка'):
        at = Af.Atom(Af.make_world(1.0, H, R, poisson_seed=1 if kind == 'посыпка' else None), 1.0, M, H, R); r = at.measure()
        nums += [num(f'E{kind[:2]}', f'{kind}: 1s (теория тем же прибором {ref["E"]:.4f})', r['E'], ref['E'], 0.05, rel=True),
                 num(f'g{kind[:2]}', f'{kind}: темп роста', r['growth'], 0.0, 0.05, fmt='{:+.3f}'),
                 num(f's{kind[:2]}', f'{kind}: наклон dE/dm² ({ref["slope"]:.3f})', r['slope'], ref['slope'], 0.2, rel=True, fmt='{:.3f}'),
                 num(f'sp{kind[:2]}', f'{kind}: спиновый сдвиг ({ref["spin"]:+.5f})', r['spin'], ref['spin'], 0.3, rel=True, fmt='{:+.5f}')]
        if kind == 'кристалл':
            pa = at.pauli(); nums += [num('uu', 'Паули: ↑↑ в 1s⊗1s / бозоны (ноль)', pa['uu'], 0.0, 1e-12, fmt='{:.1e}'), num('ud', f'Паули: ↑↓ — частота (2E₁ₛ = {2 * pa["E1"]:.4f})', pa['ud'], 2 * pa['E1'], 0.03, rel=True)]
    Af.AL = 0.45
    return {'numbers': nums, 'series': [], 'notes': ['Гребёнки нет: кристалл ≈ посыпка, разброс по сдвигам фазона ±0.001, роста нет.',
                                                    'Обе дискретные модели одинаково расходятся с теорией (уровень +15…24 %, спин ~30 %): форма ядра и глубина ядра атома не влияют, область — отчасти (R = 9/11/13: 0.369/0.359/0.352). Открыто.']}

@experiment(id='energyatom31', block='Поле, атом, водород', title='Атом 3+1: потенциал как вставка при энергии E',
    what='Лёгкая частица m = 1/τ² (без гребёнки), α = 0.6. Фаза Вильсона на длинных шагах размытого конуса даёт только прямолинейный пролёт; '
         'здесь потенциал — вставка в каждом событии m² → m² + 2EU − U² (точна для стационарного состояния), уровень — неподвижная точка ω(E) = E. '
         'Эталон — непрерывная теория тем же методом и прибором (источник и поле только внутри области).',
    params=[P('R', 'Радиус области', 11.0, 9, 11, 2)],
    presets={'быстро': {'R': 9.0}, 'как в документе': {'R': 11.0}},
    ref='dirac-doc §28 (R = 11: непрерывная 0.3358 / спин +0.0190; посыпка 0.3475 / +0.0132; кристалл 0.3555 / +0.0102)', cost='~8 мин (быстро), ~25 мин (документ)')
def exp_energyatom31(p, region=None):
    from . import atom31f as Af
    Af.AL = 0.6; M = 1 / TAU_G ** 2; R = p['R']; H = 20.0
    Ec = Af.energy_level_cont(M, R, H); Ecd = Af.energy_level_cont(M, R, H, pauli=1.0, E0=Ec); nums = []
    for kind in ('кристалл', 'посыпка'):
        at = Af.Atom(Af.make_world(1.0, H, R, poisson_seed=1 if kind == 'посыпка' else None), 1.0, M, H, R)
        E = Af.energy_level(at); Ed = Af.energy_level(at, pauli=1.0, E0=E)
        nums += [num(f'E{kind[:2]}', f'{kind}: 1s (непрерывная тем же методом {Ec:.4f})', E, Ec, 0.05, rel=True),
                 num(f's{kind[:2]}', f'{kind}: спиновый сдвиг ({Ecd - Ec:+.5f})', Ed - E, Ecd - Ec, 0.4, rel=True, fmt='{:+.5f}')]
    Af.AL = 0.45
    return {'numbers': nums, 'series': [], 'notes': ['Свободный свет и свободная массивная частица совпадают с теорией по временному ряду до 3–6 %; ошибка была в связи с ядром через фазу на длинных шагах.',
                                                    'Уровень здесь — пик спектра в окне ≈ одного периода (смесь уровня и порога), поэтому сравнивается с непрерывной теорией ТЕМ ЖЕ методом; истинный уровень 0.298.']}

@experiment(id='hinvatom31', block='Поле, атом, водород', title='Атом 3+1: один проход и гармоническая инверсия',
    what='Лёгкая частица m = 1/τ², α = 0.6. Уровень — не пиком спектра (в окне один период — смесь уровня и порога), а многоканальной гармонической '
         'инверсией по шаровым слоям. Потенциал — вставкой при энергии E; для сравнения — фазой Вильсона. Эталон — непрерывная теория тем же прибором.',
    params=[P('R', 'Радиус области', 11.0, 9, 11, 2)],
    presets={'быстро': {'R': 9.0}, 'как в документе': {'R': 11.0}},
    ref='dirac-doc §29 (R = 11: кристалл 0.3027 / спин 0.0114; непрерывная с обрезкой 0.3079 / 0.0123, без обрезки 0.2984 / 0.0108; истинный 0.2978; фазой — 0.344)', cost='~6 мин (быстро), ~20 мин (документ)')
def exp_hinvatom31(p, region=None):
    from . import atom31f as Af
    Af.AL = 0.6; M = 1 / TAU_G ** 2; R = p['R']; H = 20.0
    Ct = Af.cont_level_hinv(M, R, H); Cts = Af.cont_level_hinv(M, R, H, pauli=1.0) - Ct; Cu = Af.cont_level_hinv(M, R, H, trunc=False)
    at = Af.Atom(Af.make_world(1.0, H, R), 1.0, M, H, R); e = np.array([1, 0])
    Eph = Af.level_hinv(at, at.respond(at.B(M)) @ e); E = Af.energy_level_hinv(at); Es = Af.energy_level_hinv(at, pauli=1.0, E0=E) - E
    Af.AL = 0.45
    return {'numbers': [num('E', f'кристалл, вставка при E: 1s (непрерывная с обрезкой {Ct:.4f}; без обрезки {Cu:.4f})', E, Ct, 0.03, rel=True),
                        num('S', f'кристалл: спиновый сдвиг (непрерывная {Cts:+.5f})', Es, Cts, 0.25, rel=True, fmt='{:+.5f}'),
                        num('P', f'для сравнения — фазой Вильсона: 1s (неверно на длинных шагах; эталон {Ct:.4f})', Eph)],
            'series': [], 'notes': ['Гармоническая инверсия на непрерывной теории: окно 20 → 0.2992, окно 30 → 0.2971 при истинном 0.2978 (пик спектра давал 0.338).']}

@experiment(id='ordercone31', block='Поле, атом, водород', title='Свет 3+1 из порядка: две стороны конуса',
    what='Вес размытого конуса e^{−ρπτ⁴/24} = e^{−ρV}, V — наименьший причинный ромб, содержащий обе точки. Сравнимая сторона: V — счёт событий '
         'в интервале. Несравнимая: наименьший ромб с вершинами-событиями. Плюс средняя теория одностороннего конуса (только сравнимые пары).',
    params=[P('H', 'Высота ромба-мира', 15.0, 13, 15, 2)],
    presets={'быстро': {'H': 13.0}, 'как в документе': {'H': 15.0}},
    ref='dirac-doc §30 (H = 15: сравнимые n ≈ 1.15·ρV, разброс 0.8; несравнимые n ≈ 2.1·ρV + 2.0, разброс 2.8, вес занижен в 2.6 раза; Im G одностороннего −0.08 / −0.19 при ω = 0.7 / 1)', cost='~10 с')
def exp_ordercone31(p, region=None):
    from . import ordercone as Oc
    H = p['H']; X = Wd.gold31(1.0, 'diamond', H=H).X; X = X[np.argsort(X[:, 0])]
    A = Oc.pair_stats(X, 1.0, np.abs(X[:, 0]) + np.linalg.norm(X[:, 1:], axis=1) <= H / 2 - 3.5)
    T, S = A[A[:, 0] > 0], A[A[:, 0] < 0]; pt = np.polyfit(T[:, 1], T[:, 2], 1); ps = np.polyfit(S[:, 1], S[:, 2], 1)
    ws = np.exp(-S[:, 2]).mean() / np.exp(-S[:, 1]).mean(); oms = [0.382, 0.7, 1.0]
    g1 = Oc.mean_G(Oc.kernel_S1(), oms); h, _ = Oc.kernel_onesided(); g2 = Oc.mean_G(h, oms)
    return {'numbers': [num('ts', 'сравнимые: наклон n по ρV (счёт = объём)', pt[0], 1.0, 0.2),
                        num('tsd', 'сравнимые: разброс n', float(np.std(T[:, 2] - np.polyval(pt, T[:, 1]))), fmt='{:.2f}'),
                        num('ss', 'несравнимые, наименьший ромб: наклон n по ρV (должно быть 1)', ps[0], 1.0, 0.2),
                        num('sw', 'несравнимые: вес e^{−n} / e^{−ρV} (должно быть 1)', ws, 1.0, 0.2),
                        num('g1', 'S1 (обе стороны): Im G при ω = 1 (0 — нет раскачки)', float(g1[2].imag), 0.0, 0.03),
                        num('g2', 'односторонний (только сравнимые): Im G при ω = 1', float(g2[2].imag), 0.0, 0.03)],
            'series': [], 'notes': ['Сравнимая сторона света читается из порядка (счёт ≈ объём); несравнимая — нет (дискретные вершины ромба «съедают» объём, шум 2–3 события).',
                                    'Односторонний конус (только сравнимые пары, вес меняет знак) в средней теории раскачивается на частотах ~ зерна: обе стороны конуса нужны.',
                                    'Атом в шаре R = 6 с односторонним ядром (координаты и порядок) — неустойчив / шумит (order_dbg.py), S1 — чистая мода.']}

@experiment(id='boxorder31', block='Поле, атом, водород', title='Ящик □ из слоёв прошлого: вывод весов Соркина и проверка на кристалле',
    what='Ящик □ = Σ_k c_k Σ_{слой k}(φ(y) − φ(x)), слой k — прошлые события с k событиями в интервале. Веса — из обнуления моментов по гиперболоидам: '
         '4 слоя дают ровно веса Соркина (1, −9, 16, −8), 5 слоёв точнее. Проверка: сокращаются ли слои на кристалле в каждом событии.',
    params=[P('H', 'Высота ромба-мира', 15.0, 13, 15, 2)],
    presets={'быстро': {'H': 13.0}, 'как в документе': {'H': 15.0}},
    ref='dirac-doc §31 (веса 1, −9.00, 16.01, −8.01; 5 слоёв Im B̂/ω² = −0.010 при ω = 0.382; на кристалле W(x) = 179 ± 342 при окне 4 вместо ≈ 0.5)', cost='~30 с')
def exp_boxorder31(p, region=None):
    from . import box31 as Bx, ordercone as Oc
    P_ = [Bx.Pk_poisson(k) for k in range(5)]; c4 = Bx.design(P_[:4], 4); c5 = Bx.design(P_, 5)
    _, s4, mb4, im4 = Bx.coeffs(4); _, s5, mb5, im5 = Bx.coeffs(5)
    H = p['H']; X = Wd.gold31(1.0, 'diamond', H=H).X; X = X[np.argsort(X[:, 0])]; U, C, Cnt = Oc.causal(X); dt = X[:, 0][None, :] - X[:, 0][:, None]
    L = np.where(C & (Cnt < 5) & (dt <= 4.0), Cnt, -1).astype(int); W = np.array([c5[L[:, x][L[:, x] >= 0]].sum() for x in range(len(X))])
    ok = (np.abs(X[:, 0]) + np.linalg.norm(X[:, 1:], axis=1) <= H / 2 - 0.5) & (X[:, 0] > -H / 2 + 4.5)
    return {'numbers': [num('w1', '4 слоя: второй вес (Соркин: −9)', c4[1], -9.0, 0.002, rel=True), num('w2', '4 слоя: третий вес (Соркин: 16)', c4[2], 16.0, 0.002, rel=True),
                        num('w3', '4 слоя: четвёртый вес (Соркин: −8)', c4[3], -8.0, 0.002, rel=True),
                        num('i4', 'средняя теория, 4 слоя: Im B̂/ω² при ω = 0.382 (затухание)', im4(0.382) / 0.382 ** 2),
                        num('i5', 'средняя теория, 5 слоёв: Im B̂/ω² при ω = 0.382', im5(0.382) / 0.382 ** 2, 0.0, 0.02),
                        num('wm', 'кристалл: местный член W(x), окно 4 — среднее (должно быть ~ α ≈ 0.5)', float(W[ok].mean()), 0.5, 0.5, fmt='{:.1f}'),
                        num('ws', 'кристалл: разброс W(x) по событиям', float(W[ok].std()), fmt='{:.1f}')],
            'series': [], 'notes': ['Веса Соркина выводятся одним условием: ядро слоёв не должно иметь моментов M0, M1, M2 по τ² (иначе — масса, трение, поправка к □).',
                                    'На кристалле слои распределены не по Пуассону и сильно меняются от события к событию: сокращение не происходит, марш расходится.',
                                    'Симметричный конус с весами из счёта (сравнимые — интервал, несравнимые — наименьший ромб) и обходом по высоте: свободная частица 0.915 против 0.926 по координатам; атом пока расходится (osym.py).']}

@experiment(id='massive31', block='Поле, атом, водород', title='Массивная частица 3+1 без цепочки по массе: только сравнимые пары',
    what='G_m = δ(τ²)/2π − m J1(mτ)/(4πτ): край конуса — один шаг по связям, внутренность — по сравнимым парам с τ из счёта интервала. Масса в весе, а не '
         'повторением безмассового шага. Атом (M = 0.7, α = 0.6, ядро смягчено a = 1) в шаре R, вставка при энергии E, гармоническая инверсия. Сравнение: τ из координат, размытый конус S1, непрерывная теория.',
    params=[P('R', 'Радиус шара', 6.0, 5, 6, 1)],
    presets={'быстро': {'R': 5.0}, 'как в документе': {'R': 6.0}},
    ref='dirac-doc §32 (R = 6: по счёту 0.6415 / спин +0.0015; по координатам 0.6177 / +0.0018; S1 0.6527 / +0.0063; непрерывная 0.6240 / +0.0112)', cost='~3 мин, ~3 ГБ памяти')
def exp_massive31(p, region=None):
    from . import atom31f as Af, massive31 as Ms
    Af.AL = 0.6; Af.A = 1.0; M = 0.7; R = p['R']; H = 14.0
    try:
        ref = Af.cont_level_hinv(M, R, H); refs = Af.cont_level_hinv(M, R, H, pauli=1.0) - ref; X = Af.make_world(1.0, H, R)
        Eo, So = Ms.level_and_spin(X, 1.0, M, H, R, 'order'); Ec, Sc = Ms.level_and_spin(X, 1.0, M, H, R, 'coord', spin=False); E1, S1 = Ms.level_and_spin(X, 1.0, M, H, R, 's1')
    finally:
        Af.AL = 0.45; Af.A = 0.5
    return {'numbers': [num('Eo', f'по счёту (только порядок): 1s (непрерывная {ref:.4f})', Eo, ref, 0.03, rel=True),
                        num('Ec', 'то же, τ из координат: 1s', Ec, ref, 0.03, rel=True),
                        num('E1', 'размытый конус S1 (координаты): 1s', E1, ref, 0.03, rel=True),
                        num('So', f'по счёту: спиновый сдвиг (непрерывная {refs:+.5f})', So, refs, 0.25, rel=True, fmt='{:+.5f}'),
                        num('S1', 'S1: спиновый сдвиг (для сравнения)', S1, fmt='{:+.5f}')],
            'series': [], 'notes': ['Масса — в весе (Бессель от τ из счёта), а не повторением безмассового шага: разгонять нечему; несравнимые пары не нужны.',
                                    'Уровень лучше, чем у S1; спин-орбита — нет: короткие расстояния у ядра идут через край конуса (связи), а связи на масштабе зерна грубы.',
                                    'Плотные матрицы: мир до ~1.5·10⁴ событий. Для атома m = 1/τ² (R = 11, 10⁵ событий) нужен дешёвый счёт интервалов.']}

# ======================= ВЗЯТО РУКАМИ =======================
# Всё, что опыт берёт не из ядра (core.py: тень в шаре + порядок). Плотность ρ и размер области — масштаб и окно обзора, не физика;
# здесь их нет. Цель роадмапа — чтобы эти списки сокращались. Пустой список — опыт целиком из ядра.
JOHNSTON = 'закон движения — пропагатор Джонстона (сумма по цепочкам: ½ на шаг, −m²/ρ на остановку)'
HAND = {
    'massive31': ['масса M = 0.7, α = 0.6, ядро смягчено a = 1, шар R = 6 (малый мир)', 'таблица n → v кристалла (калибровка счёта)', 'вес связи c = √ρ/(2π√6) (Джонстон)', 'потенциал вставкой при энергии E', 'член Паули из уравнения Дирака'],
    'boxorder31': ['плотность ρ = 1, ромб-мир', 'моменты — по распределению Пуассона (как у Соркина)', 'окно по времени 4 — обрезка', 'нормировка и голая масса — средняя теория'],
    'ordercone31': ['плотность ρ = 1, ромб-мир', 'вес e^{−ρV} — гауссиан по τ² (форма — выбор)', 'вершины наименьшего ромба ищутся среди близких (≤ 12 событий)', 'β′ = 0.05 у одностороннего — подбор под лёгкую частицу'],
    'hinvatom31': ['масса m = 1/τ²', 'размытый конус (β = 0.115), голая масса из ядра', 'потенциал вставкой при энергии E (самосогласование)', 'член Паули из уравнения Дирака', 'гармоническая инверсия: 6 слоёв, 8 мод — выбор прибора'],
    'energyatom31': ['масса m = 1/τ²', 'размытый конус (β = 0.115), голая масса из ядра', 'потенциал −α/√(r² + a²), α = 0.6 — вставкой при энергии E (точна для стационарного состояния; самосогласование)', 'член Паули из уравнения Дирака', 'несравнимые пары — по координатному времени'],
    'lightatom31': ['масса m = 1/τ² (относительно зерна)', 'размытый конус (c/2)e^{−ρπτ⁴/24}(1 ∓ β), β = 0.115; голая масса — из ядра', 'потенциал ядра −α/√(r² + a²), α = 0.6 — по нити Вильсона', 'член Паули из уравнения Дирака', 'несравнимые пары — по координатному времени'],
    'ladder31': ['масса m (скан)', 'размытый конус (c/2)e^{−ρπτ⁴/24}(1 ∓ β), β = 0.115; голая масса — из ядра', 'несравнимые пары — по координатному времени'],
    'fuzzy31': ['масса m = 1 (голая — выведена из ядра)', 'закон движения — сумма по цепочкам; свет — размытый конус (c/2)e^{−ρπτ⁴/24}(1 ∓ β) по обе стороны (форма — гауссиан, выбор; β — из средней теории)', 'несравнимые пары — по координатному времени', 'потенциал ядра по нити Вильсона', 'член Паули из уравнения Дирака'],
    'grow31': ['масса m = 1', JOHNSTON + '; в 3+1 — по связям', 'средняя теория — по посыпке (приближение)', 'веса слоёв (3, −7, 4) — выведены из трёх условий'],
    'ens31': ['масса m = 1', JOHNSTON + '; в 3+1 — по связям', 'потенциал ядра по нити Вильсона', 'член Паули из уравнения Дирака', 'сдвиги шара тени — выбор ансамбля', 'экстраполяция к нулевому росту — приближение'],
    'pauli31': ['масса m = 1', JOHNSTON + '; в 3+1 — по связям', 'потенциал ядра по нити Вильсона', 'величина заряда e² = α/2', 'облако 1s — из непрерывной теории (поле — из порядка или точное)', 'первый порядок по отталкиванию (приближение)'],
    'spin31': ['масса m = 1', JOHNSTON + '; в 3+1 — по связям, множитель √ρ/(2π√6)', 'потенциал ядра −α/√(r² + a²), a = 0.5 — вставлен по нити Вильсона', 'член Паули ∓iσ·∇U — из уравнения Дирака (готовый)', 'вставка z·e^{−r/2} — пробник второго порядка'],
    'whytau': ['число θ вместо τ (выбор опыта)'],
    'nonull': [],
    'pairensemble': ['масса m = 1', JOHNSTON, 'ядро — трубка, заряд e²', 'векторная вершина (энергия в полюсе — приближение)', 'сдвиги шара тени — выбор ансамбля', 'нуль энергии — калибровка'],
    'eos': ['вселенная — посыпка с заданным a(η) (мир не кристалл)', 'постоянное w, известный темп h и начало отсчёта', 'конформная плоскость'],
    'lithiumnew': ['масса m = 1', JOHNSTON, 'ядро — трубка, заряд e²', 'векторная вершина (энергия в полюсе — приближение)', 'парное приближение для трёх тел', 'нуль энергии — калибровка'],
    'vectorvertex': ['масса m = 1', JOHNSTON, 'ядро — трубка, заряд e²', 'вершина i·(e²/ρ²)(E − U)(E − U)·[несравнимы]: форма из порядка, нормировка выведена; энергия в полюсе — приближение', 'нуль энергии — калибровка'],
    'pairnorm': ['масса m = 1', JOHNSTON, 'ядро — трубка, заряд e²', 'вершина i·λ·[несравнимы]: λ сшита с весами частиц (зависит от пары состояний)', 'нуль энергии — калибровка'],
    'pairexchange': ['масса m = 1', JOHNSTON, 'ядро — трубка, заряд e²', 'вершина пары i·λ·[несравнимы] — форма из порядка, нормировка λ не выведена', 'нуль энергии — калибровка'],
    'heliumorder': ['масса m = 1', JOHNSTON, 'ядро — трубка событий радиуса 1', 'величина заряда e² (ядро +2)', 'нуль энергии: вычтены однородные части α·t + β (калибровка)', 'Хартри и обмен — первый порядок (приближение)'],
    'orderatom': ['масса m = 1', JOHNSTON, 'ядро — трубка событий радиуса a = 1', 'величина заряда g (ядро × электрон)', 'нуль энергии: вычтена однородная часть α·t + β (калибровка)'],
    'lightspin': ['масса m = 1', JOHNSTON, 'потенциал ядра U = F|x| — вставлен готовым (фаза по световому уголку)'],
    'onebody_long': ['масса m = 1', JOHNSTON, 'поле ядра F|x| — вставлено готовым'],
    'dirac_walk': ['правило нитей (паросочетание в системе кристалла)', 'монета: смешение компонент на угол m·Δt', 'масса m', 'две компоненты — «спин» 1+1'],
    'field11': ['заряд — трубка событий (форма, радиус)', 'пропагатор света ½ за отношение (Джонстон)'],
    'pauli_atom': ['правило нитей', 'монета', 'масса m = 1', 'поле ядра −α/√(x² + 1) — вставлено готовым'],
    'helium_field': ['правило нитей', 'монета', 'масса m = 1', 'величина заряда e²'],
    'lithium_scf': ['правило нитей', 'монета', 'масса m = 1', 'величина заряда e²', 'метод Хартри (среднее поле, без обмена) — приближение'],
    'onerule': [JOHNSTON, 'масса m'],
    'world': [],
    'dispersion': ['масса m', JOHNSTON, 'источник: широкая волна e^{ikx}'],
    'clock': ['масса m', JOHNSTON, 'источник «покоя» — полоса'],
    'light31': ['масса m', JOHNSTON + '; в 3+1 — по связям, множитель √ρ/(2π√6) из предела континуума'],
    'coulomb31': ['заряд — неподвижная трубка событий (его форма и радиус)', 'потенциал = функция Грина Джонстона × число связей с трубкой'],
    'charge_force': ['внешнее поле: V-образный потенциал, его сила и место', 'масса m = 1', JOHNSTON, 'связь поля — световой уголок (выведена из калибровочной инвариантности)'],
    'atom11': ['потенциал ядра U = F|x| — вставлен готовым', 'масса m = 1', JOHNSTON],
    'hydrogen11': ['потенциал ядра −α/√(x² + a²) — вставлен готовым, размер ядра a', 'масса m = 1', JOHNSTON],
    'hydrogen31': ['потенциал ядра −α/√(r² + a²), a = 0.5 — вставлен по нити Вильсона', 'масса m = 1', JOHNSTON],
    'geom_charge': ['форма среза (закрутка κ) — задана, не из уравнений', 'масса m = 1', JOHNSTON, 'заряда q нет: только целое число оборотов m', 'контроль: внешнее поле U = F·x'],
    'light_point': ['правило события: ближайшая изометрия входа в выход, масштаб ядра m₀'],
    'bell': ['правило события (m₀ = 2)', 'накачка пары (нечётная) — выбор состояния', 'измерение — идеальный поворот базиса'],
    'pair': ['массы m₁, m₂', 'взаимодействие: мгновенная вершина F|x₁ − x₂| — сила и форма руками', 'мера вершины (нормировка не установлена)', JOHNSTON],
    'slice_rt': ['закон среза — экстремум действия (Редже–Тейтельбойм), выбран', 'начальные данные: отклонение δA от Шварцшильда', 'масса центра M = 1, скорость ω в тени'],
    'nonabelian': ['форма среза (изгиб κ, поворот плоскостей) — задана'],
    'color': ['форма среза — задана', 'масса m = 1', JOHNSTON, 'цвет источника — выбор состояния'],
}
for _k, _v in HAND.items(): REG[_k]['hand'] = _v

def catalog():
    out = []
    for e in REG.values():
        out.append({k: v for k, v in e.items() if k != 'run'})
    return out

def run(exp_id, params=None, region=None):
    e = REG[exp_id]; pr = {q['name']: q['default'] for q in e['params']}; pr.update(params or {})
    for q in e['params']:
        if q['choices'] is None and pr.get(q['name']) is not None: pr[q['name']] = float(pr[q['name']])
    T0 = time.time(); res = e['run'](pr, region); res['seconds'] = round(time.time() - T0, 1); res['params'] = pr; res['region'] = region; res['exp'] = exp_id
    return res
