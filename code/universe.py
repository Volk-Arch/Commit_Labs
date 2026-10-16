#!/usr/bin/env python3
"""СИМУЛЯТОР ВСЕЛЕННОЙ: один прогон — один мир — все параметры, которые умеем мерить.

    python3 universe.py ПРЕСЕТ [зерно] [--quick]
    python3 universe.py ПРЕСЕТ [зерно] --vs ДРУГОЙ_ПРЕСЕТ     (два мира рядом)
    пресеты: gold-flat, poisson-flat, gold-trace, poisson-trace, gold-ds, poisson-ds, gold-twist
    (можно задать свой: см. PRESETS внизу)

Мир 1+1 строится один раз: события, их место в окне (тень), порядок (плотная матрица).
Потом по очереди работают модули замеров; каждый возвращает строки
(величина, значение, ошибка, эталон, комментарий). Итог — паспорт мира:
    data/universe/<имя>.json   и   docs/runs/<имя>.md

Модули: густота, размерность по масштабам, ромбы и шум при бустах, гипероднородность,
связи, часы (самые длинные цепочки) и их изотропия, дно интервала, движение по окну
(намотка), кривизна (для де Ситтера), калибровочное поле (для закрученного среза).
"""
import sys, os, math, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.special import gamma
from scipy.optimize import brentq
from bentcut import zt_range, ST, SI, exact_area, world as ds_world, poisson_on_sigma, mdot

# ---------------------------------------------------------------- миры
class World:
    """X — координаты событий на срезе: плоский (T, x) или де Ситтер (T, x, Z);
    W — место в окне (T', x'); kind — 'flat' | 'ds'; ell — радиус dS."""
    def __init__(s, X, W, kind, rho, r, ell=None, region=None, name=''):
        o = np.argsort(X[:, 0]); s.X, s.W = X[o], W[o]
        s.kind, s.rho, s.r, s.ell, s.region, s.name = kind, rho, r, ell, region, name
        s.n = len(s.X)
    # причинность между произвольными точками (не только событиями)
    def prec(s, a, B):
        if s.kind == 'flat':
            d = B - a; return (d[..., 0] > 0) & (d[..., 0] ** 2 > d[..., 1] ** 2)
        return (B[..., 0] > a[0]) & (mdot(B, a) >= s.ell ** 2)
    def tau(s, a, b):
        if s.kind == 'flat':
            d = b - a; return np.sqrt(np.maximum(d[..., 0] ** 2 - d[..., 1] ** 2, 0))
        return s.ell * np.arccosh(np.maximum(mdot(a, b) / s.ell ** 2, 1))
    def vol(s, tau):
        return tau ** 2 / 2 if s.kind == 'flat' else exact_area(s.ell, tau)
    def inner(s, pts, margin):
        """точка далеко от края области"""
        if s.kind == 'flat':
            return np.abs(pts[..., 0]) + np.abs(pts[..., 1]) <= s.region - margin
        sm, chm = s.region; sig = np.arcsinh(pts[..., 0] / s.ell); chi = np.arctan2(pts[..., 1], pts[..., 2])
        return (np.abs(sig) <= sm - margin / s.ell) & (np.abs(chi) <= chm - margin / s.ell)

def golden_flat(rho, L, rs):
    r = math.sqrt(5 * rho / math.pi); sh = rs.uniform(0, 300, 2); so = rs.uniform(-1, 1, 2)
    t, tp = zt_range((sh[0] - L) / ST, (sh[0] + L) / ST, (so[0] - r) / SI, (so[0] + r) / SI); T, Tq = t * ST - sh[0], tp * SI - so[0]
    c, cp = zt_range(sh[1] - L, sh[1] + L, so[1] - r, so[1] + r); c, cp = c - sh[1], cp - so[1]
    i, j = np.nonzero(Tq[:, None] ** 2 + cp[None, :] ** 2 <= r * r)
    X = np.stack([T[i], c[j]], 1); W = np.stack([Tq[i], cp[j]], 1)
    m = np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L
    return World(X[m], W[m], 'flat', rho, r, region=L)

def poisson_flat(rho, L, rs):
    r = math.sqrt(5 * rho / math.pi); n = rs.poisson(rho * 2 * L * L)
    u = rs.uniform(-L / 2, L / 2, (n, 2)); X = np.stack([u[:, 0] + u[:, 1], u[:, 0] - u[:, 1]], 1)
    a = rs.uniform(0, 2 * math.pi, n); rr = r * np.sqrt(rs.random(n))
    return World(X, np.stack([rr * np.cos(a), rr * np.sin(a)], 1), 'flat', rho, r, region=L)

def golden_ds(rho, ell, sm, rs):
    r = math.sqrt(5 * rho / math.pi)
    sig, chi, X = ds_world(ell, r, sm, sm, tuple(rs.uniform(-5, 5, 2)))
    return World(X, np.zeros((len(X), 2)), 'ds', rho * (1 + r * r / (4 * ell * ell)), r, ell, (sm, sm))

def poisson_ds(rho, ell, sm, rs):
    r = math.sqrt(5 * rho / math.pi); X = poisson_on_sigma(ell, rho, sm, sm, rs)
    return World(X, np.zeros((len(X), 2)), 'ds', rho, r, ell, (sm, sm))

# ---------------------------------------------------------------- порядок
def build_order(w, rule):
    """плотная матрица R[i, j] = (i ≺ j). rule: 'proj' — по срезу; 'trace' — по полному следу."""
    X = w.X
    if w.kind == 'flat':
        d0 = X[None, :, 0] - X[:, None, 0]; s = d0 ** 2 - (X[None, :, 1] - X[:, None, 1]) ** 2
        if rule == 'trace': s = s - ((w.W[None, :, :] - w.W[:, None, :]) ** 2).sum(-1)
        return (d0 > 0) & (s > 0)
    assert rule == 'proj', 'след на кривом срезе пока не поддержан'
    G = -np.outer(X[:, 0], X[:, 0]) + np.outer(X[:, 1], X[:, 1]) + np.outer(X[:, 2], X[:, 2])
    return (X[None, :, 0] > X[:, None, 0]) & (G >= w.ell ** 2)

def longest(R, a, b):
    idx = np.nonzero(R[a] & R[:, b])[0]; idx = np.concatenate([[a], idx, [b]])
    sub = R[np.ix_(idx, idx)]; m = len(idx); dist = np.full(m, -10 ** 9); dist[0] = 0; pred = np.full(m, -1)
    for k in range(1, m):
        p = np.nonzero(sub[:k, k])[0]
        if len(p): j = p[np.argmax(dist[p])]; dist[k] = dist[j] + 1; pred[k] = j
    path = [m - 1]
    while path[-1] != 0: path.append(pred[path[-1]])
    return idx[np.array(path[::-1])]

# ---------------------------------------------------------------- случайные ромбы
def random_diamond(w, tau, rs, eta_max=1.5, margin=None):
    """точки p, q (не события) с собственным временем tau, случайное место и буст."""
    margin = tau * 1.6 if margin is None else margin
    for _ in range(1000):
        if w.kind == 'flat':
            c = rs.uniform(-w.region, w.region, 2); eta = rs.uniform(-eta_max, eta_max)
            u = np.array([math.cosh(eta), math.sinh(eta)]) * tau / 2
            p, q = c - u, c + u
            if w.inner(p, 0) and w.inner(q, 0) and w.inner(c, tau * math.cosh(eta) * 0.6):
                return p, q, eta
        else:
            from bentcut import iso
            s0 = tau / (2 * w.ell); p0 = np.array([w.ell * math.sinh(-s0), 0, w.ell * math.cosh(s0)]); q0 = np.array([w.ell * math.sinh(s0), 0, w.ell * math.cosh(s0)])
            eta = rs.uniform(-0.45, 0.45); L_ = iso(rs.uniform(-1.2, 1.2), rs.uniform(-0.9, 0.9), eta)
            p, q = L_ @ p0, L_ @ q0
            if w.inner(p, tau * 0.3) and w.inner(q, tau * 0.3): return p, q, eta
    raise RuntimeError('ромб не помещается')

def count_in(w, p, q):
    if w.kind == 'flat': inside = w.prec(p, w.X) & w.prec(w.X, q[None, :])
    else: inside = w.prec(p, w.X) & (w.X[:, 0] < q[0]) & (mdot(w.X, q) >= w.ell ** 2)
    return int(np.count_nonzero(inside))

# ---------------------------------------------------------------- модули замеров
def row(name, val, err=None, ref=None, note=''):
    return dict(name=name, value=float(val) if val is not None else None, err=None if err is None else float(err),
                ref=ref, note=note)

def m_density(w, R, rs, q):
    if w.kind == 'flat': A = 2 * w.region ** 2
    else: sm, chm = w.region; A = w.ell ** 2 * 2 * math.sinh(sm) * 2 * chm
    return [row('событий', w.n), row('густота на собственный объём', w.n / A, math.sqrt(w.n) / A, w.rho, 'эталон — заказанная густота (для dS с поправкой толщины трубки)')]

def f_of_d(d): return 2 * gamma(d + 1) * gamma(d / 2) / (4 * gamma(3 * d / 2))
def mm_dim(f): return brentq(lambda d: f_of_d(d) - f, 1.0, 14)

def pairs_by_tau(w, R, lo, hi, rs, want, central):
    out = []; tries = 0
    while len(out) < want and tries < 40000:
        tries += 1; a, b = rs.integers(w.n, size=2)
        if not R[a, b]: continue
        t = w.tau(w.X[a], w.X[b])
        if lo <= t < hi and w.inner(w.X[a], central) and w.inner(w.X[b], central): out.append((a, b, t))
    return out

def m_dimension(w, R, rs, q):
    rows = []
    for lo, hi in ((4, 7), (7, 12), (12, 20)):
        ds = []
        for a, b, t in pairs_by_tau(w, R, lo, hi, rs, 15 if q else 40, 2):
            I = np.nonzero(R[a] & R[:, b])[0]
            if len(I) < 6: continue
            S = R[np.ix_(I, I)]; n = len(I); ds.append(mm_dim(min(max(S.sum() / (n * (n - 1) / 2), 0.004), 0.99)))
        if ds: rows.append(row(f'размерность по порядку, τ ∈ [{lo},{hi})', np.mean(ds), np.std(ds) / math.sqrt(len(ds)), 2.0, f'{len(ds)} ромбов'))
    return rows

def m_diamonds(w, R, rs, q):
    rows = []
    if w.kind != 'flat': return rows
    tau = math.sqrt(2 * 40.0 / w.rho)                       # ромб, в котором в среднем 40 событий
    M = 300 if q else 1500; c = []; e = []
    for _ in range(M):
        p, qq, eta = random_diamond(w, tau, rs); c.append(count_in(w, p, qq)); e.append(abs(eta))
    c, e = np.array(c), np.array(e); mu = 40.0
    rows.append(row('ромб на 40 событий: среднее', c.mean(), c.std() / math.sqrt(M), mu))
    rows.append(row('ромб на 40 событий: разброс', c.std(), c.std() / math.sqrt(2 * M), math.sqrt(mu), 'эталон — засев Пуассона √40'))
    rows.append(row('шум ромба / шум засева', c.std() / math.sqrt(mu), c.std() / math.sqrt(mu) / math.sqrt(2 * M), 1.0, '< 1 — тише засева'))
    rows.append(row('разброс при |η| < 0.5', c[e < 0.5].std(), None, None))
    rows.append(row('разброс при |η| > 1', c[e > 1.0].std(), None, None, 'рост с бустом — признак ломки Лоренца'))
    rows.append(row('пустых ромбов', np.mean(c == 0), None, 0.0))
    return rows

def m_hyper(w, R, rs, q):
    if w.kind != 'flat': return []
    Vs = np.array([10, 20, 40, 80, 160.0]); var = []
    for V in Vs:
        tau = math.sqrt(2 * V / w.rho); cc = []
        for _ in range(150 if q else 500):
            p, qq, _ = random_diamond(w, tau, rs, eta_max=1.0); cc.append(count_in(w, p, qq))
        var.append(np.var(cc))
    sl, ic = np.polyfit(np.log(Vs), np.log(np.maximum(var, 1e-3)), 1)
    return [row('гипероднородность: Var ∝ V^α, α', sl, None, 1.0, 'Пуассон: 1; меньше — тише на больших масштабах'),
            row('дисперсия при V = 10 / 160', var[0], None, None, f'{var[0]:.2f} / {var[-1]:.2f}')]

def m_links(w, R, rs, q):
    Rf = R.astype(np.float32); L = R & ((Rf @ Rf) == 0)
    inn = w.inner(w.X, 4.0)
    return [row('связей на событие (внутри)', L[inn].sum(1).mean() + L[:, inn].sum(0).mean(), None, None, 'вход + выход; у засева растёт как ln N')]

def m_clocks(w, R, rs, q):
    rows = []; allr = []
    for lo, hi in ((8, 12), (14, 20)):
        ratio = []; rap = []
        for a, b, t in pairs_by_tau(w, R, lo, hi, rs, 10 if q else 30, 1):
            ch = longest(R, a, b); ratio.append((len(ch) - 1) / math.sqrt(w.rho * w.vol(t)))
            d = w.X[b] - w.X[a]; rap.append(abs(math.atanh(d[1] / d[0])) if w.kind == 'flat' else 0)
        if ratio:
            rows.append(row(f'часы: длина цепочки / √(ρV), τ ∈ [{lo},{hi})', np.mean(ratio), np.std(ratio) / math.sqrt(len(ratio)), 2.0, 'Бриттуэлл–Грегори: → 2 у засева при N → ∞'))
            allr += list(zip(rap, ratio))
    if w.kind == 'flat' and allr:
        A = np.array(allr); lo_, hi_ = A[A[:, 0] < 0.4, 1], A[A[:, 0] > 0.8, 1]
        if len(lo_) > 3 and len(hi_) > 3:
            rows.append(row('изотропия часов: быстрые / медленные', hi_.mean() / lo_.mean(), None, 1.0, f'медленные {lo_.mean():.3f}, быстрые {hi_.mean():.3f}'))
    return rows

def m_floor(w, R, rs, q):
    if w.kind != 'flat': return []
    i, j = np.nonzero(R); d = w.X[j] - w.X[i]; s2 = d[:, 0] ** 2 - d[:, 1] ** 2
    return [row('дно интервала: min τ² среди пар', s2.min(), None, None, 'у засева → 0; у кристалла конечно'),
            row('доля пар с τ² < 0.1', np.mean(s2 < 0.1), None, None)]

def m_window(w, R, rs, q):
    if w.kind != 'flat': return []
    res = []
    bot = np.nonzero(w.X[:, 0] < -w.region + 3)[0]; top = np.nonzero(w.X[:, 0] > w.region - 3)[0]
    for _ in range(15 if q else 50):
        a, b = rs.choice(bot), rs.choice(top)
        if not R[a, b]: continue
        ch = longest(R, a, b); sh = w.W[ch]
        th = np.unwrap(np.arctan2(sh[:, 1], sh[:, 0])); h = len(th) // 2
        res.append((np.sqrt((np.diff(sh, axis=0) ** 2).sum(1)).mean(), th[h] - th[0], th[-1] - th[h], len(ch)))
    if len(res) < 5: return []
    A = np.array(res)
    return [row('длинная цепочка: событий от низа до верха', A[:, 3].mean()),
            row('шаг цепочки по окну', A[:, 0].mean(), A[:, 0].std() / math.sqrt(len(A)), 0.905 * w.r, 'эталон — случайные точки окна'),
            row('намотка: связь половин цепочки', np.corrcoef(A[:, 1], A[:, 2])[0, 1], 1 / math.sqrt(len(A)), 0.0, '≠ 0 — сохраняющееся вращение в окне (заряд)')]

def m_curvature(w, R, rs, q):
    if w.kind != 'ds': return []
    rows = []
    for tau in (4.0, 8.0, 12.0):
        if tau > 1.6 * w.ell: continue
        V0 = tau ** 2 / 2; d = []
        for _ in range(150 if q else 600):
            p, qq, _ = random_diamond(w, tau, rs); d.append(count_in(w, p, qq) / (w.rho * V0) - 1)
        rows.append(row(f'кривизна: N/(ρ·τ²/2) − 1 при τ = {tau:g}', np.mean(d), np.std(d) / math.sqrt(len(d)), exact_area(w.ell, tau) / V0 - 1, 'эталон — точная площадь ромба в dS2 (один мир: ошибка занижена)'))
    rows.append(row('скалярная кривизна по построению R = 2/ℓ²', 2 / w.ell ** 2, None, None))
    return rows

def m_gauge(w, R, rs, q):
    if not getattr(w, 'kappa', None): return []
    from twist import make, field, _ricci_scalar, metric2
    S, tang, frame = make(w.kappa, True)
    F = [field(frame, *rs.uniform(-3, 3, 2)) for _ in range(3)]; Rs = [_ricci_scalar(lambda a, b: metric2(tang, a, b), *rs.uniform(-3, 3, 2), 1e-3) for _ in range(3)]
    return [row('поле закрутки окна F', np.mean(F), np.std(F), -2 * w.kappa ** 2, 'геометрия среза, однородное поле'),
            row('кривизна того же среза R', np.mean(Rs), np.std(Rs), 0.0, 'поле без гравитации')]

MODULES = [('густота', m_density), ('размерность', m_dimension), ('ромбы и бусты', m_diamonds), ('гипероднородность', m_hyper),
           ('связи', m_links), ('часы', m_clocks), ('дно интервала', m_floor), ('окно и намотка', m_window),
           ('кривизна', m_curvature), ('калибровочное поле', m_gauge)]

PRESETS = {
    'gold-flat':     dict(make=lambda rs: golden_flat(4.0, 22.0, rs), rule='proj', what='золотой кристалл 1+1, плоский срез, порядок по срезу'),
    'poisson-flat':  dict(make=lambda rs: poisson_flat(4.0, 22.0, rs), rule='proj', what='засев Пуассона 1+1, плоский'),
    'gold-trace':    dict(make=lambda rs: golden_flat(4.0, 22.0, rs), rule='trace', what='золотой кристалл 1+1, порядок по полному следу'),
    'poisson-trace': dict(make=lambda rs: poisson_flat(4.0, 22.0, rs), rule='trace', what='засев + случайное место в окне, порядок по следу'),
    'gold-ds':       dict(make=lambda rs: golden_ds(4.0, 12.0, 1.5, rs), rule='proj', what='золотой кристалл на изогнутом срезе: де Ситтер 1+1, ℓ = 12'),
    'poisson-ds':    dict(make=lambda rs: poisson_ds(4.0, 12.0, 1.5, rs), rule='proj', what='засев на де Ситтере 1+1, ℓ = 12'),
    'gold-twist':    dict(make=lambda rs: golden_flat(4.0, 22.0, rs), rule='proj', kappa=0.05, what='золотой кристалл 1+1 + геометрия закрученного среза (κ = 0.05)'),
}

def verdict(r):
    v, e, ref = r['value'], r['err'], r['ref']
    if ref is None or v is None: return ''
    if e: z = abs(v - ref) / e; return f'{z:.1f}σ'
    return f'{v - ref:+.3g}'

def run(name, seed=0, quick=False):
    P = PRESETS[name]; rs = np.random.default_rng(seed); t0 = time.time()
    w = P['make'](rs); w.kappa = P.get('kappa'); R = build_order(w, P['rule'])
    out = dict(preset=name, what=P['what'], seed=seed, rule=P['rule'], n=w.n, window_r=w.r, rho=w.rho, sections=[])
    print(f'== {name}: {P["what"]}; событий {w.n}, порядок «{P["rule"]}»  [{time.time() - t0:.0f} с]'); sys.stdout.flush()
    for title, fn in MODULES:
        t1 = time.time()
        try: rows = fn(w, R, rs, quick)
        except Exception as ex: rows = [row('ошибка модуля', None, note=repr(ex))]
        if not rows: continue
        out['sections'].append(dict(title=title, rows=rows, seconds=round(time.time() - t1, 1)))
        for r in rows:
            vs = '—' if r['value'] is None else f"{r['value']:.4g}"
            print(f"  [{title}] {r['name']}: {vs}{' ± %.2g' % r['err'] if r['err'] else ''}{'   (эталон %s, %s)' % (r['ref'] if not isinstance(r['ref'], float) else '%.4g' % r['ref'], verdict(r)) if r['ref'] is not None else ''}{'   — ' + r['note'] if r['note'] else ''}")
        sys.stdout.flush()
    out['seconds'] = round(time.time() - t0, 1)
    tag = f'{name}_s{seed}' + ('_q' if quick else '')
    os.makedirs('/home/claude/data/universe', exist_ok=True); os.makedirs('/home/claude/docs/runs', exist_ok=True)
    json.dump(out, open(f'/home/claude/data/universe/{tag}.json', 'w'), ensure_ascii=False, indent=1)
    with open(f'/home/claude/docs/runs/{tag}.md', 'w') as f:
        f.write(f'# Паспорт мира `{tag}`\n\n{P["what"]}. Событий {w.n}, порядок «{P["rule"]}», окно r = {w.r:.3f}, зерно {seed}. Счёт {out["seconds"]} с.\n\n')
        for s in out['sections']:
            f.write(f'## {s["title"]}\n\n| величина | значение | ± | эталон | отклонение | комментарий |\n|---|---|---|---|---|---|\n')
            for r in s['rows']:
                vs = '—' if r['value'] is None else f"{r['value']:.4g}"
                f.write(f"| {r['name']} | {vs} | {'%.2g' % r['err'] if r['err'] else ''} | {'' if r['ref'] is None else ('%.4g' % r['ref'] if isinstance(r['ref'], float) else r['ref'])} | {verdict(r)} | {r['note']} |\n")
            f.write('\n')
    print(f'паспорт: docs/runs/{tag}.md, data/universe/{tag}.json  [{out["seconds"]} с]')
    return out

def compare(a, b, seed=0, quick=False):
    """два мира рядом: паспорт с двумя колонками (обычно кристалл против засева)."""
    A, B = run(a, seed, quick), run(b, seed, quick)
    tag = f'{a}_vs_{b}_s{seed}' + ('_q' if quick else '')
    rowsB = {(s['title'], r['name']): r for s in B['sections'] for r in s['rows']}
    def fmt(r):
        if r is None or r['value'] is None: return '—'
        return f"{r['value']:.4g}" + (f" ± {r['err']:.2g}" if r['err'] else '')
    with open(f'/home/claude/docs/runs/{tag}.md', 'w') as f:
        f.write(f'# Два мира рядом: `{a}` и `{b}`\n\nA: {A["what"]} ({A["n"]} событий, порядок «{A["rule"]}»).\n\nB: {B["what"]} ({B["n"]} событий, порядок «{B["rule"]}»). Зерно {seed}.\n\n')
        for s in A['sections']:
            f.write(f'## {s["title"]}\n\n| величина | A | B | эталон | комментарий |\n|---|---|---|---|---|\n')
            for r in s['rows']:
                rb = rowsB.get((s['title'], r['name']))
                ref = '' if r['ref'] is None else ('%.4g' % r['ref'] if isinstance(r['ref'], float) else r['ref'])
                f.write(f"| {r['name']} | {fmt(r)} | {fmt(rb)} | {ref} | {r['note']} |\n")
            f.write('\n')
    print(f'сравнение: docs/runs/{tag}.md')

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    seed = int(args[1]) if len(args) > 1 and args[1].isdigit() else 0
    if '--vs' in sys.argv:
        compare(args[0], sys.argv[sys.argv.index('--vs') + 1], seed, '--quick' in sys.argv)
    else:
        run(args[0] if args else 'gold-flat', seed, '--quick' in sys.argv)
