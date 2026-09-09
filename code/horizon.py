"""Горизонты: есть ли области, из которых нельзя выбраться.

В искривлённой области число действующих шкал меньше, значит событий,
сравнимых с внешними, меньше. Вопрос: может ли это дойти до полной
изоляции — когда из области ничего не выходит наружу.
"""
import numpy as np
import sys
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from measure_constant import local_density, metric, causal_order


def build_world(N, k, alpha, mass, spread, centre, seed, radius=None):
    rs = np.random.default_rng(seed)
    bg = rs.random((N, k))
    body = np.clip(centre + rs.normal(0, spread, (mass, k)), 0, 1)
    X = np.vstack([bg, body])
    inside = np.zeros(len(X), dtype=bool)
    inside[N:] = True
    if radius is None:
        radius = 0.2
    dens = local_density(X, radius)
    W = metric(dens, k, alpha)
    R = causal_order(X, W)
    return X, R, W, inside, dens


def escape_stats(X, R, centre, rad):
    """Сколько связей ведёт ИЗ области наружу и снаружи внутрь."""
    d = np.linalg.norm(X - centre, axis=1)
    inn = d < rad
    out = ~inn
    if inn.sum() < 5 or out.sum() < 5:
        return None
    # R[i,j] значит j раньше i
    out_from_in = int(R[np.ix_(out, inn)].sum())   # наружу: внутреннее раньше внешнего
    in_from_out = int(R[np.ix_(inn, out)].sum())   # внутрь
    npairs = int(inn.sum())*int(out.sum())
    return dict(n_in=int(inn.sum()), n_out=int(out.sum()),
                out_rate=out_from_in/max(npairs, 1),
                in_rate=in_from_out/max(npairs, 1),
                asym=(out_from_in-in_from_out)/max(out_from_in+in_from_out, 1))
