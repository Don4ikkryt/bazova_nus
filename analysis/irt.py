"""Модель Раша: MML/EM для складності завдань, WLE для здібності учнів.

Реалізовано напряму на numpy: пакетів IRT у середовищі немає, а модель проста
й добре описана (Bock & Aitkin 1981 для EM; Warm 1989 для WLE).
"""
import numpy as np


def _p(theta, b):
    """Ймовірність правильної відповіді: theta (n,) або (n,1), b (k,)."""
    return 1.0 / (1.0 + np.exp(-(theta[..., None] - b[None, ...])))


def fit_rasch(X, n_nodes=41, max_iter=300, tol=1e-6):
    """MML-оцінка складності завдань через EM з квадратурою Гаусса.

    X: матриця (n учнів × k завдань) із 0/1 (без пропусків).
    Ідентифікація: розподіл здібностей N(0, sigma), середнє зафіксовано на 0.
    Повертає (b, sigma, n_iter).
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    nodes = np.linspace(-5, 5, n_nodes)
    p_mean = X.mean(axis=0).clip(1e-4, 1 - 1e-4)
    b = -np.log(p_mean / (1 - p_mean))          # старт: логіт-складність
    b -= b.mean()
    sigma = 1.0

    for it in range(max_iter):
        w = np.exp(-0.5 * (nodes / sigma) ** 2)
        w /= w.sum()
        P = _p(nodes, b)                         # (q, k)
        logL = X @ np.log(P.T) + (1 - X) @ np.log(1 - P.T)   # (n, q)
        logL += np.log(w)[None, :]
        logL -= logL.max(axis=1, keepdims=True)
        post = np.exp(logL)
        post /= post.sum(axis=1, keepdims=True)  # (n, q)

        nq = post.sum(axis=0)                    # (q,)
        rq = post.T @ X                          # (q, k)

        b_new = b.copy()
        for _ in range(40):                      # Ньютон по кожному завданню
            P = _p(nodes, b_new)
            num = (nq[:, None] * P).sum(axis=0) - rq.sum(axis=0)
            den = (nq[:, None] * P * (1 - P)).sum(axis=0)
            step = num / np.maximum(den, 1e-9)
            b_new += step
            if np.max(np.abs(step)) < 1e-9:
                break
        b_new -= b_new.mean()

        m = (post * nodes[None, :]).sum(axis=1)
        v = (post * (nodes[None, :] - m[:, None]) ** 2).sum(axis=1)
        sigma_new = float(np.sqrt((m.var() + v.mean())))

        delta = max(np.max(np.abs(b_new - b)), abs(sigma_new - sigma))
        b, sigma = b_new, sigma_new
        if delta < tol:
            break
    return b, sigma, it + 1


def wle(X, b, max_iter=100, tol=1e-8):
    """Warm's weighted likelihood estimate здібності + стандартна похибка."""
    X = np.asarray(X, dtype=float)
    r = X.sum(axis=1)
    theta = np.log((r + 0.5) / (X.shape[1] - r + 0.5))
    for _ in range(max_iter):
        P = _p(theta, b)                          # (n, k)
        W = P * (1 - P)
        I = W.sum(axis=1)
        d1 = (X - P).sum(axis=1) + (W * (1 - 2 * P)).sum(axis=1) / (2 * I)
        d2 = -I
        step = d1 / d2
        theta = theta - step
        theta = np.clip(theta, -6, 6)
        if np.max(np.abs(step)) < tol:
            break
    P = _p(theta, b)
    se = 1.0 / np.sqrt((P * (1 - P)).sum(axis=1))
    return theta, se


def fit_stats(X, theta, b):
    """Infit / outfit MNSQ для кожного завдання."""
    X = np.asarray(X, dtype=float)
    P = _p(theta, b)
    W = P * (1 - P)
    z2 = (X - P) ** 2 / np.maximum(W, 1e-9)
    outfit = z2.mean(axis=0)
    infit = (W * z2).sum(axis=0) / W.sum(axis=0)
    return infit, outfit


def point_biserial(X, total):
    """Кореляція «завдання ↔ решта тесту» (виправлена на власний внесок)."""
    X = np.asarray(X, dtype=float)
    out = []
    for j in range(X.shape[1]):
        rest = total - X[:, j]
        sd = X[:, j].std()
        out.append(np.corrcoef(X[:, j], rest)[0, 1] if sd > 0 else np.nan)
    return np.array(out)


def mantel_haenszel(X, focal, n_strata=10):
    """DIF за Mantel-Haenszel: alpha_MH, delta ETS, класифікація A/B/C.

    focal: булевий масив (True = фокусна група, тут пілотні класи).
    Стратифікація за сумарним балом (квантилі).
    """
    X = np.asarray(X, dtype=float)
    total = X.sum(axis=1)
    qs = np.quantile(total, np.linspace(0, 1, n_strata + 1))
    qs[-1] += 1e-6
    strata = np.digitize(total, qs[1:-1])
    res = []
    for j in range(X.shape[1]):
        num = den = 0.0
        for s in np.unique(strata):
            m = strata == s
            f, r = focal[m], ~focal[m]
            x = X[m, j]
            a = (x[f] == 1).sum(); bb = (x[f] == 0).sum()
            c = (x[r] == 1).sum(); d = (x[r] == 0).sum()
            n = a + bb + c + d
            if n == 0:
                continue
            num += a * d / n
            den += bb * c / n
        alpha = num / den if den > 0 else np.nan
        delta = -2.35 * np.log(alpha) if alpha and alpha > 0 else np.nan
        if not np.isfinite(delta):
            cat = "?"
        elif abs(delta) < 1.0:
            cat = "A"
        elif abs(delta) < 1.5:
            cat = "B"
        else:
            cat = "C"
        res.append((alpha, delta, cat))
    return res


def to_scale(theta, mean=500, sd=100, ref=None):
    """Лінійне перетворення θ у шкалу із заданим середнім і SD."""
    ref = theta if ref is None else ref
    return mean + sd * (theta - np.mean(ref)) / np.std(ref, ddof=1)
