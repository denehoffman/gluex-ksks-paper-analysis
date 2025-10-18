from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from gluex_ksks_paper_analysis.environment import BLACK, BLUE, GREEN, PURPLE, RED

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

RFL_RANGE: tuple[float, float] = (0.0, 0.2)


@runtime_checkable
class ComponentPDF(Protocol):
    def pdf(self, x: ArrayLike, theta: ArrayLike) -> NDArray: ...
    def pdf1d(self, x: ArrayLike, theta: ArrayLike) -> NDArray: ...
    def log_penalty(self, theta: ArrayLike) -> float: ...
    def n_theta(self) -> int: ...
    def theta_bounds(self) -> list[tuple[float | None, float | None]]: ...
    def theta0(self, data: ArrayLike) -> NDArray: ...


@dataclass
class StepwisePDF2D(ComponentPDF):
    x_edges: NDArray
    y_edges: NDArray
    base_density: NDArray
    sigma_alpha: NDArray
    alpha_bounds: tuple[float | None, float | None] = (0.2, 5.0)
    _: KW_ONLY
    bblite: bool = False

    @staticmethod
    def from_mc(
        t1: ArrayLike,
        t2: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        bins: int = 2000,
        limits: tuple[float, float] = (0.0, 2.0),
        eps: float = np.finfo(float).tiny,
        bblite: bool = False,
    ) -> StepwisePDF2D:
        t1 = np.asarray(t1)
        t2 = np.asarray(t2)
        weights = (
            np.asarray(weights)
            if weights is not None
            else np.ones_like(t1, dtype=float)
        )
        counts, x_edges, y_edges = np.histogram2d(
            t1, t2, bins=[bins, bins], range=[limits, limits], weights=weights
        )
        counts = np.maximum(counts, eps)
        x_widths = np.diff(x_edges)
        y_widths = np.diff(y_edges)
        area = np.outer(x_widths, y_widths)
        total = counts.sum()
        density = counts / (total * area)
        sigma_alpha = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        return StepwisePDF2D(x_edges, y_edges, density, sigma_alpha, bblite=bblite)

    def n_theta(self) -> int:
        if self.bblite:
            nx, ny = self.base_density.shape
            return nx * ny
        return 0

    def theta_bounds(self) -> list[tuple[float | None, float | None]]:
        if self.bblite:
            return [self.alpha_bounds] * self.n_theta()
        return []

    def theta0(self, _data: ArrayLike) -> NDArray:
        if self.bblite:
            return np.ones(self.n_theta(), dtype=float)
        return np.array([])

    def _norm_and_eff(self, alpha_flat: ArrayLike) -> tuple[float, NDArray]:
        nx, ny = self.base_density.shape
        alpha = (
            np.asarray(alpha_flat).reshape(nx, ny)
            if self.bblite
            else np.ones((nx, ny), dtype=float)
        )
        eff = alpha * self.base_density
        x_widths = np.diff(self.x_edges)
        y_widths = np.diff(self.y_edges)
        area = np.outer(x_widths, y_widths)
        norm = np.sum(eff * area)
        return norm, eff

    def pdf(self, x: ArrayLike, theta: ArrayLike) -> NDArray:
        x = np.asarray(x)
        t1, t2 = x[:, 0], x[:, 1]
        ix = np.searchsorted(self.x_edges, t1, side='right') - 1
        iy = np.searchsorted(self.y_edges, t2, side='right') - 1
        valid = (
            (ix >= 0)
            & (iy >= 0)
            & (ix < self.base_density.shape[0])
            & (iy < self.base_density.shape[1])
        )

        dens = np.full(x.shape[0], np.finfo(float).tiny, dtype=float)
        norm, eff = self._norm_and_eff(theta)
        dens[valid] = eff[ix[valid], iy[valid]] / norm  # ty: ignore
        return np.maximum(dens, np.finfo(float).tiny)

    def pdf1d(self, x: ArrayLike, theta: ArrayLike) -> NDArray:
        x = np.asarray(x)
        ix = np.searchsorted(self.x_edges, x, side='right') - 1
        valid = (ix >= 0) & (ix < self.base_density.shape[0])

        dens = np.full(x.shape[0], np.finfo(float).tiny, dtype=float)
        norm, eff = self._norm_and_eff(theta)
        y_widths = np.diff(self.y_edges)
        eff1d = eff @ y_widths
        dens[valid] = eff1d[ix[valid]] / norm  # ty: ignore
        return np.maximum(dens, np.finfo(float).tiny)

    def log_penalty(self, theta: ArrayLike) -> float:
        if self.bblite:
            r = (np.asarray(theta) - 1.0) / self.sigma_alpha.flatten()
            return 0.5 * np.dot(r, r)
        return 0.0


class SharedTauExponential2D(ComponentPDF):
    def __init__(self, *, lda0: float = 110.0):
        self.lda0 = lda0
        super().__init__()

    def n_theta(self) -> int:
        return 1

    def theta_bounds(self) -> list[tuple[float | None, float | None]]:
        return [(1e-12, None)]

    def theta0(self, _data: ArrayLike) -> NDArray:
        return np.asarray([self.lda0], dtype=float)

    def _norm(self, lda: float) -> float:
        a, b = RFL_RANGE
        z1 = np.exp(-lda * a) - np.exp(-lda * b)
        z = z1 * z1
        return max(float(z), np.finfo(float).tiny)

    def _norm1d(self, lda: float) -> float:
        a, b = RFL_RANGE
        z1 = np.exp(-lda * a) - np.exp(-lda * b)
        return max(float(z1), np.finfo(float).tiny)

    def pdf(self, x: ArrayLike, theta: ArrayLike) -> NDArray:
        x = np.asarray(x)
        t1, t2 = x[:, 0], x[:, 1]
        lda = float(np.asarray(theta)[0])
        return np.exp(-lda * (t1 + t2)) * lda * lda / self._norm(lda)

    def pdf1d(self, x: ArrayLike, theta: ArrayLike) -> NDArray:
        x = np.asarray(x)
        lda = float(np.asarray(theta)[0])
        return np.exp(-lda * x) * lda / self._norm1d(lda)

    def log_penalty(self, _theta: ArrayLike) -> float:
        return 0.0


def extended_unbinned_nll(
    params: ArrayLike,
    data: ArrayLike,
    weights: ArrayLike,
    signal: ComponentPDF,
    background: ComponentPDF,
) -> float:
    params = np.asarray(params)
    data = np.asarray(data)
    weights = np.asarray(weights)
    n_s = params[0]
    n_b = params[1]
    n_theta_s = signal.n_theta()
    n_theta_b = background.n_theta()
    theta_s = params[2 : 2 + n_theta_s]
    theta_b = params[2 + n_theta_s : 2 + n_theta_s + n_theta_b]

    p_s = signal.pdf(data, theta_s)
    p_b = background.pdf(data, theta_b)
    terms = np.log(n_s * p_s + n_b * p_b)
    nll = (n_s + n_b) - np.sum(weights * terms)
    nll += float(signal.log_penalty(theta_s)) + float(background.log_penalty(theta_b))
    nll *= 2.0
    print(nll)
    return float(nll)


@dataclass
class FitResult:
    n_s: float
    n_b: float
    theta_s: NDArray
    theta_b: NDArray
    success: bool
    message: str
    signal: ComponentPDF
    background: ComponentPDF
    v: NDArray
    denom: NDArray

    def get_sweights(
        self,
        data: ArrayLike,
    ) -> NDArray:
        data = np.asarray(data)
        p_s = self.signal.pdf(data, self.theta_s)
        p_b = self.background.pdf(data, self.theta_b)

        denom = np.maximum(
            self.n_s * p_s + self.n_b * p_b, np.sqrt(np.finfo(float).tiny)
        )
        return (self.v[0, 0] * p_s + self.v[0, 1] * p_b) / denom

    def plot_projection(
        self,
        data: ArrayLike,
        weights: ArrayLike,
    ):
        plt.style.use('gluex_ksks_paper_analysis.style')
        data = np.asarray(data)[:, 0]
        weights = np.asarray(weights)
        rfls = np.linspace(*RFL_RANGE, 2000)

        p_sig = self.signal.pdf1d(rfls, self.theta_s)
        p_bkg = self.background.pdf1d(rfls, self.theta_b)
        sig = p_sig * self.n_s
        bkg = p_bkg * self.n_b
        tot = sig + bkg

        nbins = 200
        edges = np.linspace(RFL_RANGE[0], RFL_RANGE[1], nbins + 1)
        counts, _ = np.histogram(data, bins=edges, weights=weights)
        w2, _ = np.histogram(data, bins=edges, weights=weights**2)
        errs = np.sqrt(w2)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        binw = widths[0]

        tot_at_centers = np.interp(centers, rfls, tot) * binw

        fig, (ax, axr) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}
        )
        fig.subplots_adjust(hspace=0)

        ax.step(edges[:-1], counts, where='post', label='Data', lw=1, color=BLUE)
        ax.errorbar(centers, counts, yerr=errs, fmt='none', lw=0.8, color=BLUE)

        ax.plot(rfls, sig * binw, label='Signal', lw=0.4, color=GREEN)
        ax.plot(rfls, bkg * binw, label='Background', lw=0.4, color=RED)
        ax.plot(rfls, tot * binw, label='Total', lw=0.8, color=PURPLE)

        ax.set_ylabel('Counts / $1$ ps')
        ax.legend()

        with np.errstate(divide='ignore', invalid='ignore'):
            resid = (counts - tot_at_centers) / errs
            resid[~np.isfinite(resid)] = 0.0

        axr.axhline(0.0, ls='--', lw=0.6, color=BLACK)
        axr.errorbar(
            centers, resid, yerr=np.ones_like(resid), fmt='o', ms=1, lw=1, color=BLACK
        )
        axr.set_ylabel('Pull')
        axr.set_xlabel('$K_S^0$ Rest-frame Lifetime (ns)')

        return fig, (ax, axr)


def fit_mixture(
    data: ArrayLike,
    weights: ArrayLike,
    signal: ComponentPDF,
    background: ComponentPDF,
    p0_yields: tuple[float, float] | None = None,
) -> FitResult:
    data = np.asarray(data)
    mask = (
        (data[:, 0] > RFL_RANGE[0])
        & (data[:, 0] < RFL_RANGE[1])
        & (data[:, 1] > RFL_RANGE[0])
        & (data[:, 1] < RFL_RANGE[1])
    )
    data = data[mask]
    weights = np.asarray(weights)
    weights = weights[mask]
    n_theta_s = signal.n_theta()
    n_theta_b = background.n_theta()
    bounds = [
        (0.0, None),
        (0.0, None),
        *signal.theta_bounds(),
        *background.theta_bounds(),
    ]
    n_total = np.sum(weights)
    n0_s, n0_b = p0_yields if p0_yields is not None else (0.5 * n_total, 0.5 * n_total)
    theta0 = np.concatenate(
        [[n0_s, n0_b], signal.theta0(data), background.theta0(data)]
    )

    res = minimize(
        fun=lambda p: extended_unbinned_nll(p, data, weights, signal, background),
        x0=theta0,
        method='Nelder-Mead',
        bounds=bounds,
    )
    print(res)

    n_s, n_b = res.x[0], res.x[1]
    theta_s_fit = res.x[2 : 2 + n_theta_s]
    theta_b_fit = res.x[2 + n_theta_s : 2 + n_theta_s + n_theta_b]

    p_s = signal.pdf(data, theta_s_fit)
    p_b = background.pdf(data, theta_b_fit)
    weights = np.asarray(weights)

    denom = np.maximum(n_s * p_s + n_b * p_b, np.sqrt(np.finfo(float).tiny))
    v_inv_ss = np.sum(weights * p_s * p_s / np.power(denom, 2))
    v_inv_sb = np.sum(weights * p_s * p_b / np.power(denom, 2))
    v_inv_bb = np.sum(weights * p_b * p_b / np.power(denom, 2))
    v = np.asarray([[v_inv_bb, -v_inv_sb], [-v_inv_sb, v_inv_ss]]) / (
        v_inv_ss * v_inv_bb - v_inv_sb * v_inv_sb
    )

    return FitResult(
        n_s,
        n_b,
        theta_s_fit.copy(),
        theta_b_fit.copy(),
        res.success,
        res.message,
        signal,
        background,
        v,
        denom,
    )
