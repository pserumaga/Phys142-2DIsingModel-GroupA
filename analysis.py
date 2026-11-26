"""PHYS 142 Final Project 2 | Analysis of 2D Ising Monte Carlo simulations.

What this module does
---------------------
* Estimate the autocorrelation function and integrated autocorrelation time.
* Inspect how the error bar changes as the block length increases.
* Estimate <m>, <|m|>, susceptibility, and heat capacity, with error bars.
* Use a blocked jackknife so correlations and nonlinear observables are handled.
* Combine independent chains without incorrectly treating their junctions as
  consecutive Monte Carlo steps.

COURSE CONVENTIONS: J = k_B = 1; lattice has L*L spins. Temperature T > 0.
Use total energy E (NOT energy per spin) as the input to this module.

Final Project 2, Part C specifically defines
    chi = (L**2 / T) * ( <m**2> - <|m|>**2 )
and heat capacity PER SPIN
    c = ( <E**2> - <E>**2 ) / (L**2 * T**2).
This chi differs from the usual signed-magnetization response susceptibility
at B=0, which substitutes <m>**2 for <|m|>**2. We follow the handout.

The inputs must ALREADY exclude burn-in. Values in m and E must be recorded
on the same sweeps. No automatic thinning: correlation is handled explicitly.

Dependencies: numpy only. Tested with Python 3.12 and NumPy 2.x.
Reference: UCSD PHYS 142 Lab 12 (statistical error estimation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np


@dataclass(frozen=True)
class AutocorrelationResult:
    """Autocorrelation statistics; tau_int is measured in recorded sweeps."""

    acf: np.ndarray
    tau_int: float
    window: int
    n_eff: float
    window_reached_limit: bool


def _series(values: np.ndarray, name: str, min_size: int = 32) -> np.ndarray:
    """Coerce a scalar time series and reject malformed/nonfinite input."""
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size < min_size:
        raise ValueError(f"{name} must be a 1D array with at least {min_size} samples")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def autocorrelation(values: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """FFT autocorrelation rho(k), normalized so rho(0)=1.

    Center before correlating. Divide covariance at lag k by (n-k) rather than
    n, a finite-sample correction. Large lag estimates become noisy, so the
    default only uses at most one quarter of the available series.

    Raises ValueError on a constant series: correlation time cannot be inferred
    from data that never fluctuate. Such a run may be trapped or too short.
    """
    a = _series(values, "values")
    n = len(a)
    if max_lag is None:
        max_lag = min(10_000, n // 4)
    if not isinstance(max_lag, (int, np.integer)) or not 1 <= max_lag < n:
        raise ValueError("max_lag must be an integer between 1 and n-1")

    centered = a - a.mean()
    var = float(np.mean(centered * centered))
    if var <= np.finfo(float).eps * max(1.0, float(np.mean(a * a))):
        raise ValueError("series is nearly constant; autocorrelation is undefined")

    # FFT uses convolution; zero padding prevents circular wraparound.
    fft_size = 1 << ((2 * n - 1).bit_length())
    z = np.fft.rfft(centered, n=fft_size)
    cov = np.fft.irfft(z * np.conj(z), n=fft_size)[: max_lag + 1]
    cov /= np.arange(n, n - max_lag - 1, -1)
    return np.asarray(cov / cov[0], dtype=float)


def correlation_diagnostics(
    values: np.ndarray, max_lag: int | None = None, window_factor: float = 6.0
) -> AutocorrelationResult:
    """Estimate tau_int = 1/2 + sum_{k=1}^W rho(k).

    The self-consistent window is W >= window_factor*tau_int. Stop before the
    first nonpositive estimated correlation to limit noisy tails. This is a
    heuristic, not a guarantee that long critical correlations were measured.
    """
    if window_factor <= 1 or not np.isfinite(window_factor):
        raise ValueError("window_factor must be greater than one")
    a = _series(values, "values")
    rho = autocorrelation(a, max_lag=max_lag)
    tau = 0.5
    window = 0
    hit_limit = True
    for lag in range(1, len(rho)):
        if rho[lag] <= 0:
            hit_limit = False
            break
        tau += float(rho[lag])
        window = lag
        if lag >= window_factor * tau:
            hit_limit = False
            break
    if hit_limit:
        warnings.warn(
            "Autocorrelation window reached max_lag. Increase run length or "
            "max_lag and check for critical slowing down.", RuntimeWarning,
            stacklevel=2,
        )
    n_eff = min(float(len(a)), len(a) / max(1.0, 2.0 * tau))
    return AutocorrelationResult(
        acf=rho, tau_int=float(tau), window=window,
        n_eff=float(n_eff), window_reached_limit=hit_limit,
    )


def blocking_curve(
    values: np.ndarray, block_sizes: Sequence[int] | None = None,
    min_blocks: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (block lengths, standard errors of the mean).

    Consecutive sweeps are kept together; leftover samples are discarded.
    Inspect the curve for a plateau before trusting a chosen block length.
    """
    a = _series(values, "values")
    if min_blocks < 2:
        raise ValueError("min_blocks must be at least 2")
    if block_sizes is None:
        block_sizes = [2**k for k in range(int(np.log2(len(a) // min_blocks)) + 1)]
    widths, errs = [], []
    for b in block_sizes:
        b = int(b)
        if b < 1 or len(a) // b < min_blocks:
            continue
        nblocks = len(a) // b
        block_means = a[: nblocks * b].reshape(nblocks, b).mean(axis=1)
        widths.append(b)
        errs.append(float(block_means.std(ddof=1) / np.sqrt(nblocks)))
    if not widths:
        raise ValueError("no block length leaves enough blocks")
    return np.array(widths, dtype=int), np.array(errs, dtype=float)


def _check_chains(
    m_chains: Sequence[np.ndarray], energy_chains: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if len(m_chains) != len(energy_chains) or len(m_chains) < 1:
        raise ValueError("supply matching nonempty lists of independent m and E chains")
    ms, es = [], []
    for i, (m, e) in enumerate(zip(m_chains, energy_chains)):
        m = _series(m, f"m_chains[{i}]", min_size=64)
        e = _series(e, f"energy_chains[{i}]", min_size=64)
        if len(m) != len(e):
            raise ValueError(f"m and E chain {i} must be recorded on the same sweeps")
        if np.any(np.abs(m) > 1 + 1e-9):
            raise ValueError("m must be magnetization PER SPIN in [-1,1]")
        ms.append(m)
        es.append(e)
    return ms, es


def _observables(moment_mean: np.ndarray, T: float, nsites: int, e_ref: float) -> np.ndarray:
    """Moments: (<m>, <|m|>, <m^2>, <E-ref>, <(E-ref)^2>)."""
    signed, absolute, m2, dE, dE2 = moment_mean
    # Centering E first avoids subtracting two huge O(Nsites^2) numbers.
    var_e = max(0.0, dE2 - dE * dE)
    chi = (nsites / T) * max(0.0, m2 - absolute * absolute)
    cv_per_spin = var_e / (nsites * T * T)
    energy_per_spin = (e_ref + dE) / nsites
    return np.array([signed, absolute, chi, cv_per_spin, energy_per_spin])


def analyze_ising_chains(
    m_chains: Sequence[np.ndarray],
    energy_chains: Sequence[np.ndarray],
    T: float,
    L: int,
    *,
    block_size: int | None = None,
    min_blocks: int = 16,
    block_factor: float = 10.0,
    max_lag: int | None = None,
) -> dict:
    """Project 2 estimates with autocorrelation-aware blocked jackknife errors.

    Inputs
    ------
    m_chains: independent equilibrated magnetization-per-spin traces.
    energy_chains: MATCHING total-energy traces, including magnetic field term.
    T, L: temperature and square lattice side length.
    block_size: user-supplied block size in sweeps, or estimate from measured
                correlations. Inspect blocking curves before final plots.

    Output
    ------
    A dictionary with 1 sigma errors for each observable, correlation times,
    block size, and counts. Different chains are never concatenated before
    computing the autocorrelation function.

    NOTE: The automatic block size is only a starting choice. The report should
    demonstrate an error plateau and consistent independent chains. If too few
    blocks remain, this function raises rather than returning false precision.
    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be positive and finite")
    if int(L) != L or L < 2:
        raise ValueError("L must be an integer at least 2")
    if min_blocks < 8:
        raise ValueError("min_blocks must be at least 8")
    if block_factor <= 1:
        raise ValueError("block_factor must exceed one")
    ms, es = _check_chains(m_chains, energy_chains)
    nsites = int(L) ** 2

    # Timescale diagnostics include quantities important for chi and heat capacity.
    # These are per-chain estimates, because crossing between separately seeded
    # chains does not correspond to an additional MCMC sweep.
    names = ("m", "abs_m", "m2", "energy", "energy_fluct2")
    taus = {name: [] for name in names}
    n_eff = {name: [] for name in names}
    hit_limits = {name: [] for name in names}
    tau_values = []
    for m, e in zip(ms, es):
        dE = e - e.mean()
        obs = (m, np.abs(m), m*m, e, dE*dE)
        for name, series in zip(names, obs):
            try:
                d = correlation_diagnostics(series, max_lag=max_lag)
            except ValueError as exc:
                if "nearly constant" not in str(exc):
                    raise
                taus[name].append(None)
                n_eff[name].append(None)
                hit_limits[name].append(False)
                continue
            taus[name].append(d.tau_int)
            n_eff[name].append(d.n_eff)
            hit_limits[name].append(d.window_reached_limit)
            tau_values.append(d.tau_int)

    if block_size is None:
        if not tau_values:
            raise ValueError("all recorded observables are constant; cannot estimate uncertainties")
        block_size = max(16, int(np.ceil(block_factor * max(tau_values))))
    else:
        if int(block_size) != block_size or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        block_size = int(block_size)

    counts = [len(m) // block_size for m in ms]
    if min(counts) < 2 or sum(counts) < min_blocks:
        raise ValueError(
            f"Insufficient independent blocks at block_size={block_size}: "
            f"{sum(counts)} total across {len(ms)} chains. "
            "Extend the runs, add independent chains, or inspect the blocking "
            "curve before justifying a shorter block size."
        )

    # All selected blocks have equal length, so combining moments gives equal
    # weight per retained sweep. The final incomplete block of each chain is cut.
    trimmed_es = [e[:count * block_size] for e, count in zip(es, counts)]
    e_ref = float(np.mean(np.concatenate(trimmed_es)))
    block_moments = []
    for m, e, count in zip(ms, es, counts):
        m = m[: count * block_size].reshape(count, block_size)
        e = e[: count * block_size].reshape(count, block_size) - e_ref
        block_moments.append(
            np.column_stack((
                m.mean(axis=1),
                np.abs(m).mean(axis=1),
                (m*m).mean(axis=1),
                e.mean(axis=1),
                (e*e).mean(axis=1),
            ))
        )
    blocks = np.concatenate(block_moments, axis=0)
    k = len(blocks)
    overall = blocks.mean(axis=0)
    value = _observables(overall, T, nsites, e_ref)

    # Leave one whole block out. Linear moments alone would not capture the
    # uncertainty in non-linear fluctuations like susceptibility and C/N.
    leave_out_moments = (k * overall[None, :] - blocks) / (k - 1)
    jack = np.array([_observables(v, T, nsites, e_ref) for v in leave_out_moments])
    jack_var = (k - 1) / k * np.sum((jack - jack.mean(axis=0))**2, axis=0)
    error = np.sqrt(jack_var)

    keys = ("m", "abs_m", "chi", "heat_capacity_per_spin", "energy_per_spin")
    result = {}
    for i, key in enumerate(keys):
        result[key] = float(value[i])
        result[key + "_err"] = float(error[i])
    result.update({
        "T": float(T),
        "L": int(L),
        "n_chains": len(ms),
        "n_samples_used": int(k * block_size),
        "n_samples_supplied": int(sum(map(len, ms))),
        "n_blocks": int(k),
        "block_size": int(block_size),
        "tau_by_observable": taus,
        "n_eff_by_observable": n_eff,
        "acf_window_hit_limit": hit_limits,
    })
    if any(any(v) for v in hit_limits.values()):
        warnings.warn(
            "Some autocorrelation estimates hit the lag limit: increase run "
            "length / max_lag before trusting statistical uncertainties.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def analyze_ising(
    m: np.ndarray, energy: np.ndarray, T: float, L: int, **kwargs
) -> dict:
    """Convenience wrapper for a single equilibrated chain."""
    return analyze_ising_chains([m], [energy], T, L, **kwargs)


if __name__ == "__main__":
    # Fast, reproducible smoke test (NOT measured Ising project data).
    rng = np.random.default_rng(142)
    x = rng.standard_normal(65_536)
    diag = correlation_diagnostics(x)
    widths, errs = blocking_curve(x)
    print("Independent random sample, approximate expected tau = 0.5")
    print(f"  tau_int={diag.tau_int:.3f}, effective N={diag.n_eff:.0f}")
    print(f"  blocking widths={widths.tolist()}")
    print(f"  last blocking error={errs[-1]:.5f}")
