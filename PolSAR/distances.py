"""
Distance measures for Hermitian (covariance/coherency) matrices.
All functions take two numpy arrays A, B of shape (k, k) where k is
the number of polarimetric channels (2 for dual-pol, 3 for full-pol).
"""

import numpy as np
from numpy.linalg import inv


def _logm_hermitian(A: np.ndarray) -> np.ndarray:
    """Matrix log of a Hermitian positive-definite matrix via eigendecomposition.

    Orders of magnitude faster than scipy.linalg.logm on the small (k, k)
    covariances used here: scipy's general-purpose inverse-scaling-and-squaring
    costs ~4 ms per 4x4 call, eigh ~20 µs, and it dominated the entire forest
    runtime. Raises LinAlgError on non-PD input (the RF discards the candidate)."""
    w, V = np.linalg.eigh(A)
    if w[0] <= 0.0:
        raise np.linalg.LinAlgError("matrix is not positive definite")
    return (V * np.log(w)) @ V.conj().T


def dist_span(A: np.ndarray, B: np.ndarray) -> float:
    """Absolute difference of total backscattered power (span)."""
    return abs(np.trace(A).real - np.trace(B).real)


def dist_euclidean(A: np.ndarray, B: np.ndarray) -> float:
    """Euclidean distance of real-valued main diagonal elements only."""
    return float(np.linalg.norm(np.diag(A).real - np.diag(B).real))


def dist_frobenius(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm over all matrix elements (Eq. 6)."""
    return float(np.linalg.norm(A - B, "fro"))


def dist_wishart(A: np.ndarray, B: np.ndarray) -> float:
    """Wishart distance (Eq. 7). Not symmetric."""
    try:
        return float(np.log(abs(np.linalg.det(B))).real + np.trace(inv(B) @ A).real)
    except np.linalg.LinAlgError:
        return np.inf


def dist_wishart_sym(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Wishart distance (Eq. 8)."""
    return 0.5 * (dist_wishart(A, B) + dist_wishart(B, A))


def dist_bartlett(A: np.ndarray, B: np.ndarray) -> float:
    """Bartlett distance (Eq. 9). Semimetric."""
    try:
        mid = (A + B) / 2
        num = abs(np.linalg.det(mid))
        den = np.sqrt(abs(np.linalg.det(A)) * abs(np.linalg.det(B)))
        return float(np.log(num / (den + 1e-15)).real)
    except np.linalg.LinAlgError:
        return np.inf


def dist_revised_wishart(A: np.ndarray, B: np.ndarray) -> float:
    """Revised Wishart distance (Eq. 10). Not symmetric."""
    try:
        ratio = np.log(abs(np.linalg.det(B) / np.linalg.det(A))).real
        return float(ratio + np.trace(inv(B) @ A).real)
    except np.linalg.LinAlgError:
        return np.inf


def dist_revised_wishart_sym(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric revised Wishart distance (Eq. 11). Semimetric."""
    try:
        return float(0.5 * np.trace(inv(A) @ B + inv(B) @ A).real)
    except np.linalg.LinAlgError:
        return np.inf


def dist_log_euclidean(A: np.ndarray, B: np.ndarray) -> float:
    """Log-Euclidean distance (Eq. 13). Best performer in the paper."""
    return float(np.linalg.norm(_logm_hermitian(A) - _logm_hermitian(B), "fro").real)


# All available distances indexed by short name for easy lookup in the RF
ALL_DISTANCES = {
    "span":     dist_span,
    "euclid":   dist_euclidean,
    "frob":     dist_frobenius,
    "wishart":  dist_wishart,
    "wishart_s":dist_wishart_sym,
    "bartlett": dist_bartlett,
    "rw":       dist_revised_wishart,
    "rw_s":     dist_revised_wishart_sym,
    "log_euclid": dist_log_euclidean,
}
