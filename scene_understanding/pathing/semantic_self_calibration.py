"""Self-calibration layer (C) for the semantic cost map.

The idea: layers A (geometry) and B (VLM) already produce a rough cost.
We use them as a *prior* to automatically pick positive/negative exemplar
pixels (no user input, no magic thresholds), then fit a small GMM in
feature space to those exemplars and score every pixel by its posterior
probability of being obstacle vs. walkable.

Feature vector per pixel:
    [L, a, b, depth_m, edge_magnitude]
    (all z-normalised on the image's own statistics)

The GMM component count is chosen by BIC (min 1, max 4) so simple scenes
get a compact model and busy scenes get finer-grained clusters. This is
purely data-driven — we never hardcode a cluster count.

If any dependency (sklearn) is missing we return precision=0 so the
fused cost is unaffected.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from scene_understanding.pathing.semantic_cost import CostLayer, _otsu_split, _unit_minmax

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _features(img_bgr: np.ndarray, depth_m: Optional[np.ndarray]) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("self-calibration needs OpenCV")
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy).astype(np.float32)
    if depth_m is not None and depth_m.shape[:2] == (h, w):
        d = np.asarray(depth_m, dtype=np.float32)
    else:
        d = np.zeros((h, w), dtype=np.float32)
    feats = np.stack([lab[..., 0], lab[..., 1], lab[..., 2], d, edge], axis=-1)
    # z-normalise each channel by image statistics.
    for k in range(feats.shape[-1]):
        ch = feats[..., k]
        mu = float(np.nanmean(ch))
        sd = float(np.nanstd(ch))
        if sd < 1e-6:
            feats[..., k] = 0.0
        else:
            feats[..., k] = (ch - mu) / sd
    return feats.reshape(-1, feats.shape[-1]).astype(np.float32)


def _fit_gmm_by_bic(
    X: np.ndarray,
    min_k: int = 1,
    max_k: int = 4,
    rng: int = 0,
) -> Optional[object]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return None
    best = None
    best_bic = np.inf
    for k in range(min_k, max_k + 1):
        try:
            gm = GaussianMixture(n_components=k, covariance_type="diag", random_state=rng, reg_covar=1e-4)
            gm.fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best = gm
        except Exception:
            continue
    return best


def layer_self_calibration(
    img_bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    prior_cost: np.ndarray,
    sample_cap: int = 20000,
) -> CostLayer:
    """Self-calibration layer (C).

    ``prior_cost`` is expected to be the already-fused A+B cost (or any
    scalar in [0, 1] the caller trusts). We pick exemplars by Otsu split
    on this prior, fit two GMMs (positive / negative) and score every
    pixel by the log-ratio of their likelihoods.
    """

    if cv2 is None:
        h, w = prior_cost.shape[:2]
        return CostLayer(
            name="C_self_calibration",
            cost=np.zeros((h, w), dtype=np.float32),
            precision=np.zeros((h, w), dtype=np.float32),
            provenance="OpenCV missing — layer refused to build.",
        )
    h, w = img_bgr.shape[:2]
    prior = np.asarray(prior_cost, dtype=np.float32).reshape(h, w)
    split = _otsu_split(prior)
    pos_mask = prior > split
    neg_mask = prior <= split
    if pos_mask.sum() < 200 or neg_mask.sum() < 200:
        # Not enough signal — refuse rather than guess.
        return CostLayer(
            name="C_self_calibration",
            cost=np.zeros((h, w), dtype=np.float32),
            precision=np.zeros((h, w), dtype=np.float32),
            provenance=(
                f"Otsu split {split:.3f} left too few exemplars "
                f"(pos={int(pos_mask.sum())}, neg={int(neg_mask.sum())}); refused."
            ),
        )

    feats_all = _features(img_bgr, depth_m)
    rng = np.random.default_rng(0)

    def _sample(mask: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask.reshape(-1))
        if idx.size > sample_cap:
            idx = rng.choice(idx, size=sample_cap, replace=False)
        return feats_all[idx]

    pos = _sample(pos_mask)
    neg = _sample(neg_mask)
    gm_pos = _fit_gmm_by_bic(pos)
    gm_neg = _fit_gmm_by_bic(neg)
    if gm_pos is None or gm_neg is None:
        return CostLayer(
            name="C_self_calibration",
            cost=np.zeros((h, w), dtype=np.float32),
            precision=np.zeros((h, w), dtype=np.float32),
            provenance="scikit-learn not available — layer refused to build.",
        )

    # Score every pixel in batches for memory friendliness.
    log_pos = gm_pos.score_samples(feats_all)
    log_neg = gm_neg.score_samples(feats_all)
    log_ratio = (log_pos - log_neg).reshape(h, w).astype(np.float32)
    # Sigmoid of the log-ratio → posterior of "obstacle".
    cost = 1.0 / (1.0 + np.exp(-log_ratio))
    # Precision: |2·cost − 1| — confidence peaks at 0 and 1, lowest at 0.5.
    precision = np.abs(2.0 * cost - 1.0)
    cost = np.clip(cost, 0.0, 1.0).astype(np.float32)
    precision = np.clip(precision, 0.0, 1.0).astype(np.float32)

    return CostLayer(
        name="C_self_calibration",
        cost=cost,
        precision=precision,
        provenance=(
            f"Two-class GMM self-calibration. Otsu on prior at {split:.3f}. "
            f"BIC-selected components: pos={getattr(gm_pos, 'n_components', '?')}, "
            f"neg={getattr(gm_neg, 'n_components', '?')}."
        ),
        diagnostics={
            "otsu_split": float(split),
            "pos_exemplars": float(pos.shape[0]),
            "neg_exemplars": float(neg.shape[0]),
        },
    )


__all__ = ["layer_self_calibration"]
