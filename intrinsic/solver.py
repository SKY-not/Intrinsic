"""Planar calibration and held-out validation; no camera/SDK dependencies."""
from __future__ import annotations

import cv2
import numpy as np

from .board import Board
from .session import Session, utc_now


def residual_metrics(residuals) -> dict:
    """Statistics of Euclidean corner errors (sample variance, ddof=1)."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite Nx2 residuals")
    distances = np.linalg.norm(values, axis=1)
    variance = float(np.var(distances, ddof=1)) if len(distances) > 1 else None
    return {"point_count": len(distances), "mre_px": float(distances.mean()),
            "rmse_px": float(np.sqrt(np.mean(distances**2))),
            "std_px": None if variance is None else float(np.sqrt(variance)),
            "variance_px2": variance, "max_error_px": float(distances.max())}


def view_metrics(views) -> dict | None:
    return residual_metrics(np.concatenate([v["residuals"] for v in views])) if views else None


def add_result_metrics(result: dict) -> None:
    """Enrich a loaded result in memory, including legacy results."""
    for group in (result, result.get("validation")):
        if group is None:
            continue
        views = group.get("views", [])
        for view in views:
            view["metrics"] = residual_metrics(view["residuals"])
        group["metrics"] = view_metrics(views)
        group["image_count"] = len(views)


def view_constraints(object_points: np.ndarray, observations: list[np.ndarray], size: tuple[int, int],
                     object_subsets=None) -> dict:
    # Zhang's v12 and (v11-v22), with image and object coordinates normalized
    # to avoid the pixel/meter scale dominating the singular values.
    xy = object_points[:, :2].astype(float)
    center, scale = xy.mean(axis=0), np.max(np.ptp(xy, axis=0))
    subsets = [object_points] * len(observations) if object_subsets is None else object_subsets
    def v(a, b):
        return np.array([a[0]*b[0], a[0]*b[1]+a[1]*b[0], a[1]*b[1],
                         a[2]*b[0]+a[0]*b[2], a[2]*b[1]+a[1]*b[2], a[2]*b[2]])
    constraints = []
    for objects, points in zip(subsets, observations):
        xy = (objects[:, :2].astype(float) - center) / scale
        pixels = (points - np.asarray(size) / 2) / max(size)
        h, _ = cv2.findHomography(xy, pixels, 0)
        if h is None or not np.isfinite(h).all():
            raise ValueError("Cannot estimate board homography")
        for row in (v(h[:, 0], h[:, 1]), v(h[:, 0], h[:, 0]) - v(h[:, 1], h[:, 1])):
            constraints.append(row / max(np.linalg.norm(row), 1e-15))
    singular = np.linalg.svd(constraints, compute_uv=False)
    strength = float(singular[-2] / singular[0])
    if strength < 1e-4:
        raise ValueError("Degenerate views: tilt the board about BOTH horizontal and vertical axes; do not keep it front-facing")
    all_points = np.concatenate(observations)
    bins = np.minimum((all_points / np.asarray(size) * [4, 3]).astype(int), [3, 2])
    covered = len(set(map(tuple, bins.tolist())))
    warnings = []
    if strength < .005:
        warnings.append("Weak orientation diversity: add views tilted about both board axes")
    if covered < 10:
        warnings.append(f"Limited image coverage ({covered}/12 cells); add edge/corner views")
    return {"constraint_singular_values": singular.tolist(), "constraint_strength": strength,
            "coverage_cells": covered, "coverage_total": 12, "warnings": warnings}


def model_diagnostics(K: np.ndarray, D: np.ndarray, size: tuple[int, int]) -> list[str]:
    if K.shape != (3, 3) or D.size != 5 or not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("Non-finite or invalid intrinsic model")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Focal lengths must be positive")
    if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-9:
        raise ValueError("Unsupported intrinsic matrix")
    w, h = size
    warnings = []
    if not (0 <= K[0, 2] < w and 0 <= K[1, 2] < h):
        warnings.append("Principal point lies outside the image; inspect calibration")
    # Sample both radial monotonicity and the full distortion Jacobian over the
    # normalized image rectangle. This is a diagnostic, not a global proof.
    xs = np.linspace((0-K[0, 2])/K[0, 0], (w-K[0, 2])/K[0, 0], 41)
    ys = np.linspace((0-K[1, 2])/K[1, 1], (h-K[1, 2])/K[1, 1], 41)
    x, y = np.meshgrid(xs, ys)
    r2 = x*x + y*y
    k1, k2, p1, p2, k3 = D.ravel()
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    derivative = 1 + 3*k1*r2 + 5*k2*r2**2 + 7*k3*r2**3
    slope = k1 + 2*k2*r2 + 3*k3*r2**2
    jxx = radial + 2*x*x*slope + 2*p1*y + 6*p2*x
    jyy = radial + 2*y*y*slope + 6*p1*y + 2*p2*x
    jxy = 2*x*y*slope + 2*p1*x + 2*p2*y
    if np.any(derivative <= 0) or np.any(jxx*jyy-jxy*jxy <= 0):
        raise ValueError("Non-monotonic/folded distortion mapping within the sampled field; collect better views")
    return warnings


def residual_record(row: dict, points: np.ndarray, K, D, rvec, tvec, object_points, point_ids=None) -> dict:
    rotation, _ = cv2.Rodrigues(rvec)
    if np.any((object_points @ rotation.T + np.asarray(tvec).reshape(1, 3))[:, 2] <= 0):
        raise ValueError(f"Board behind camera: {row['id']}")
    projected = cv2.projectPoints(object_points, rvec, tvec, K, D)[0].reshape(-1, 2)
    residuals = points - projected
    errors = np.linalg.norm(residuals, axis=1)
    return {"id": row["id"], "image": row["image"], "role": row["role"],
            "rms_px": float(np.sqrt(np.mean(errors**2))), "max_error_px": float(errors.max()),
            "points": points.tolist(), "residuals": residuals.tolist(),
            "point_ids": list(range(len(points))) if point_ids is None else point_ids,
            "metrics": residual_metrics(residuals),
            "rvec": np.asarray(rvec).ravel().tolist(), "tvec_m": np.asarray(tvec).ravel().tolist()}


def calibrate(session: Session, exclude: list[str] | None = None) -> dict:
    excluded = set(exclude or [])
    training = [row for row in session.records if row["role"] == "calibration"]
    unknown = excluded - {row["id"] for row in training}
    if unknown:
        raise ValueError(f"Unknown/non-training excluded frames: {sorted(unknown)}")
    rows = [r for r in training if r["id"] not in excluded]
    if len(rows) < session.config["calibration"]["min_views"]:
        raise ValueError(f"Need at least {session.config['calibration']['min_views']} calibration images; got {len(rows)}")
    for row in rows:
        session.checked_image(row)
    board = Board.from_config(session.config)
    pairs = [board.correspondences(r["observation"]) for r in rows]
    objects, observations = map(list, zip(*pairs))
    size = tuple(session.manifest["image_size"])
    diagnostics = view_constraints(board.points, observations, size, objects)
    rms, K, D, rotations, translations, std_k, _, _ = cv2.calibrateCameraExtended(
        objects, observations, size, None, None,
        flags=0, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-10))
    warnings = diagnostics.pop("warnings") + model_diagnostics(K, D, size)
    views = [residual_record(r, p, K, D, rv, tv, obj, board.point_ids(r["observation"]["ids"]))
             for r, p, rv, tv, obj in zip(rows, observations, rotations, translations, objects)]
    threshold = session.config["calibration"]["review_rms_px"]
    mark_reviews(views, threshold)
    review = [v["id"] for v in views if v["rms_px"] > threshold]
    if review:
        warnings.append(f"{len(review)} views exceed review RMS {threshold:g} px; inspect before excluding")
    std = std_k.ravel()[:9]
    if not np.isfinite(std).all() or not np.isfinite(rms):
        raise ValueError("Unstable calibration uncertainty; add diverse views")
    return {"schema_version": 1, "created_utc": utc_now(), "model": "opencv_pinhole_brown5",
            "source_session": str(session.path),
            "config": session.config, "camera": session.manifest["camera"], "image_size": list(size),
            "K": K.tolist(), "D": D.ravel().tolist(), "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
            "std_order": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"],
            "std_intrinsics": std.tolist(), "rms_px": float(rms),
            "max_error_px": max(v["max_error_px"] for v in views), "diagnostics": diagnostics,
            "excluded_frames": sorted(excluded), "review_frames": review,
            "warnings": warnings, "views": views, "metrics": view_metrics(views), "image_count": len(views),
            "validation_status": "not_run", "metric_accuracy_verified": False}


def validate(session: Session, result: dict) -> dict:
    if result.get("model") != "opencv_pinhole_brown5" or result.get("schema_version") != 1:
        raise ValueError("Unsupported calibration result model/schema")
    for key in ("image_size", "camera"):
        if result[key] != session.manifest[key]:
            raise ValueError(f"Calibration/session mismatch: {key}")
    if result["source_session"] != str(session.path):
        raise ValueError("Validation must use the source session's held-out images")
    K, D = np.asarray(result["K"], float), np.asarray(result["D"], float)
    model_diagnostics(K, D, tuple(result["image_size"]))
    rows = [r for r in session.records if r["role"] == "validation"]
    if not rows:
        raise ValueError("No held-out validation images; capture with V or import --role validation")
    board = Board.from_config(session.config)
    views = []
    for row in rows:
        session.checked_image(row)
        objects, pixels = board.correspondences(row["observation"])
        point_ids = board.point_ids(row["observation"]["ids"])
        ok, rotations, translations, _ = cv2.solvePnPGeneric(objects, pixels, K, D, flags=cv2.SOLVEPNP_IPPE)
        candidates = []
        if ok:
            for rv, tv in zip(rotations, translations):
                if np.all((objects @ cv2.Rodrigues(rv)[0].T + tv.reshape(1, 3))[:, 2] > 0):
                    candidates.append(residual_record(row, pixels, K, D, rv, tv, objects, point_ids))
        if not candidates:
            raise ValueError(f"Cannot estimate positive-depth validation pose: {row['id']}")
        best = min(candidates, key=lambda x: x["rms_px"])
        rv, tv = cv2.solvePnPRefineLM(objects, pixels, K, D, np.array(best["rvec"]), np.array(best["tvec_m"]))
        refined = residual_record(row, pixels, K, D, rv, tv, objects, point_ids)
        views.append(refined if refined["rms_px"] <= best["rms_px"] else best)
    mark_reviews(views, session.config["calibration"]["review_rms_px"])
    rms = float(np.sqrt(sum(v["rms_px"]**2 * len(v["points"]) for v in views) /
                        sum(len(v["points"]) for v in views)))
    return {"created_utc": utc_now(), "rms_px": rms,
            "max_error_px": max(v["max_error_px"] for v in views), "views": views,
            "metrics": view_metrics(views), "image_count": len(views),
            "warnings": ["Fewer than five held-out views"] if len(views) < 5 else [],
            "interpretation": "Held-out image reprojection with fitted per-image pose; not metric accuracy ground truth"}


def mark_reviews(views, threshold):
    for view in views:
        rms = view.get("rms_px")
        if rms is None or not np.isfinite(rms):
            view.update(review_status="not_checked", review_reason="No finite reprojection RMS")
        elif rms > threshold:
            view.update(review_status="needs_review", review_reason=f"RMS {rms:.4f} px exceeds {threshold:.4f} px")
        else:
            view.update(review_status="not_flagged", review_reason=f"RMS <= {threshold:.4f} px; not a manual approval")


def apply_validation(session, result, mode="auto"):
    """Attach optional held-out evaluation without hiding an evaluation failure."""
    if mode not in ("auto", "required", "off"):
        raise ValueError("Validation mode must be auto, required or off")
    result.pop("validation", None)
    result["validation_mode"] = mode
    if mode == "off":
        result.update(validation_status="skipped_off", validation_reason="Validation disabled by request")
    elif not any(r["role"] == "validation" for r in session.records):
        result.update(validation_status="missing_required" if mode == "required" else "skipped_no_data",
                      validation_reason="No held-out validation images")
    else:
        try:
            result["validation"] = validate(session, result)
            result.update(validation_status="evaluated", validation_reason="Held-out images evaluated with fixed intrinsics")
        except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as error:
            result.update(validation_status="failed", validation_reason=str(error))
    return result["validation_status"] not in ("missing_required", "failed")
