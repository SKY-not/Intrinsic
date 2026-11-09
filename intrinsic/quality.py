"""Advisory marker size and sharpness checks, shared with MultiCamCalib."""
import cv2
import numpy as np


DEFAULTS = {"min_tag_edge_px": 30.0, "min_sharpness": 80.0}
LEGACY_KEYS = {"tag_edge_min_px", "blur_variance_min", "saturation_fraction_max"}


def quality_config(config=None):
    supplied = {} if config is None else config
    if not isinstance(supplied, dict):
        raise ValueError("Quality configuration must be an object")
    if set(supplied) == LEGACY_KEYS:
        # Old whole-board blur values are not comparable to per-marker sharpness.
        for name, value in supplied.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"quality.{name} must be finite and positive")
        if supplied["saturation_fraction_max"] > 1:
            raise ValueError("saturation_fraction_max must be in (0, 1]")
        supplied = {"min_tag_edge_px": supplied["tag_edge_min_px"]}
    result = {**DEFAULTS, **supplied}
    if set(result) != set(DEFAULTS):
        raise ValueError("Unknown quality setting")
    for name, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"quality.{name} must be a finite nonnegative number")
    return result


def assess(gray, regions, thresholds):
    """Laplacian variance inside each marker polygon, excluding its boundary."""
    metrics, warnings = [], []
    for marker_id, corners in regions:
        corners = np.asarray(corners, np.float32).reshape(-1, 2)
        edge = float(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1).min())
        lo = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 1, gray.shape[::-1])
        crop = gray[lo[1]:hi[1], lo[0]:hi[0]]
        if not crop.size:
            continue
        mask = np.zeros(crop.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(corners - lo).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        values = cv2.Laplacian(crop, cv2.CV_64F)[mask > 0]
        sharpness = float(values.var()) if values.size else 0.0
        metrics.append({"tag_id": int(marker_id), "min_edge_px": edge, "sharpness": sharpness})
        for code, value, limit, message in (
            ("tag_too_small", edge, thresholds["min_tag_edge_px"], "标签像素太小"),
            ("possibly_blurred", sharpness, thresholds["min_sharpness"], "标签区域疑似模糊"),
        ):
            if value < limit:
                warnings.append({"code": code, "tag_id": int(marker_id), "message": message,
                                 "value": value, "threshold": limit})
    return {"metrics": metrics, "thresholds": dict(thresholds), "warnings": warnings}


def warning_text(warning, *, display=False):
    if isinstance(warning, str):
        return warning
    message = warning["code"] if display else warning["message"]
    return f"tag {warning['tag_id']} {message} ({warning['value']:.1f} < {warning['threshold']:.1f})"
