"""Board geometry and ordered, full-resolution image observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version

import cv2
import numpy as np


from .config import DEFAULT_CONFIG, load_config, validate_config, detector_settings


@dataclass(frozen=True)
class Board:
    size: float = .07
    tag_size: float = .02
    gap: float = .01
    ids: tuple[int, ...] = (1, 2, 3, 4)
    # The supplied physical board uses a half-turn relative to OpenCV's marker images.
    rotations: tuple[float, ...] = (180, 180, 180, 180)

    def __post_init__(self):
        if len(self.ids) != 4 or len(set(self.ids)) != 4 or any(type(i) is not int or not 0 <= i < 587 for i in self.ids):
            raise ValueError("Board requires four unique tag36h11 IDs in [0, 586]")
        values = [self.size, self.tag_size, self.gap, *self.rotations]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
            raise ValueError("Board dimensions and rotations must be numeric")
        if len(self.rotations) != 4 or not np.isfinite(values).all():
            raise ValueError("Board dimensions and four rotations must be finite")
        if self.tag_size <= 0 or self.gap <= 0 or self.size <= 2 * self.tag_size + self.gap:
            raise ValueError("Board must have positive tag size, spacing and white outer margin")
        if any(not np.isclose(r % 90, 0) for r in self.rotations):
            raise ValueError("Per-tag rotation must be a multiple of 90 degrees")

    @classmethod
    def from_config(cls, config: dict) -> "Board":
        b = config["board"]
        if b["family"] != "tag36h11":
            raise ValueError("Only tag36h11 is supported")
        try:
            return cls(b["board_size_m"], b["tag_size_m"], b["gap_m"], tuple(b["ids"]), tuple(b["rotations_deg"]))
        except (TypeError, KeyError) as error:
            raise ValueError(f"Invalid board configuration: {error}") from error

    @property
    def points(self) -> np.ndarray:
        # OpenCV decoded marker order: TL, TR, BR, BL in its canonical orientation.
        local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * self.tag_size / 2
        centers = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) * (self.tag_size + self.gap) / 2
        points = []
        for center, angle in zip(centers, self.rotations):
            a = np.deg2rad(angle)
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            points.append(np.column_stack((local @ rotation.T + center, np.zeros(4))))
        return np.asarray(points, np.float32).reshape(-1, 3)

    def render(self, pixels: int = 1400) -> np.ndarray:
        """Reference image (not a substitute for checking physical print dimensions)."""
        image = np.full((pixels, pixels), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        for tag_id, angle, points in zip(self.ids, self.rotations, self.points.reshape(4, 4, 3)):
            lo = np.rint((points[:, :2].min(axis=0) / self.size + .5) * pixels).astype(int)
            hi = np.rint((points[:, :2].max(axis=0) / self.size + .5) * pixels).astype(int)
            marker = cv2.aruco.generateImageMarker(dictionary, tag_id, int(hi[0] - lo[0]))
            marker = np.rot90(marker, -int(round(angle / 90)))
            image[lo[1]:hi[1], lo[0]:hi[0]] = cv2.resize(marker, tuple(hi - lo), interpolation=cv2.INTER_NEAREST)
        return image


@dataclass
class Observation:
    ids: list[int]
    corners: np.ndarray
    valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ids": self.ids, "corners": self.corners.tolist(), "valid": self.valid,
                "issues": self.issues, "warnings": self.warnings, "metrics": self.metrics}


class Detector:
    def __init__(self, config: dict, backend: str | None = None, parameters: dict | None = None):
        self.board = Board.from_config(config)
        self.quality = config["quality"]
        self.family = config["board"]["family"]
        settings = detector_settings(config, backend, parameters)
        self.backend = backend = settings["backend"]
        if backend == "opencv":
            params = cv2.aruco.DetectorParameters()
            method = settings[backend]["corner_refinement"]
            params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_" + method)
            params.cornerRefinementWinSize = settings[backend]["window_size"]
            self.detector = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
            self.description = {"backend": backend, "version": cv2.__version__,
                                "family": self.family, "corner_refinement": method,
                                "window_size": params.cornerRefinementWinSize, "corner_permutation": [0, 1, 2, 3],
                                "pixel_offset_xy": [-.5, -.5] if method == "APRILTAG" else [0, 0]}
        elif backend == "pyapriltags":
            try:
                from pyapriltags import Detector as AprilTagDetector
            except ImportError as error:
                raise RuntimeError("pyapriltags is unavailable; install the pinned dependency in environment.yml") from error
            self.detector = AprilTagDetector(families=self.family, **settings[backend])
            native = self.detector.tag_detector_ptr.contents
            self.description = {"backend": backend, "version": version("pyapriltags"),
                                "family": self.family, "quad_decimate": float(native.quad_decimate),
                                "refine_edges": int(native.refine_edges),
                                "decode_sharpening_requested": self.detector.params["decode_sharpening"],
                                "decode_sharpening_actual": float(native.decode_sharpening),
                                "corner_permutation": [1, 0, 3, 2], "pixel_offset_xy": [-.5, -.5],
                                "estimate_tag_pose": False}
        else:
            raise ValueError(f"Unknown detector backend: {backend}")

    def detect(self, image: np.ndarray) -> Observation:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.dtype != np.uint8:
            raise ValueError("Detection requires 8-bit gray/BGR images")
        corners, ids = self._detect_opencv(gray) if self.backend == "opencv" else self._detect_pyapriltags(gray)
        return self.assemble(gray, corners, ids)

    def _detect_opencv(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        if self.detector.getDetectorParameters().cornerRefinementMethod == cv2.aruco.CORNER_REFINE_APRILTAG:
            corners = [np.asarray(corner) - .5 for corner in corners]
        return corners, [] if ids is None else ids.flatten().tolist()

    def _detect_pyapriltags(self, gray):
        detections = self.detector.detect(np.ascontiguousarray(gray), estimate_tag_pose=False)
        # Fixed decoded order and pixel-origin conversion, before shared checks.
        return [tag.corners[[1, 0, 3, 2]] - .5 for tag in detections], [int(tag.tag_id) for tag in detections]

    def assemble(self, gray: np.ndarray, corners, ids: list[int]) -> Observation:
        selected = [(int(i), np.asarray(c).reshape(4, 2)) for i, c in zip(ids, corners) if i in self.board.ids]
        issues, warnings = [], []
        if len(set(i for i, _ in selected)) != len(selected):
            issues.append("duplicate board tag ID")
        missing = set(self.board.ids) - {i for i, _ in selected}
        if missing:
            issues.append(f"missing tags: {sorted(missing)}")
        selected.sort(key=lambda pair: self.board.ids.index(pair[0]))
        points = np.asarray([c for _, c in selected], np.float32).reshape(-1, 2)
        metrics = {}
        if len(points):
            height, width = gray.shape
            if not np.isfinite(points).all() or np.any(points < 1) or np.any(points >= [width - 1, height - 1]):
                issues.append("invalid/border-clipped corners")
            else:
                quads = points.reshape(-1, 4, 2)
                edge = np.linalg.norm(quads - np.roll(quads, -1, axis=1), axis=2).min()
                mask = np.zeros_like(gray)
                cv2.fillConvexPoly(mask, cv2.convexHull(points).astype(np.int32), 255)
                roi = gray[mask > 0]
                lap = cv2.Laplacian(gray, cv2.CV_32F)[mask > 0]
                metrics = {"min_edge_px": float(edge), "blur_variance": float(lap.var()),
                           "saturation_fraction": float(np.mean(roi >= 254)),
                           "area_fraction": float(cv2.contourArea(cv2.convexHull(points)) / (width * height))}
                if edge < self.quality["tag_edge_min_px"]:
                    warnings.append("tags too small; move closer")
                if metrics["blur_variance"] < self.quality["blur_variance_min"]:
                    warnings.append("possible blur; hold board still and check focus")
                if metrics["saturation_fraction"] > self.quality["saturation_fraction_max"]:
                    warnings.append("possible clipping; inspect exposure (white board may also trigger this)")
        return Observation([i for i, _ in selected], points, not issues, issues, warnings, metrics)
