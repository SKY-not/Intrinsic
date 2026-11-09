"""Board geometry and ordered, full-resolution image observations."""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version

import cv2
import numpy as np


from .config import DEFAULT_CONFIG, load_config, validate_config, detector_settings
from .quality import assess, quality_config


@dataclass(frozen=True)
class Board:
    kind = "four_apriltag"
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
        if not isinstance(b, dict):
            raise ValueError("Board configuration must be an object")
        kind = b.get("type", "four_apriltag")
        if kind == "charuco":
            if set(b) != {"type", "dictionary", "squares", "square_length_m", "marker_length_m"}:
                raise ValueError("Missing/unknown ChArUco board fields")
            return CharucoBoard(b["squares"], b["dictionary"], b["square_length_m"], b["marker_length_m"])
        required = {"family", "board_size_m", "tag_size_m", "gap_m", "ids", "rotations_deg"}
        if kind != "four_apriltag" or set(b) - required - {"type"} or required - set(b):
            raise ValueError("Missing/unknown four-AprilTag board fields")
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

    def point_ids(self, ids):
        if any(type(i) is not int or i not in self.ids for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid board tag IDs")
        return [4 * self.ids.index(tag_id) + corner for tag_id in ids for corner in range(4)]

    def correspondences(self, observation):
        obs = observation.to_dict() if isinstance(observation, Observation) else observation
        if obs["ids"] != list(self.ids):
            raise ValueError("Four ordered board tags must be visible")
        return _paired_points(self, obs)

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


class CharucoBoard:
    kind = "charuco"

    def __init__(self, squares, dictionary, square_length, marker_length):
        if not isinstance(squares, (list, tuple)) or len(squares) != 2 or any(type(i) is not int or i < 3 for i in squares):
            raise ValueError("ChArUco requires at least 3 x 3 squares")
        dimensions = (square_length, marker_length)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in dimensions) or not np.isfinite(dimensions).all():
            raise ValueError("ChArUco dimensions must be finite numeric")
        if not 0 < marker_length < square_length:
            raise ValueError("Require 0 < marker length < square length")
        dictionary_id = getattr(cv2.aruco, dictionary, None) if isinstance(dictionary, str) and dictionary.startswith("DICT_") else None
        if type(dictionary_id) is not int:
            raise ValueError(f"Unknown OpenCV ChArUco dictionary: {dictionary}")
        native_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if squares[0] * squares[1] // 2 > len(native_dictionary.bytesList):
            raise ValueError("ChArUco board requires more markers than its dictionary contains")
        self.squares, self.dictionary = tuple(squares), dictionary
        self.native = cv2.aruco.CharucoBoard(tuple(squares), square_length, marker_length, native_dictionary)
        self.native.setLegacyPattern(False)
        self.points = self.native.getChessboardCorners().astype(np.float32)

    def point_ids(self, ids):
        if any(type(i) is not int or not 0 <= i < len(self.points) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Invalid/duplicate ChArUco corner IDs")
        return list(ids)

    def correspondences(self, observation):
        obs = observation.to_dict() if isinstance(observation, Observation) else observation
        if len(obs["ids"]) < 6:
            raise ValueError("At least six ChArUco corners are required")
        return _paired_points(self, obs)

    def render(self, pixels=1400):
        x, y = self.squares
        return self.native.generateImage((pixels, round(pixels * y / x)), marginSize=0, borderBits=1)


def _paired_points(board, observation):
    ids = board.point_ids(observation["ids"])
    pixels = np.asarray(observation["corners"], np.float32)
    if not observation["valid"] or pixels.shape != (len(ids), 2) or not np.isfinite(pixels).all():
        raise ValueError("Invalid ordered observation")
    objects = board.points[ids]
    if np.linalg.matrix_rank(objects[:, :2] - objects[:, :2].mean(axis=0)) < 2:
        raise ValueError("Board points are collinear")
    return objects, pixels


@dataclass
class Observation:
    ids: list[int]
    corners: np.ndarray
    valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str | dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ids": self.ids, "corners": self.corners.tolist(), "valid": self.valid,
                "issues": self.issues, "warnings": self.warnings, "metrics": self.metrics}


class Detector:
    def __init__(self, config: dict, backend: str | None = None, parameters: dict | None = None):
        self.board = Board.from_config(config)
        self.quality = quality_config(config["quality"])
        settings = detector_settings(config, backend, parameters)
        self.backend = backend = settings["backend"]
        if backend == "opencv_charuco":
            self.detector = cv2.aruco.CharucoDetector(self.board.native)
            self.description = {"backend": backend, "version": cv2.__version__,
                                "dictionary": self.board.dictionary, "legacy_pattern": False,
                                "pixel_offset_xy": [0, 0], "uses_camera_intrinsics": False}
        elif backend == "opencv":
            self.family = config["board"]["family"]
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
            self.family = config["board"]["family"]
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

    def detect(self, image: np.ndarray, *, check_quality=True) -> Observation:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.dtype != np.uint8:
            raise ValueError("Detection requires 8-bit gray/BGR images")
        if self.backend == "opencv_charuco":
            return self._detect_charuco(gray, check_quality)
        corners, ids, regions = self._detect_opencv(gray) if self.backend == "opencv" else self._detect_pyapriltags(gray)
        return self.assemble(gray, corners, ids, check_quality=check_quality, regions=regions)

    def _detect_opencv(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        ids = [] if ids is None else ids.flatten().tolist()
        regions = list(zip(ids, corners))
        if self.detector.getDetectorParameters().cornerRefinementMethod == cv2.aruco.CORNER_REFINE_APRILTAG:
            corners = [np.asarray(corner) - .5 for corner in corners]
        return corners, ids, regions

    def _detect_pyapriltags(self, gray):
        detections = self.detector.detect(np.ascontiguousarray(gray), estimate_tag_pose=False)
        # Fixed decoded order and pixel-origin conversion, before shared checks.
        return ([tag.corners[[1, 0, 3, 2]] - .5 for tag in detections],
                [int(tag.tag_id) for tag in detections], [(int(tag.tag_id), tag.corners) for tag in detections])

    def _detect_charuco(self, gray, check_quality):
        corners, found, markers, marker_ids = self.detector.detectBoard(gray)
        ids = [] if found is None else found.ravel().tolist()
        points = np.empty((0, 2), np.float32) if corners is None else corners.reshape(-1, 2)
        order = np.argsort(ids)
        observation = Observation([ids[i] for i in order], points[order], True)
        try:
            self.board.correspondences(observation)
        except ValueError as error:
            observation.issues.append(str(error))
        self._check_bounds(gray, observation)
        observation.valid = not observation.issues
        if check_quality and observation.valid:
            regions = [] if marker_ids is None else list(zip(marker_ids.ravel(), markers))
            self._assess(gray, observation, regions)
        return observation

    @staticmethod
    def _check_bounds(gray, observation):
        points = observation.corners
        height, width = gray.shape
        if len(points) and (not np.isfinite(points).all() or np.any(points < 1) or np.any(points >= [width - 1, height - 1])):
            observation.issues.append("invalid/border-clipped corners")

    def _assess(self, gray, observation, regions):
        quality = assess(gray, regions, self.quality)
        observation.metrics = {"tags": quality["metrics"], "thresholds": quality["thresholds"]}
        observation.warnings = quality["warnings"]

    def assemble(self, gray: np.ndarray, corners, ids: list[int], *, check_quality=True, regions=None) -> Observation:
        selected = [(int(i), np.asarray(c).reshape(4, 2)) for i, c in zip(ids, corners) if i in self.board.ids]
        issues, warnings = [], []
        if len(set(i for i, _ in selected)) != len(selected):
            issues.append("duplicate board tag ID")
        missing = set(self.board.ids) - {i for i, _ in selected}
        if missing:
            issues.append(f"missing tags: {sorted(missing)}")
        selected.sort(key=lambda pair: self.board.ids.index(pair[0]))
        points = np.asarray([c for _, c in selected], np.float32).reshape(-1, 2)
        observation = Observation([i for i, _ in selected], points, not issues, issues, warnings)
        self._check_bounds(gray, observation)
        observation.valid = not observation.issues
        if check_quality and observation.valid:
            regions = selected if regions is None else [(i, p) for i, p in regions if i in self.board.ids]
            self._assess(gray, observation, regions)
        return observation
