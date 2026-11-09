"""Immutable session settings and transactionally committed image observations."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import uuid

import cv2
import numpy as np

from . import __version__
from .board import Observation, validate_config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    data = json.dumps(value, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError(f"Invalid image; expected 8-bit PNG/JPEG/BMP: {path}")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError(f"Expected gray or BGR image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Image encoding failed: {path}")
    temporary = path.with_name(path.name + ".tmp")
    encoded.tofile(temporary)
    temporary.replace(path)


class Session:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._source_session = None
        if (self.path / "session_view.json").exists():
            self._load_view()
            return
        self.manifest = read_json(self.path / "session.json")
        if self.manifest.get("schema_version") != 1 or self.manifest.get("corner_space") != "raw_pixels":
            raise ValueError("Unsupported session schema or corner coordinate space")
        validate_config(self.manifest["config"])
        self._records = [read_json(p) for p in sorted((self.path / "frames").glob("*.json"))]
        for row in self._records:
            self._check_record(row)
        if len({r["id"] for r in self._records}) != len(self._records):
            raise ValueError("Duplicate frame identifiers in session")

    @classmethod
    def view(cls, source, detections, frame_ids):
        """Use alternate observations in memory; images stay in the source session."""
        if source.readonly:
            raise ValueError("Experiment views must reference an original session")
        view = cls.__new__(cls)
        view.path = source.path
        view.manifest = copy.deepcopy(source.manifest)
        view._source_session = source
        detected = {row["id"]: row for row in detections["frames"]}
        original = {row["id"]: row for row in source.records}
        ids = list(frame_ids)
        if (len(detected) != len(detections["frames"]) or len(set(ids)) != len(ids)
                or set(ids) - original.keys() or set(ids) - detected.keys()):
            raise ValueError("Invalid experiment frame selection")
        view._records = []
        for frame_id in ids:
            row, detection = copy.deepcopy(original[frame_id]), detected[frame_id]
            if detection.get("role", row["role"]) != row["role"]:
                raise ValueError("Detection/source role mismatch")
            row["observation"] = copy.deepcopy(detection["observation"])
            view._check_record(row)
            view._records.append(row)
        return view

    def _load_view(self):
        descriptor = read_json(self.path / "session_view.json")
        if descriptor.get("schema_version") != 1:
            raise ValueError("Unsupported experiment view schema")
        source_path = (self.path / descriptor["source_session"]).resolve()
        if (source_path / "session_view.json").exists():
            raise ValueError("Experiment views cannot reference other views")
        source = Session(source_path)
        detections = read_json((self.path / descriptor["detections_file"]).resolve())
        view = self.view(source, detections, descriptor["frame_ids"])
        self.manifest, self._source_session, self._records = view.manifest, source, view._records

    @property
    def readonly(self):
        return self._source_session is not None

    @classmethod
    def create(cls, root: str | Path, config: dict, camera: dict, image_size: tuple[int, int]) -> "Session":
        validate_config(config)
        if not camera.get("serial") or len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError("Camera serial/identity and positive image dimensions are required")
        safe_serial = re.sub(r"[^A-Za-z0-9_.-]", "_", str(camera["serial"]))[:80].strip(".") or "camera"
        path = Path(root) / safe_serial / new_id()
        (path / "frames").mkdir(parents=True, exist_ok=False)
        (path / "images").mkdir()
        manifest = {"schema_version": 1, "created_utc": utc_now(), "software_version": __version__,
                    "opencv_version": cv2.__version__, "camera": camera, "image_size": list(image_size),
                    "corner_space": "raw_pixels", "config": copy.deepcopy(config)}
        write_json(path / "session.json", manifest)
        return cls(path)

    @property
    def config(self) -> dict:
        return copy.deepcopy(self.manifest["config"])

    @property
    def records(self) -> list[dict]:
        return copy.deepcopy(self._records)

    def _check_record(self, row: dict):
        if row.get("role") not in ("calibration", "validation"):
            raise ValueError("Invalid frame role")
        if not re.fullmatch(r"f[0-9a-f]{12}", row.get("id", "")):
            raise ValueError("Invalid frame identifier")
        if row.get("image") != f"images/{row['id']}.png":
            raise ValueError("Invalid frame image path")
        observation = row["observation"]
        points = np.asarray(observation["corners"], float)
        if not observation["valid"] or observation["ids"] != self.config["board"]["ids"] or points.shape != (16, 2):
            raise ValueError(f"Invalid ordered observation: {row['id']}")
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= self.manifest["image_size"]):
            raise ValueError(f"Out-of-bounds corners: {row['id']}")

    def save(self, image: np.ndarray, observation: Observation, role: str, metadata: dict | None = None) -> dict:
        if self.readonly:
            raise ValueError("Experiment session views are read-only")
        if list(image.shape[1::-1]) != self.manifest["image_size"]:
            raise ValueError("Image resolution changed; start a new session")
        if image.dtype != np.uint8:
            raise ValueError("Expected 8-bit image")
        frame_id = "f" + uuid.uuid4().hex[:12]
        metadata = copy.deepcopy(metadata or {})
        detector = metadata.pop("detector", None)
        detector_index = None
        if detector is not None:
            detectors = self.manifest.setdefault("detectors", [])
            if detector not in detectors:
                detectors.append(detector)
                write_json(self.path / "session.json", self.manifest)
            detector_index = detectors.index(detector)
        row = {"id": frame_id, "role": role, "image": f"images/{frame_id}.png", "saved_utc": utc_now(),
               "observation": observation.to_dict(), "metadata": metadata}
        if detector_index is not None:
            row["detector_index"] = detector_index
        self._check_record(row)
        write_image(self.path / row["image"], image)
        # JSON is the commit marker: incomplete saves are ignored by readers.
        write_json(self.path / "frames" / f"{frame_id}.json", row)
        self._records.append(row)
        return row

    def checked_image(self, row: dict) -> np.ndarray:
        if self.readonly:
            return self._source_session.checked_image(row)
        image = read_image(self.path / row["image"])
        if list(image.shape[1::-1]) != self.manifest["image_size"]:
            raise ValueError(f"Image dimensions changed: {row['id']}")
        return image

    def new_run(self, name: str) -> Path:
        if getattr(self, "_source_session", None) is not None:
            raise ValueError("Experiment session views cannot create source-session runs")
        prefix = "calib" if name == "calibrate" else name
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M")
        base = f"{prefix}_{stamp}"
        index = 0
        while True:
            suffix = f"_{index:02d}" if index else ""
            output = self.path / "results" / (base + suffix)
            try:
                output.mkdir(parents=True, exist_ok=False)
                return output
            except FileExistsError:
                index += 1
