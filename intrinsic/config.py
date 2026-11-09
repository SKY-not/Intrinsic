"""Configuration loading; persisted session snapshots remain self-contained."""
import copy
import json
from pathlib import Path
import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs/pipeline.json"
DEFAULT_COMPARE_PATH = DEFAULT_PATH.with_name("compare.json")
DEFAULT_DETECTOR = {"backend": "opencv", "opencv": {"corner_refinement": "SUBPIX", "window_size": 5},
                    "pyapriltags": {"quad_decimate": 1.0, "refine_edges": 1}}
DEFAULT_CONFIG = {
    "board": {"family": "tag36h11", "board_size_m": .07, "tag_size_m": .02,
              "gap_m": .01, "ids": [1, 2, 3, 4], "rotations_deg": [180, 180, 180, 180]},
    "camera": {"exposure_us": None, "gain_db": None, "gain_auto": "off", "resolution": "max", "lens_note": ""},
    "capture": {"detection_hz": 5, "preview_width": 1280},
    "quality": {"blur_variance_min": 40, "tag_edge_min_px": 30, "saturation_fraction_max": .4},
    "calibration": {"min_views": 15, "review_rms_px": 1.0},
    "detector": copy.deepcopy(DEFAULT_DETECTOR),
}


def detector_settings(config, backend=None, parameters=None):
    """Resolve defaults without changing an immutable historical snapshot."""
    settings = copy.deepcopy(DEFAULT_DETECTOR)
    supplied = config.get("detector", {})
    if not isinstance(supplied, dict) or set(supplied) - set(settings):
        raise ValueError("Invalid detector configuration")
    for key, value in supplied.items():
        if key == "backend":
            settings[key] = value
        else:
            if not isinstance(value, dict) or set(value) - set(settings[key]):
                raise ValueError(f"Invalid detector parameters: {key}")
            settings[key].update(value)
    if backend is not None:
        settings["backend"] = backend
    if settings["backend"] not in ("opencv", "pyapriltags"):
        raise ValueError("detector.backend must be opencv or pyapriltags")
    if parameters is not None:
        target = settings[settings["backend"]]
        if not isinstance(parameters, dict) or set(parameters) - set(target):
            raise ValueError("Unknown parameters for selected detector backend")
        target.update(parameters)
    cv = settings["opencv"]
    if cv["corner_refinement"] not in ("NONE", "SUBPIX", "CONTOUR", "APRILTAG"):
        raise ValueError("Invalid OpenCV corner_refinement")
    if type(cv["window_size"]) is not int or cv["window_size"] <= 0:
        raise ValueError("OpenCV window_size must be a positive integer")
    april = settings["pyapriltags"]
    value = april["quad_decimate"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 1:
        raise ValueError("pyapriltags quad_decimate must be finite and >= 1")
    if type(april["refine_edges"]) is not int or april["refine_edges"] not in (0, 1):
        raise ValueError("pyapriltags refine_edges must be 0 or 1")
    return settings


def load_compare_schemes(path=None, config=None):
    import re
    schemes = json.loads(Path(path or DEFAULT_COMPARE_PATH).read_text(encoding="utf-8-sig"))
    if not isinstance(schemes, list) or not schemes:
        raise ValueError("Compare schemes must be a nonempty JSON list")
    names = set()
    for scheme in schemes:
        if not isinstance(scheme, dict) or set(scheme) != {"name", "backend", "parameters"}:
            raise ValueError("Each scheme requires name, backend and parameters")
        name = scheme["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("Scheme names must be unique safe directory names")
        names.add(name)
        detector_settings(config or {}, scheme["backend"], scheme["parameters"])
    return schemes


def load_config(path: str | Path | None = None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        path = DEFAULT_PATH
    if path is not None:
        path = Path(path).resolve()
        override = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(override, dict):
            raise ValueError("Configuration must be a JSON object")
        board_file = override.pop("board_file", None)
        override.pop("camera_dir", None)
        if board_file is not None:
            if "board" in override:
                raise ValueError("Use board_file or inline board, not both")
            override["board"] = json.loads((path.parent / board_file).read_text(encoding="utf-8-sig"))
        for key, value in override.items():
            if key == "detector":
                config[key] = detector_settings({"detector": value})
                continue
            if key not in config or not isinstance(value, dict):
                raise ValueError(f"Unknown configuration section: {key}")
            if set(value) - set(config[key]):
                raise ValueError(f"Unknown configuration keys in {key}: {set(value) - set(config[key])}")
            config[key].update(value)
    validate_config(config)
    return config


def camera_profile(serial: str, config_path=None, explicit_path=None) -> dict:
    app_path = Path(config_path or DEFAULT_PATH).resolve()
    app = json.loads(app_path.read_text(encoding="utf-8-sig"))
    directory = app_path.parent / app.get("camera_dir", "cameras")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", serial):
        raise ValueError("Unsupported camera serial for profile lookup")
    path = Path(explicit_path).resolve() if explicit_path else directory / f"{serial}.json"
    if not path.exists() and not explicit_path:
        print(f"No profile for {serial}; using default settings")
        return {}
    profile = json.loads(path.read_text(encoding="utf-8-sig"))
    if profile.pop("serial", None) != serial:
        raise ValueError(f"Camera profile serial does not match selected device: {path}")
    if set(profile) - set(DEFAULT_CONFIG["camera"]):
        raise ValueError(f"Unknown camera profile fields: {path}")
    return profile


def validate_config(config: dict) -> None:
    if set(config) - set(DEFAULT_CONFIG) or set(DEFAULT_CONFIG) - set(config) - {"detector"}:
        raise ValueError("Missing/unknown configuration sections")
    for section, defaults in DEFAULT_CONFIG.items():
        if section == "detector":
            detector_settings(config)
            continue
        keys = set(config[section]) if isinstance(config[section], dict) else set()
        # Pre-existing session snapshots omit gain_auto and mean manual gain.
        allowed_missing = {"gain_auto", "resolution", "lens_note"} if section == "camera" else set()
        if not isinstance(config[section], dict) or keys - set(defaults) or set(defaults) - keys - allowed_missing:
            raise ValueError(f"Missing/unknown configuration fields: {section}")
    gain_auto = config["camera"].get("gain_auto", "off")
    if gain_auto not in ("off", "continuous"):
        raise ValueError("camera.gain_auto must be off or continuous")
    if gain_auto == "continuous" and config["camera"]["gain_db"] is not None:
        raise ValueError("Continuous automatic gain cannot be combined with a fixed gain_db")
    if config["camera"].get("resolution", "current") not in ("max", "current"):
        raise ValueError("camera.resolution must be max or current")
    if not isinstance(config["camera"].get("lens_note", ""), str):
        raise ValueError("camera.lens_note must be a string")
    from .board import Board
    Board.from_config(config)
    for section in ("camera", "capture", "quality", "calibration"):
        for name, value in config[section].items():
            if section == "camera" and name in ("gain_auto", "resolution", "lens_note"):
                continue
            if section == "camera" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{section}.{name} must be finite numeric")
            if value < 0 or (value == 0 and name != "gain_db"):
                raise ValueError(f"{section}.{name} must be positive (gain may be zero)")
    if config["capture"]["preview_width"] < 320 or type(config["capture"]["preview_width"]) is not int:
        raise ValueError("preview_width must be an integer >= 320")
    if config["calibration"]["min_views"] < 15 or type(config["calibration"]["min_views"]) is not int:
        raise ValueError("At least 15 calibration views are required (integer min_views)")
    if config["quality"]["saturation_fraction_max"] > 1:
        raise ValueError("saturation_fraction_max must be in (0, 1]")


def resolve_camera_config(config, serial, config_path=None, profile_path=None, overrides=None):
    result = copy.deepcopy(config)
    result["camera"].update(camera_profile(serial, config_path, profile_path))
    result["camera"].update(overrides or {})
    validate_config(result)
    return result


