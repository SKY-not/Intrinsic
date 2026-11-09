"""Small detector regressions; real-data comparisons require explicit --session."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from intrinsic.board import Board, Detector, Observation
from intrinsic.cli import configured_detection, compare_detectors, main, parser
from intrinsic.config import load_config, validate_config, detector_settings, load_compare_schemes
from intrinsic.session import Session, read_json, write_json, write_image
from intrinsic.report import read_detector_comparison, read_calibration


class DetectorComparisonTests(unittest.TestCase):
    def test_configuration_priority_and_parameter_validation(self):
        config = load_config()
        original = copy.deepcopy(config)
        args = parser().parse_args(["capture"])
        self.assertIsNone(args.detector)
        self.assertEqual(configured_detection(config, args)["detector"]["backend"], config["detector"]["backend"])
        args = parser().parse_args(["capture", "--detector", "opencv", "--corner-refinement", "contour", "--corner-window-size", "7"])
        resolved = configured_detection(config, args)
        self.assertEqual(resolved["detector"]["backend"], "opencv")
        self.assertEqual(resolved["detector"]["opencv"], {"corner_refinement": "CONTOUR", "window_size": 7})
        self.assertEqual(config, original)
        for backend, parameters in (("opencv", {"window_size": 0}), ("opencv", {"corner_refinement": "bad"}),
                                    ("pyapriltags", {"quad_decimate": .5}), ("pyapriltags", {"refine_edges": True})):
            with self.subTest(backend=backend, parameters=parameters), self.assertRaises(ValueError):
                detector_settings(config, backend, parameters)
        invalid = parser().parse_args(["capture", "--detector", "pyapriltags", "--corner-refinement", "NONE"])
        with self.assertRaises(ValueError):
            configured_detection(config, invalid)

    def test_legacy_snapshot_and_default_scheme_file(self):
        config = load_config()
        config.pop("detector")
        before = copy.deepcopy(config)
        validate_config(config)
        self.assertEqual(detector_settings(config)["backend"], "opencv")
        self.assertEqual(config, before)
        schemes = load_compare_schemes()
        self.assertEqual({s["backend"] for s in schemes}, {"opencv", "pyapriltags"})
        self.assertEqual({s["parameters"]["corner_refinement"] for s in schemes if s["backend"] == "opencv"},
                         {"NONE", "SUBPIX", "CONTOUR", "APRILTAG"})

    def test_corner_order_pixel_origin_and_all_methods(self):
        config = load_config()
        config["board"]["rotations_deg"] = [0, 90, 180, 270]
        board = Board.from_config(config)
        for scheme in load_compare_schemes():
            with self.subTest(scheme=scheme["name"]):
                detector = Detector(config, scheme["backend"], scheme["parameters"])
                image = board.render()
                expected = (board.points[:, :2] / board.size + .5) * 1400 - .5
                method = scheme["parameters"].get("corner_refinement")
                if scheme["backend"] == "opencv":
                    self.assertEqual(detector.detector.getDetectorParameters().cornerRefinementMethod,
                                     getattr(cv2.aruco, "CORNER_REFINE_" + method))
                tolerance = .7 if method in ("NONE", "CONTOUR") else .15
                for _ in range(4):
                    observation = detector.detect(image)
                    self.assertTrue(observation.valid, observation.issues)
                    self.assertEqual(observation.ids, config["board"]["ids"])
                    np.testing.assert_allclose(observation.corners, expected, atol=tolerance)
                    self.assertEqual(set(observation.to_dict()), {"ids", "corners", "valid", "issues", "warnings", "metrics"})
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    expected = np.column_stack((1399 - expected[:, 1], expected[:, 0]))

    def test_raw_gray_no_pose_and_conversion_before_quality_checks(self):
        config = load_config()
        detector = Detector(config, "pyapriltags")
        image = cv2.cvtColor(Board().render(), cv2.COLOR_GRAY2BGR)
        with patch.object(detector.detector, "detect", return_value=[]) as call:
            self.assertFalse(detector.detect(image).valid)
        np.testing.assert_array_equal(call.call_args.args[0], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.assertEqual(call.call_args.kwargs, {"estimate_tag_pose": False})
        detector = Detector(config, "opencv", {"corner_refinement": "APRILTAG"})
        expected = (Board().points[:, :2] / Board().size + .5) * 1400 - .5
        raw = (expected + .5).reshape(4, 1, 4, 2)
        detector.detector = types.SimpleNamespace(detectMarkers=lambda gray: (raw, np.array([[1], [2], [3], [4]]), []),
            getDetectorParameters=lambda: types.SimpleNamespace(cornerRefinementMethod=cv2.aruco.CORNER_REFINE_APRILTAG))
        with patch.object(detector, "assemble", wraps=detector.assemble) as assemble:
            detector.detect(image)
        np.testing.assert_allclose(np.asarray(assemble.call_args.args[1]).reshape(16, 2), expected, atol=1e-5)

    @staticmethod
    def source_fixture(root, legacy=True):
        config = load_config()
        if legacy:
            config.pop("detector")
        else:
            config["detector"]["opencv"]["window_size"] = 19
        board = Board()
        points = ((board.points[:, :2] / board.size + .5) * 700 - .5).astype(np.float32)
        observation = Observation([1, 2, 3, 4], points, True)
        source = Session.create(root / "original", config, {"serial": "fixture"}, (700, 700))
        for i, role in enumerate(("calibration", "calibration", "validation")):
            image = np.full((700, 700), 180, np.uint8)
            image[0, 0] = i
            source.save(image, observation, role)
        return source, observation

    def test_readonly_view_reads_source_and_accepts_legacy_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, observation = self.source_fixture(root)
            manifest = read_json(source.path / "session.json")
            manifest["config_hash"] = "ignored legacy value"
            write_json(source.path / "session.json", manifest)
            row_path = source.path / "frames" / (source.records[0]["id"] + ".json")
            row = read_json(row_path)
            row.update(image_hash="ignored legacy value", config_hash="ignored legacy value")
            write_json(row_path, row)
            source = Session(source.path)
            before = (source.path / "session.json").read_bytes()
            detection = root / "method" / "detections.json"
            detection.parent.mkdir()
            write_json(detection, {"detector": {"backend": "opencv"}, "frames": [
                {"id": r["id"], "observation": observation.to_dict()}
                for r in source.records]})
            view = Session.view(source, read_json(detection), [r["id"] for r in source.records])
            self.assertTrue(view.readonly)
            self.assertFalse((root / "method/session").exists())
            self.assertNotIn("detector", view.config)
            self.assertEqual((source.path / "session.json").read_bytes(), before)
            reloaded = Session.view(Session(source.path), read_json(detection), [r["id"] for r in source.records])
            np.testing.assert_array_equal(reloaded.checked_image(reloaded.records[0]), source.checked_image(source.records[0]))
            with self.assertRaises(ValueError):
                view.save(source.checked_image(source.records[0]), observation, "calibration")
            with self.assertRaises(ValueError):
                view.new_run("calibrate")
            image = source.checked_image(source.records[0]); image[1, 1] = 0
            write_image(source.path / source.records[0]["image"], image)
            self.assertEqual(reloaded.checked_image(reloaded.records[0])[1, 1], 0)
            write_image(source.path / source.records[0]["image"], image[:200])
            with self.assertRaisesRegex(ValueError, "dimensions"):
                reloaded.checked_image(reloaded.records[0])

    def test_compare_shared_and_fixed_sets_no_copies_complete_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, observation = self.source_fixture(root, legacy=False)
            original = {p: p.read_bytes() for p in source.path.rglob("*") if p.is_file()}
            schemes = root / "schemes.json"
            write_json(schemes, [{"name": "cv_candidate", "backend": "opencv", "parameters": {"corner_refinement": "NONE"}},
                                 {"name": "tag_candidate", "backend": "pyapriltags", "parameters": {}}])
            def detector(config, backend, parameters):
                if backend == "opencv":
                    self.assertEqual(parameters["window_size"], 5)
                def detect(image):
                    result = copy.deepcopy(observation)
                    if backend == "pyapriltags" and image[0, 0] == 1:
                        result.valid = False; result.issues = ["missing tags: [1]"]
                    return result
                return types.SimpleNamespace(detect=detect, description={"backend": backend, "version": "fixture", "family": "tag36h11"})
            for fixed in (False, True):
                captured = []
                def fit(view):
                    captured.append([(r["id"], r["role"]) for r in view.records])
                    raise ValueError("fixture stops before calibration")
                output = root / ("fixed" if fixed else "shared")
                args = types.SimpleNamespace(session=[str(source.path)], output=str(output), schemes=str(schemes),
                    frame_ids=[r["id"] for r in source.records] if fixed else None)
                with patch("intrinsic.cli.Detector", side_effect=detector), patch("intrinsic.cli.calibrate", side_effect=fit):
                    with self.assertRaises(RuntimeError):
                        compare_detectors(args)
                expected = sorted((r["id"], r["role"]) for r in source.records if fixed or source.checked_image(r)[0, 0] != 1)
                self.assertEqual(captured, [expected] if fixed else [expected, expected])
                self.assertFalse(list(output.glob("methods/*/session")))
                manifest = read_json(output / "input_manifest.json")["datasets"][0]
                self.assertEqual((output / manifest["source_session"]).resolve(), source.path)
                self.assertEqual(manifest["selection_policy"], "fixed" if fixed else "common_success")
                for key in ("board", "quality", "solver", "versions", "methods", "input_frames"):
                    self.assertIn(key, manifest)
                html = (output / "comparison.html").read_text(encoding="utf-8")
                for content in ("cv_candidate", "tag_candidate", 'href="input_manifest.json"', source.records[0]["id"], "未完成标定"):
                    self.assertIn(content, html)
            self.assertEqual({p: p.read_bytes() for p in original}, original)

    def test_successful_compare_has_compact_records_and_self_contained_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, observation = self.source_fixture(root)
            schemes = root / "schemes.json"
            write_json(schemes, [{"name": "cv", "backend": "opencv", "parameters": {}}])
            metrics = dict(point_count=16, mre_px=.25, rmse_px=.25, max_error_px=.25, std_px=0., variance_px2=0.)
            def fit(view):
                def record(row):
                    return {"id": row["id"], "image": row["image"], "role": row["role"],
                        "points": observation.corners.tolist(), "residuals": [[.25, 0.]] * 16,
                        "metrics": metrics, "rms_px": .25, "max_error_px": .25,
                        "rvec": [0., 0., 0.], "tvec_m": [0., 0., 1.]}
                return dict(schema_version=1, model="opencv_pinhole_brown5", created_utc="2026-11-03T00:00:00+00:00",
                    source_session=str(view.path), config=view.config, camera=view.manifest["camera"], image_size=[700, 700],
                    K=[[500., 0., 350.], [0., 500., 350.], [0., 0., 1.]], D=[0.] * 5,
                    std_order=["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"], std_intrinsics=[.1] * 9,
                    diagnostics={"fixture": 1}, warnings=["fixture diagnostic"], review_frames=[], metrics=metrics,
                    views=[record(r) for r in view.records if r["role"] == "calibration"],
                    validation={"views": [record(r) for r in view.records if r["role"] == "validation"],
                                "metrics": metrics, "warnings": []}, validation_status="evaluated", validation_reason="fixture")
            args = types.SimpleNamespace(session=[str(source.path)], output=str(root / "comparison"), schemes=str(schemes))
            detector = types.SimpleNamespace(detect=lambda image: copy.deepcopy(observation),
                                            description={"backend": "opencv", "version": "fixture"})
            with patch("intrinsic.cli.Detector", return_value=detector), patch("intrinsic.cli.calibrate", side_effect=fit), \
                    patch("intrinsic.cli.apply_validation", return_value=True):
                compare_detectors(args)
            output = Path(args.output)
            self.assertEqual({p.name for p in output.iterdir()}, {"input_manifest.json", "comparison.json", "comparison.html",
                "per_view_comparison.csv", "corner_differences.csv", "corner_previews", "methods"})
            self.assertEqual({p.name for p in (output / "methods/cv").iterdir()}, {"detections.json", "calibration.json"})
            saved = read_json(output / "methods/cv/calibration.json")
            self.assertNotIn("config", saved)
            self.assertNotIn("points", saved["views"][0])
            self.assertNotIn("image", saved["views"][0])
            restored = read_calibration(output / "methods/cv/calibration.json")
            self.assertEqual(restored["views"][0]["points"], observation.corners.tolist())
            self.assertEqual(restored["K"], saved["K"])
            self.assertEqual(read_detector_comparison(output)["datasets"][0]["results"]["cv"]["training_metrics"], metrics)
            html = (output / "comparison.html").read_text(encoding="utf-8")
            self.assertIn("已独立验证", html)
            self.assertNotIn("results/previews", html)
            self.assertNotIn("results/report", html)


if __name__ == "__main__":
    if "--session" in sys.argv:
        raise SystemExit(main(["compare-detectors", *sys.argv[1:]]))
    unittest.main()
