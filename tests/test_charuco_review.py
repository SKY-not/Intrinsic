"""ChArUco correspondences, shared quality rules and reviewed capture, without hardware."""
import contextlib
import copy
import csv
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from intrinsic.board import Board, Detector, Observation
from intrinsic.capture import CaptureReview, capture, preview
from intrinsic.cli import main, configured_detection, parser
from intrinsic.config import load_config, validate_config, detector_settings
from intrinsic.quality import assess, quality_config
from intrinsic.report import export_report, read_calibration, anomaly_summary
from intrinsic.session import Session, read_json, write_json
from intrinsic.solver import calibrate, validate


ROOT = Path(__file__).resolve().parents[1]


def charuco_config():
    config = load_config()
    config["board"] = read_json(ROOT / "configs/boards/charuco.json")
    config["detector"]["backend"] = "auto"
    return config


class CharucoTests(unittest.TestCase):
    def test_switch_only_board_file_and_resolve_actual_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pipeline.json"
            pipeline = read_json(ROOT / "configs/pipeline.json")
            for name, backend in (("apriltag_20mm", "pyapriltags"), ("charuco", "opencv_charuco")):
                pipeline["board_file"] = str(ROOT / f"configs/boards/{name}.json")
                write_json(path, pipeline)
                config = load_config(path)
                self.assertEqual(config["detector"]["backend"], backend)
                self.assertEqual(config["quality"], {"min_tag_edge_px": 30, "min_sharpness": 80})
                session = Session.create(temporary, config, {"serial": name}, (960, 720))
                self.assertEqual(session.config["detector"]["backend"], backend)
            charuco = charuco_config()
            with self.assertRaises(ValueError):
                detector_settings(charuco, "pyapriltags")
            with self.assertRaises(ValueError):
                configured_detection(charuco, parser().parse_args(["capture", "--corner-refinement", "SUBPIX"]))

    def test_full_partial_rotated_detection_and_raw_pixel_origin(self):
        config = charuco_config()
        detector = Detector(config)
        board = detector.board
        image = cv2.copyMakeBorder(board.render(1200), 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        observation = detector.detect(image)
        self.assertTrue(observation.valid, observation.issues)
        self.assertEqual(observation.ids, list(range(88)))
        expected = board.points[:, :2] / .03 * 100 + 30
        np.testing.assert_allclose(observation.corners, expected, atol=.15)
        for cropped in (image[180:700, 300:950], cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)):
            obs = detector.detect(cropped)
            self.assertTrue(obs.valid, obs.issues)
            if cropped.shape[0] == 520:
                self.assertLess(len(obs.ids), 88)
            self.assertEqual(len(obs.corners), len(obs.ids))
            self.assertTrue(obs.metrics["tags"])
            objects, pixels = board.correspondences(obs)
            np.testing.assert_array_equal(objects, board.points[obs.ids])
            self.assertEqual(pixels.shape, objects[:, :2].shape)
            self.assertEqual(preview(cropped, obs, [], 700, kind="charuco").ndim, 3)

    def test_corner_id_validation_and_apriltag_completeness(self):
        board = Board.from_config(charuco_config())
        for ids in ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 11, 12, 12, 23], [0, 1, 11, 12, 22, 88]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                board.correspondences(Observation(ids, np.ones((len(ids), 2)), True))
        ids = [23, 0, 12, 1, 22, 11]
        objects, _ = board.correspondences(Observation(ids, np.ones((6, 2)), True))
        np.testing.assert_array_equal(objects, board.points[ids])
        april = Detector(load_config())
        obs = april.detect(Board().render())
        missing = april.assemble(Board().render(), obs.corners.reshape(4, 4, 2)[:3], [1, 2, 3])
        self.assertFalse(missing.valid)

    def test_invalid_charuco_geometry(self):
        for field, value in (("squares", [12, 2]), ("squares", [True, 9]),
                             ("dictionary", "DICT_4X4_50"), ("dictionary", "bad"),
                             ("square_length_m", float("nan")), ("marker_length_m", .031)):
            config = charuco_config()
            config["board"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_config(config)

    def test_variable_subsets_calibration_validation_and_report_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = charuco_config()
            board = Board.from_config(config)
            size = (960, 720)
            K = np.array([[900., 0, 478.], [0, 910., 362.], [0, 0, 1.]])
            D = np.array([-.12, .05, .001, -.002, .01])
            session = Session.create(temporary, config, {"serial": "charuco-synthetic"}, size)
            image = np.full(size[::-1], 128, np.uint8)
            rng = np.random.default_rng(719)
            for i in range(25):
                rv = rng.uniform([-.65, -.65, -.4], [.65, .65, .4])
                center = board.points.mean(axis=0)
                tv = rng.uniform([-.08, -.06, .6], [.08, .06, .95]) - cv2.Rodrigues(rv)[0] @ center
                pixels = cv2.projectPoints(board.points, rv, tv, K, D)[0].reshape(-1, 2)
                visible = np.flatnonzero(np.all((pixels > [1, 1]) & (pixels < np.asarray(size) - 1), axis=1))
                ids = sorted(rng.choice(visible, min(18 + i * 3 % 60, len(visible)), replace=False).tolist())
                points = pixels[ids] + rng.normal(0, .02, (len(ids), 2))
                session.save(image, Observation(ids, points.astype(np.float32), True), "calibration" if i < 20 else "validation")
            session = Session(session.path)
            result = calibrate(session)
            np.testing.assert_allclose(np.asarray(result["K"])[[0, 1], [0, 1]], [900, 910], atol=.8)
            result["validation"] = validate(session, result)
            result["validation_status"] = "evaluated"
            self.assertLess(result["validation"]["rms_px"], .06)
            self.assertAlmostEqual(result["validation"]["rms_px"], result["validation"]["metrics"]["rmse_px"])
            rows = {r["id"]: r for r in session.records}
            for view in result["views"] + result["validation"]["views"]:
                self.assertEqual(view["point_ids"], rows[view["id"]]["observation"]["ids"])
            output = session.new_run("calibrate")
            export_report(session, result, output)
            saved = read_calibration(output / "calibration.json")
            self.assertEqual(saved["views"][0]["point_ids"], result["views"][0]["point_ids"])
            with (output / "residuals.csv").open(encoding="utf-8-sig", newline="") as handle:
                exported = list(csv.DictReader(handle))
            first = result["views"][0]
            self.assertEqual([int(r["corner"]) for r in exported if r["id"] == first["id"]], first["point_ids"])
            _, worst = anomaly_summary(result["views"])
            self.assertIn(worst["corner"], next(v for v in result["views"] if v["id"] == worst["id"])["point_ids"])
            self.assertTrue((output / "report.html").is_file())

    def test_import_and_reference_image_use_board_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = root / "pipeline.json"
            write_json(cfg, {"board_file": str(ROOT / "configs/boards/charuco.json"), "detector": {"backend": "auto"}})
            images = root / "input"
            images.mkdir()
            self.assertEqual(main(["board-image", "--config", str(cfg), "--output", str(images / "board.png")]), 0)
            self.assertEqual(main(["import-images", "--config", str(cfg), "--images", str(images),
                                   "--camera-id", "charuco", "--output", str(root / "data")]), 0)
            session = Session(next((root / "data/charuco").iterdir()))
            self.assertEqual(len(session.records), 1)
            self.assertEqual(session.manifest["detectors"][0]["backend"], "opencv_charuco")
            output = root / "comparison"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["compare-detectors", "--session", str(session.path), "--output", str(output)]), 1)
            self.assertFalse(output.exists())

    def test_legacy_configuration_and_warning_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config()
            config["board"].pop("type")
            config.pop("detector")
            config["quality"] = {"blur_variance_min": 40, "tag_edge_min_px": 30, "saturation_fraction_max": .4}
            before = copy.deepcopy(config)
            session = Session.create(temporary, config, {"serial": "legacy"}, (1400, 1400))
            obs = Detector(config).detect(Board().render())
            obs.warnings = ["historical warning"]
            obs.metrics = {"blur_variance": 42}
            session.save(Board().render(), obs, "calibration")
            loaded = Session(session.path)
            self.assertEqual(loaded.config, before)
            self.assertEqual(loaded.records[0]["observation"], obs.to_dict())


class QualityReviewTests(unittest.TestCase):
    def test_sharpness_is_per_marker_and_ignores_the_gap_between_markers(self):
        image = np.full((140, 300), 128, np.uint8)
        image[20:100, 20:100] = ((np.indices((80, 80)).sum(axis=0) // 5 % 2) * 255).astype(np.uint8)
        left = np.array([[20, 20], [99, 20], [99, 99], [20, 99]])
        right = left + [180, 0]
        regions = [(1, left), (2, right)]
        quality = assess(image, regions, quality_config())
        self.assertEqual([w["tag_id"] for w in quality["warnings"]], [2])
        changed = image.copy()
        changed[:, 110:190] = np.random.default_rng(5).integers(0, 256, (140, 80), dtype=np.uint8)
        self.assertEqual(assess(changed, regions, quality_config()), quality)

    def test_per_marker_quality_matches_reference_formula_and_is_advisory(self):
        gray = np.full((100, 100), 128, np.uint8)
        corners = np.array([[20., 20.], [40., 20.], [40., 40.], [20., 40.]])
        quality = assess(gray, [(7, corners)], quality_config())
        self.assertEqual({w["code"] for w in quality["warnings"]}, {"tag_too_small", "possibly_blurred"})
        self.assertEqual(quality["metrics"], [{"tag_id": 7, "min_edge_px": 20., "sharpness": 0.}])
        image = Board().render()
        config = load_config()
        config["quality"] = {"min_tag_edge_px": 1000, "min_sharpness": 1e12}
        obs = Detector(config).detect(image)
        self.assertTrue(obs.valid)
        self.assertEqual(len(obs.warnings), 8)
        self.assertEqual(Detector(config).detect(image, check_quality=False).warnings, [])
        with tempfile.TemporaryDirectory() as temporary:
            session = Session.create(temporary, config, {"serial": "warnings"}, image.shape[1::-1])
            row = session.save(image, obs, "calibration")
            self.assertEqual(row["observation"]["warnings"], obs.warnings)

    def test_frozen_candidate_retry_and_exclusion_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config()
            image = Board().render()
            obs = Detector(config).detect(image)
            frame = types.SimpleNamespace(image=image.copy(), number=1)
            session = Session.create(temporary, config, {"serial": "review"}, image.shape[1::-1])
            review = CaptureReview(session)
            review.stage(frame, obs, "validation", {"frame": 1})
            original = frame.image.copy()
            frame.image[:] = 0
            obs.corners[:] = 0
            with self.assertRaises(ValueError):
                review.stage(frame, obs, "calibration", {})
            self.assertEqual(session.records, [])
            with patch.object(session, "save", side_effect=OSError("disk full")), self.assertRaises(OSError):
                review.resolve(True, "user_confirmed")
            self.assertIsNotNone(review.pending)
            row = review.resolve(True, "user_confirmed")
            np.testing.assert_array_equal(session.checked_image(row), original)
            self.assertEqual(row["role"], "validation")
            review.stage(types.SimpleNamespace(image=image, number=2), Detector(config).detect(image), "calibration", {})
            folder = review.resolve(False, "user_deleted")
            self.assertEqual(read_json(folder / "frame.json")["reason"], "user_deleted")
            self.assertEqual(len(Session(session.path).records), 1)

    def run_capture_keys(self, keys):
        config = load_config()
        image = Board().render(700)
        camera = types.SimpleNamespace(metadata={"serial": "capture"})
        sequence = iter(range(1, 30))
        def latest():
            number = next(sequence)
            return types.SimpleNamespace(image=image.copy(), number=number, metadata={"captured": number})
        @contextlib.contextmanager
        def configured(*args, **kwargs):
            yield camera, config
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            stack.enter_context(patch("intrinsic.capture.configured_camera", configured))
            stream = stack.enter_context(patch("intrinsic.capture.LatestFrame"))
            stream.return_value.__enter__.return_value.latest.side_effect = latest
            stack.enter_context(patch("intrinsic.capture.time.monotonic", side_effect=range(100, 140)))
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch("intrinsic.capture.cv2." + name))
            stack.enter_context(patch("intrinsic.capture.cv2.getWindowProperty", return_value=1))
            stack.enter_context(patch("intrinsic.capture.cv2.waitKeyEx", side_effect=keys))
            session = capture(config, temporary)
            rows = Session(session.path).records
            excluded = [read_json(p) for p in (session.path / "exclude").glob("*/frame.json")]
            return rows, excluded

    def test_windows_delete_and_enter_keep_the_frozen_frame_and_role(self):
        rows, excluded = self.run_capture_keys([32, ord("v"), 13, ord("v"), 3014656, ord("q")])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metadata"]["captured"], 1)
        self.assertEqual(rows[0]["role"], "calibration")
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["metadata"]["captured"], 4)
        self.assertEqual(excluded[0]["role"], "validation")
        self.assertEqual(excluded[0]["reason"], "user_deleted")

    def test_exit_excludes_unconfirmed_candidate(self):
        rows, excluded = self.run_capture_keys([ord("v"), 27])
        self.assertEqual(rows, [])
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["reason"], "exit_without_confirmation")


if __name__ == "__main__":
    unittest.main()
