from __future__ import annotations

import copy
import ctypes as ct
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from intrinsic.board import Board, Detector, Observation, load_config, validate_config
from intrinsic.camera import Frame, HikCamera, LatestFrame, Runtime
from intrinsic.capture import similar_view
from intrinsic.cli import main, parser, compare_detectors
from intrinsic.report import export_report
from intrinsic.session import Session, read_json, write_image, write_json
from intrinsic.solver import calibrate, model_diagnostics, validate, view_constraints


SIZE = (960, 720)
K = np.array([[900., 0, 478.], [0, 910., 362.], [0, 0, 1.]])
D = np.array([-.12, .05, .001, -.002, .01])


def synthetic_views(count=30, distortion=D, noise=0., frontal=False):
    rng = np.random.default_rng(714)
    objects = Board().points
    samples = []
    while len(samples) < count:
        rv = np.zeros(3) if frontal else rng.uniform([-.7, -.7, -.5], [.7, .7, .5])
        tv = np.array([rng.uniform(-.075, .075), rng.uniform(-.055, .055), rng.uniform(.19, .34)])
        pixels = cv2.projectPoints(objects, rv, tv, K, distortion)[0].reshape(-1, 2)
        if np.any(pixels < [8, 8]) or np.any(pixels >= np.array(SIZE)-8):
            continue
        pixels += rng.normal(0, noise, pixels.shape)
        samples.append((pixels.astype(np.float32), rv, tv))
    return samples


def make_session(root, count=25, noise=.02, outlier=False):
    session = Session.create(root, load_config(), {"serial": "synthetic", "source": "test"}, SIZE)
    for i, (pixels, rv, tv) in enumerate(synthetic_views(count+5, noise=noise)):
        if outlier and i == 0:
            pixels = pixels.copy()
            pixels[0] += [12, -9]
        image = np.full(SIZE[::-1], 180, np.uint8)
        cv2.putText(image, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 30, 2)
        for p in pixels:
            cv2.circle(image, tuple(p.astype(int)), 3, 20, -1)
        session.save(image, Observation([1, 2, 3, 4], pixels, True),
                     "calibration" if i < count else "validation", {"true_rvec": rv.tolist(), "true_tvec": tv.tolist()})
    return session


class GeometryTests(unittest.TestCase):
    def test_board_dimensions_and_order(self):
        board = Board()
        blocks = board.points.reshape(4, 4, 3)
        np.testing.assert_allclose(blocks.mean(axis=1)[:, :2], [[-.015, -.015], [.015, -.015], [-.015, .015], [.015, .015]], atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(blocks[:, 1]-blocks[:, 0], axis=1), .02)
        np.testing.assert_allclose(blocks[0, 0], [-.005, -.005, 0])

    def test_default_matches_supplied_board_orientation(self):
        board = Board()
        obs = Detector(load_config()).detect(board.render())
        # On the supplied board the first decoded corner is physically bottom-right.
        np.testing.assert_allclose(obs.corners[:4], [[599.5, 599.5], [199.5, 599.5],
                                                   [199.5, 199.5], [599.5, 199.5]], atol=1)

    def test_render_detection_and_rotations(self):
        config = load_config()
        config["board"]["rotations_deg"] = [0, 90, 180, 270]
        board = Board.from_config(config)
        detector = Detector(config)
        expected = (board.points[:, :2]/board.size + .5)*1400 - .5
        image = board.render()
        for _ in range(4):
            obs = detector.detect(image)
            self.assertTrue(obs.valid, obs.issues)
            self.assertEqual(obs.ids, [1, 2, 3, 4])
            np.testing.assert_allclose(obs.corners, expected, atol=1)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            expected = np.column_stack((1399-expected[:, 1], expected[:, 0]))

    def test_missing_and_duplicate_ids_rejected(self):
        detector = Detector(load_config())
        obs = detector.detect(Board().render())
        corners = obs.corners.reshape(4, 4, 2)
        bad = detector.assemble(Board().render(), corners, [1, 1, 3, 4])
        self.assertFalse(bad.valid)
        self.assertTrue(any("duplicate" in x for x in bad.issues))
        self.assertFalse(detector.assemble(Board().render(), corners[:3], [1, 2, 3]).valid)

    def test_invalid_board_geometry(self):
        for kwargs in ({"tag_size": -.02}, {"ids": (1, 1, 3, 4)}, {"rotations": (0, 45, 0, 0)}, {"size": .05}):
            with self.assertRaises(ValueError):
                Board(**kwargs)

    def test_config_defaults_and_invalid_numeric_settings(self):
        file_config = load_config(Path(__file__).resolve().parents[1]/"configs/pipeline.json")
        self.assertEqual(load_config()["board"], file_config["board"])
        for section, key, value in (("capture", "detection_hz", 0), ("camera", "exposure_us", float("nan")),
                                    ("quality", "saturation_fraction_max", 2), ("calibration", "min_views", 14)):
            config = load_config()
            config[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_similar_view(self):
        p = synthetic_views(1)[0][0]
        self.assertTrue(similar_view(p+1, [p], SIZE))
        self.assertFalse(similar_view(p+100, [p], SIZE))


class DetectorBackendTests(unittest.TestCase):
    def test_pyapriltags_fixed_mapping_and_pixel_origin(self):
        config = load_config()
        config["board"]["rotations_deg"] = [0, 90, 180, 270]
        board = Board.from_config(config)
        detector = Detector(config, backend="pyapriltags")
        expected = (board.points[:, :2] / board.size + .5) * 1400 - .5
        image = board.render()
        for _ in range(4):
            observation = detector.detect(image)
            self.assertTrue(observation.valid, observation.issues)
            self.assertEqual(observation.ids, [1, 2, 3, 4])
            # Tight enough to detect an omitted half-pixel conversion.
            np.testing.assert_allclose(observation.corners, expected, atol=.15)
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            expected = np.column_stack((1399 - expected[:, 1], expected[:, 0]))

    def test_raw_grayscale_without_tag_pose(self):
        config = load_config()
        image = cv2.cvtColor(Board().render(), cv2.COLOR_GRAY2BGR)
        image[:, :, 0] = 100
        detector = Detector(config, backend="pyapriltags")
        with patch.object(detector.detector, "detect", return_value=[]) as detect:
            observation = detector.detect(image)
        args, kwargs = detect.call_args
        np.testing.assert_array_equal(args[0], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.assertEqual(kwargs, {"estimate_tag_pose": False})
        self.assertFalse(observation.valid)

    def test_opencv_default_and_cli_selection(self):
        config = load_config()
        config.pop("detector", None)
        image = Board().render()
        np.testing.assert_array_equal(Detector(config).detect(image).corners,
                                      Detector(config, backend="opencv").detect(image).corners)
        self.assertIsNone(parser().parse_args(["capture"]).detector)
        self.assertIsNone(parser().parse_args(["import-images", "--images", "x", "--camera-id", "x"]).detector)
        self.assertEqual(parser().parse_args(["capture", "--detector", "pyapriltags"]).detector, "pyapriltags")

    def test_pyapriltags_import_persists_compatible_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "input"
            images.mkdir()
            write_image(images / "board.png", Board().render())
            root = Path(tmp) / "sessions"
            self.assertEqual(main(["import-images", "--images", str(images), "--camera-id", "test",
                                   "--detector", "pyapriltags", "--output", str(root)]), 0)
            session = Session(next((root / "test").iterdir()))
            self.assertEqual(len(session.records), 1)
            row = session.records[0]
            self.assertEqual(session.manifest["detectors"][row["detector_index"]]["backend"], "pyapriltags")
            self.assertEqual(row["observation"]["ids"], [1, 2, 3, 4])
            self.assertEqual(np.asarray(row["observation"]["corners"]).shape, (16, 2))
            session.checked_image(row)

    def test_compare_preserves_ids_roles_and_uses_common_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            image = Board().render()
            observation = Detector(config).detect(image)
            source = Session.create(Path(tmp) / "original", config, {"serial": "test"}, image.shape[1::-1])
            rows = []
            for i, role in enumerate(("calibration", "calibration", "validation")):
                frame = image.copy()
                frame[0, 0] = i
                rows.append(source.save(frame, observation, role))
            before = {p: p.read_bytes() for p in source.path.rglob("*") if p.is_file()}

            def backend(config, backend, parameters=None):
                def detect(frame):
                    result = copy.deepcopy(observation)
                    if backend == "pyapriltags" and frame[0, 0] == 1:
                        result.valid = False
                        result.issues = ["missing tags: [1]"]
                    return result
                return types.SimpleNamespace(detect=detect, description={"backend": backend})

            fitted = []
            def fit(session):
                fitted.append([(r["id"], r["role"]) for r in session.records])
                for row in session.records:
                    session.checked_image(row)
                raise ValueError("test stops before optimization")

            schemes = Path(tmp) / "schemes.json"
            write_json(schemes, [{"name": "opencv", "backend": "opencv", "parameters": {}},
                                 {"name": "pyapriltags", "backend": "pyapriltags", "parameters": {}}])
            args = types.SimpleNamespace(session=[str(source.path)], output=str(Path(tmp) / "comparison"), schemes=str(schemes))
            with patch("intrinsic.cli.Detector", side_effect=backend), patch("intrinsic.cli.calibrate", side_effect=fit):
                with self.assertRaisesRegex(RuntimeError, "fits/validations failed"):
                    compare_detectors(args)
            expected = [(r["id"], r["role"]) for r in rows if r["id"] != rows[1]["id"]]
            self.assertEqual(fitted, [sorted(expected), sorted(expected)])
            from intrinsic.report import read_detector_comparison
            summary = read_detector_comparison(Path(args.output))["datasets"][0]
            self.assertEqual(summary["detection"]["opencv"]["success_count"], 3)
            self.assertEqual(summary["detection"]["pyapriltags"]["success_count"], 2)
            self.assertEqual({p: p.read_bytes() for p in before}, before)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.session = make_session(cls.tmp.name)
        cls.result = calibrate(cls.session)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_recovers_intrinsics_with_noise(self):
        estimated = np.array(self.result["K"])
        np.testing.assert_allclose(estimated[[0, 1], [0, 1]], K[[0, 1], [0, 1]], rtol=.002)
        np.testing.assert_allclose(estimated[:2, 2], K[:2, 2], atol=1)
        self.assertLess(self.result["rms_px"], .05)
        self.assertEqual(len(self.result["D"]), 5)

    def test_validation_not_in_training_and_fixed_parameters(self):
        original = copy.deepcopy(self.result)
        held_out = validate(self.session, self.result)
        self.assertEqual(original, self.result)
        self.assertEqual(len(held_out["views"]), 5)
        self.assertLess(held_out["rms_px"], .08)
        self.assertTrue(set(v["id"] for v in held_out["views"]).isdisjoint(v["id"] for v in self.result["views"]))

    def test_front_parallel_degeneracy(self):
        samples = synthetic_views(20, distortion=np.zeros(5), frontal=True)
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            view_constraints(Board().points, [s[0] for s in samples], SIZE)

    def test_bad_model_diagnostics(self):
        for coefficients in ([np.nan]*5, [-10., 0, 0, 0, 0]):
            with self.assertRaises(ValueError):
                model_diagnostics(K, np.array(coefficients), SIZE)
        wrong = K.copy()
        wrong[0, 0] = -1
        with self.assertRaises(ValueError):
            model_diagnostics(wrong, D, SIZE)

    def test_insufficient_views_and_unknown_exclusions(self):
        ids = [v["id"] for v in self.result["views"]]
        with self.assertRaisesRegex(ValueError, "at least"):
            calibrate(self.session, ids[:11])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            calibrate(self.session, ["nonexistent"])

    def test_other_camera_rejected(self):
        wrong = copy.deepcopy(self.result)
        wrong["camera"]["serial"] = "another_camera"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate(self.session, wrong)

    def test_outlier_review_and_explicit_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = make_session(tmp, outlier=True)
            result = calibrate(session)
            bad_id = session.records[0]["id"]
            self.assertIn(bad_id, result["review_frames"])
            improved = calibrate(session, [bad_id])
            self.assertLess(improved["rms_px"], result["rms_px"])
            self.assertEqual(improved["excluded_frames"], [bad_id])
            self.assertEqual(len(session.records), 30)

    def test_exports_and_reproducible_reload(self):
        result = copy.deepcopy(self.result)
        result["validation"] = validate(self.session, result)
        result["validation_status"] = "evaluated"
        output = self.session.new_run("test")
        export_report(self.session, result, output)
        for name in ("calibration.json", "intrinsics.json", "per_view.csv", "residuals.csv", "report.html"):
            self.assertTrue((output/name).is_file())
        self.assertFalse((output/"calibration.yml").exists())
        np.testing.assert_array_equal(read_json(output/"intrinsics.json")["K"], result["K"])
        repeat = calibrate(Session(self.session.path))
        np.testing.assert_allclose(repeat["K"], result["K"], atol=1e-6)
        self.assertEqual(read_json(output/"calibration.json")["validation_status"], "evaluated")


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = load_config()
        self.session = Session.create(self.tmp.name, self.config, {"serial": "camera-A"}, (1400, 1400))
        self.image = Board().render()
        self.obs = Detector(self.config).detect(self.image)

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_pixels_allowed_with_distinct_frame_ids(self):
        first = self.session.save(self.image, self.obs, "calibration")
        second = Session(self.session.path).save(self.image, self.obs, "validation")
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(len(Session(self.session.path).records), 2)

    def test_resolution_and_observation_validation(self):
        with self.assertRaisesRegex(ValueError, "resolution"):
            self.session.save(self.image[:500], self.obs, "calibration")
        wrong = copy.deepcopy(self.obs)
        wrong.ids = [4, 3, 2, 1]
        with self.assertRaises(ValueError):
            self.session.save(self.image, wrong, "calibration")

    def test_write_failure_not_committed(self):
        with patch("intrinsic.session.write_image", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.session.save(self.image, self.obs, "calibration")
        self.assertEqual(Session(self.session.path).records, [])
        with patch("intrinsic.session.write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.session.save(self.image, self.obs, "calibration")
        self.assertEqual(Session(self.session.path).records, [])

    def test_configuration_copy_and_no_integrity_audit(self):
        self.config["board"]["tag_size_m"] = .01
        self.assertEqual(self.session.config["board"]["tag_size_m"], .02)
        manifest = read_json(self.session.path/"session.json")
        manifest["config"]["board"]["tag_size_m"] = .021
        write_json(self.session.path/"session.json", manifest)
        self.assertEqual(Session(self.session.path).config["board"]["tag_size_m"], .021)

    def test_readable_changed_image_allowed(self):
        row = self.session.save(self.image, self.obs, "calibration")
        altered = self.image.copy()
        altered[0, 0] = 0
        write_image(self.session.path/row["image"], altered)
        np.testing.assert_array_equal(self.session.checked_image(row), altered)

    def test_camera_directories_are_isolated(self):
        other = Session.create(self.tmp.name, self.config, {"serial": "camera-B"}, (1400, 1400))
        self.assertNotEqual(other.path.parent, self.session.path.parent)


def fake_camera(pixel_type=1, conversion_error=0):
    """A fake SDK with real ctypes-owned buffers; no vendor DLL import."""
    sdk = types.SimpleNamespace(PixelType_Gvsp_Mono8=1, PixelType_Gvsp_BGR8_Packed=3,
                                MV_E_NODATA=0x80000007, MV_CC_PIXEL_CONVERT_PARAM_EX=types.SimpleNamespace)
    buffer = (ct.c_ubyte * 12)(*range(12))
    output = types.SimpleNamespace(pBufAddr=ct.cast(buffer, ct.POINTER(ct.c_ubyte)),
        stFrameInfo=types.SimpleNamespace(nWidth=2, nHeight=2, nFrameLen=12, enPixelType=pixel_type,
                                         nFrameNum=7, nDevTimeStampHigh=0, nDevTimeStampLow=123))
    sdk.MV_FRAME_OUT = lambda: output
    calls = []
    class Handle:
        def MV_CC_GetImageBuffer(self, *args):
            return 0
        def MV_CC_FreeImageBuffer(self, *args):
            calls.append("free")
            ct.memset(buffer, 0, len(buffer))
            return 0
        def MV_CC_ConvertPixelTypeEx(self, params):
            if conversion_error:
                return conversion_error
            ct.memmove(params.pDstBuffer, buffer, params.nDstBufferSize)
            params.nDstLen = params.nDstBufferSize
            return 0
        def MV_CC_StopGrabbing(self):
            calls.append("stop")
            return 1  # even a failed stop must not skip close/destroy
        def MV_CC_CloseDevice(self):
            calls.append("close")
            return 0
        def MV_CC_DestroyHandle(self):
            calls.append("destroy")
            return 0
    camera = HikCamera.__new__(HikCamera)
    camera.sdk, camera.cam, camera.metadata = sdk, Handle(), {}
    camera.created = camera.opened = camera.grabbing = True
    camera._conversion_buffer = None
    return camera, calls


class CameraTests(unittest.TestCase):
    def test_buffer_copied_before_release(self):
        camera, calls = fake_camera()
        frame = camera.read()
        self.assertEqual(calls, ["free"])
        np.testing.assert_array_equal(frame.image, [[0, 1], [2, 3]])
        self.assertEqual(frame.metadata["frame_number"], 7)

    def test_bgr_conversion(self):
        camera, calls = fake_camera(pixel_type=2)
        frame = camera.read()
        self.assertEqual(frame.image.shape, (2, 2, 3))
        self.assertEqual(frame.image[1, 1, 2], 11)
        self.assertEqual(calls, ["free"])

    def test_conversion_failure_releases_buffer(self):
        camera, calls = fake_camera(pixel_type=2, conversion_error=10)
        with self.assertRaises(RuntimeError):
            camera.read()
        self.assertEqual(calls, ["free"])

    def test_mono_high_depth_conversion(self):
        camera, calls = fake_camera(pixel_type=9)
        camera.sdk.PixelType_Gvsp_Mono16 = 9
        frame = camera.read()
        self.assertEqual(frame.image.shape, (2, 2))
        self.assertEqual(calls, ["free"])

    def test_timeout_no_buffer_release(self):
        camera, calls = fake_camera()
        camera.cam.MV_CC_GetImageBuffer = lambda *args: camera.sdk.MV_E_NODATA
        with self.assertRaises(TimeoutError):
            camera.read()
        self.assertEqual(calls, [])

    def test_cleanup_continues_on_sdk_error(self):
        camera, calls = fake_camera()
        with self.assertWarns(UserWarning):
            camera.close()
        camera.close()
        self.assertEqual(calls, ["stop", "close", "destroy"])

    def test_latest_frame_reports_disconnect(self):
        class BrokenCamera:
            def read(self):
                raise RuntimeError("USB disconnected")
        with LatestFrame(BrokenCamera()) as source:
            source._thread.join(timeout=1)
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                source.latest()

    def test_runtime_always_finalizes(self):
        calls = []
        sdk = types.SimpleNamespace(MvCamera=types.SimpleNamespace(
            MV_CC_Initialize=lambda: 0, MV_CC_Finalize=lambda: calls.append("finalize") or 0))
        with self.assertRaises(ValueError):
            with Runtime(sdk):
                raise ValueError("stop")
        self.assertEqual(calls, ["finalize"])

    def test_selection_locking_readback_and_restore(self):
        calls = []
        nodes = {"TriggerMode": 1, "ExposureAuto": 2, "GainAuto": 2, "ExposureTime": 6000., "Gain": 3.,
                 "Width": 960, "Height": 720, "OffsetX": 12, "OffsetY": 8, "PixelFormat": 1,
                 "BinningHorizontal": 1, "BinningVertical": 1, "DecimationHorizontal": 1, "DecimationVertical": 1}
        original = nodes.copy()
        class Handle:
            @staticmethod
            def MV_CC_GetSDKVersion():
                return 123
            def MV_CC_CreateHandle(self, info):
                calls.append(("select", info))
                return 0
            def MV_CC_OpenDevice(self, *args):
                return 0
            def MV_CC_GetEnumValue(self, name, result):
                result.nCurValue = nodes[name]
                return 0
            MV_CC_GetIntValueEx = MV_CC_GetEnumValue
            def MV_CC_GetFloatValue(self, name, result):
                result.fCurValue = nodes[name]
                return 0
            def MV_CC_SetEnumValue(self, name, value):
                nodes[name] = value
                return 0
            MV_CC_SetFloatValue = MV_CC_SetEnumValue
            def MV_CC_StartGrabbing(self):
                return 0
            def MV_CC_StopGrabbing(self):
                return 0
            def MV_CC_CloseDevice(self):
                calls.append("close")
                return 0
            def MV_CC_DestroyHandle(self):
                calls.append("destroy")
                return 0
        sdk = types.SimpleNamespace(MvCamera=Handle, MV_ACCESS_Exclusive=1,
              MVCC_ENUMVALUE=types.SimpleNamespace, MVCC_FLOATVALUE=types.SimpleNamespace, MVCC_INTVALUE_EX=types.SimpleNamespace)
        runtime = types.SimpleNamespace(sdk=sdk, devices=lambda: [
            {"serial": "A", "model": "fake", "_info": 11}, {"serial": "B", "model": "fake", "_info": 22}])
        with self.assertRaisesRegex(ValueError, "Expected one"):
            HikCamera(runtime, None)
        with HikCamera(runtime, "B", exposure_us=5000, gain_db=0) as camera:
            self.assertEqual(camera.metadata["serial"], "B")
            self.assertEqual(camera.metadata["settings"]["OffsetX"], 12)
            self.assertEqual(nodes["ExposureAuto"], 0)
            self.assertEqual(nodes["GainAuto"], 0)
            self.assertEqual(nodes["ExposureTime"], 5000)
        self.assertEqual(nodes, original)
        self.assertEqual(calls, [("select", 22), "close", "destroy"])


def auto_gain_runtime(unsupported=False):
    nodes = {"TriggerMode": 1, "ExposureAuto": 2, "GainAuto": 2, "ExposureTime": 6000., "Gain": 3.,
             "Width": 960, "Height": 720, "OffsetX": 0, "OffsetY": 0, "PixelFormat": 1}
    writes = []
    class Handle:
        @staticmethod
        def MV_CC_GetSDKVersion():
            return 123
        def MV_CC_CreateHandle(self, info):
            return 0
        def MV_CC_OpenDevice(self, *args):
            return 0
        def MV_CC_GetEnumValue(self, name, result):
            if name not in nodes:
                return 1
            result.nCurValue = nodes[name]
            return 0
        MV_CC_GetIntValueEx = MV_CC_GetEnumValue
        def MV_CC_GetFloatValue(self, name, result):
            result.fCurValue = nodes[name]
            return 0
        def MV_CC_SetEnumValue(self, name, value):
            writes.append((name, value))
            nodes[name] = value
            return 0
        def MV_CC_SetFloatValue(self, name, value):
            if name == "Gain" and nodes["GainAuto"] != 0:
                return 1  # real cameras can make Gain read-only while auto is enabled
            writes.append((name, value))
            nodes[name] = value
            return 0
        def MV_CC_SetEnumValueByString(self, name, value):
            if unsupported:
                return 1
            writes.append((name, value))
            nodes[name] = 2
            return 0
        def MV_CC_StartGrabbing(self):
            return 0
        def MV_CC_StopGrabbing(self):
            return 0
        def MV_CC_CloseDevice(self):
            writes.append(("close", None))
            return 0
        def MV_CC_DestroyHandle(self):
            writes.append(("destroy", None))
            return 0
    sdk = types.SimpleNamespace(MvCamera=Handle, MV_ACCESS_Exclusive=1, MV_GAIN_MODE_CONTINUOUS=2,
        MVCC_ENUMVALUE=types.SimpleNamespace, MVCC_FLOATVALUE=types.SimpleNamespace, MVCC_INTVALUE_EX=types.SimpleNamespace)
    runtime = types.SimpleNamespace(sdk=sdk, devices=lambda: [{"serial": "A", "model": "fake", "_info": 11}])
    return runtime, nodes, writes


class AutoGainTests(unittest.TestCase):
    def test_continuous_gain_and_restoration(self):
        runtime, nodes, writes = auto_gain_runtime()
        original = nodes.copy()
        with HikCamera(runtime, "A", exposure_us=50000, gain_auto="continuous") as camera:
            self.assertEqual(nodes["GainAuto"], 2)
            self.assertEqual(nodes["ExposureAuto"], 0)
            self.assertEqual(nodes["ExposureTime"], 50000)
            self.assertEqual(camera.metadata["gain_auto_mode"], "continuous")
            self.assertFalse(any(name == "Gain" for name, _ in writes))
            nodes["Gain"] = 8  # auto gain changed while acquiring
        self.assertEqual(nodes, original)
        self.assertIn(("GainAuto", "Continuous"), writes)
        self.assertEqual(writes[-2:], [("close", None), ("destroy", None)])

    def test_unsupported_auto_gain_reports_error_and_cleans_up(self):
        runtime, nodes, writes = auto_gain_runtime(unsupported=True)
        original = nodes.copy()
        with self.assertRaisesRegex(RuntimeError, "enable continuous"):
            HikCamera(runtime, "A", gain_auto="continuous")
        self.assertEqual(nodes, original)
        self.assertEqual(writes[-2:], [("close", None), ("destroy", None)])

    def test_configuration_conflict_and_legacy_snapshot(self):
        config = load_config()
        config["camera"]["gain_auto"] = "continuous"
        config["camera"]["gain_db"] = 3
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_config(config)
        with self.assertRaises(ValueError):
            HikCamera(None, "A", gain_db=3, gain_auto="continuous")
        old = load_config()
        del old["camera"]["gain_auto"]
        with tempfile.TemporaryDirectory() as tmp:
            session = Session.create(tmp, old, {"serial": "old_camera"}, SIZE)
            reloaded = Session(session.path)
            self.assertEqual(reloaded.config, old)

    def test_cli_auto_and_manual_overrides(self):
        with patch("intrinsic.capture.capture") as capture:
            self.assertEqual(main(["capture", "--gain-auto", "continuous"]), 0)
            self.assertEqual(capture.call_args.args[0]["camera"]["gain_auto"], "continuous")
            self.assertIsNone(capture.call_args.args[0]["camera"]["gain_db"])
            self.assertEqual(main(["capture", "--config", str(Path(__file__).resolve().parents[1]/"configs/pipeline.json"), "--gain-db", "0"]), 0)
            self.assertEqual(capture.call_args.args[0]["camera"]["gain_auto"], "off")
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                main(["capture", "--gain-db", "3", "--gain-auto", "continuous"])

    def test_frame_gain_readback_is_recorded(self):
        camera, calls = fake_camera()
        camera.metadata = {"gain_auto_mode": "continuous", "settings": {"Width": 2, "Height": 2,
                           "PixelFormat": 1, "Gain": 3, "ExposureTime": 50000}}
        camera.read_node = lambda *_: 7.5
        frame = camera.read()
        self.assertEqual(frame.metadata["gain_db"], 7.5)
        self.assertEqual(frame.metadata["gain_auto"], "continuous")
        self.assertEqual(frame.metadata["controls_source"], "node_readback_after_frame")
        self.assertEqual(calls, ["free"])

    def test_frame_metadata_preferred_over_later_readback(self):
        camera, _ = fake_camera()
        camera.metadata = {"gain_auto_mode": "continuous", "settings": {"Width": 2, "Height": 2,
                           "PixelFormat": 1, "Gain": 3, "ExposureTime": 50000}}
        camera.sdk.MV_FRAME_OUT().stFrameInfo.fExposureTime = 50000.
        camera.sdk.MV_FRAME_OUT().stFrameInfo.fGain = 4.5
        camera.read_node = lambda *_: self.fail("Should use SDK frame metadata")
        frame = camera.read()
        self.assertEqual(frame.metadata["gain_db"], 4.5)
        self.assertEqual(frame.metadata["controls_source"], "sdk_frame_info")


class ImagePipelineTests(unittest.TestCase):
    def test_cli_rendered_images_to_calibration_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training, heldout = root/"train", root/"validation"
            training.mkdir()
            heldout.mkdir()
            texture = Board().render(1000)
            extent = Board().size / 2
            board_corners = np.array([[-extent, -extent, 0], [extent, -extent, 0], [extent, extent, 0], [-extent, extent, 0]], np.float32)
            src = np.array([[0, 0], [999, 0], [999, 999], [0, 999]], np.float32)
            for i, (_, rv, tv) in enumerate(synthetic_views(25, distortion=np.zeros(5))):
                dst = cv2.projectPoints(board_corners, rv, tv, K, np.zeros(5))[0].reshape(4, 2)
                transform = cv2.getPerspectiveTransform(src, dst)
                image = cv2.warpPerspective(texture, transform, SIZE, borderValue=180)
                write_image((training if i < 20 else heldout)/f"{i:03d}.png", image)
            self.assertEqual(main(["import-images", "--images", str(training), "--camera-id", "rendered", "--output", str(root/"data")]), 0)
            path = next((root/"data"/"rendered").iterdir())
            self.assertEqual(main(["import-images", "--images", str(heldout), "--camera-id", "rendered", "--session", str(path), "--role", "validation"]), 0)
            self.assertEqual(main(["calibrate", "--session", str(path)]), 0)
            calibration = next((path/"results").glob("calib_*/calibration.json"))
            self.assertEqual(main(["validate", "--session", str(path), "--result", str(calibration)]), 0)
            final = read_json(next((path/"results").glob("validate_*/calibration.json")))
            self.assertLess(final["rms_px"], .5)
            self.assertLess(final["validation"]["rms_px"], .5)
            np.testing.assert_allclose(np.asarray(final["K"])[[0, 1], [0, 1]], K[[0, 1], [0, 1]], rtol=.02)
            self.assertEqual(len(final["views"]), 20)
            self.assertEqual(len(final["validation"]["views"]), 5)


if __name__ == "__main__":
    unittest.main()
