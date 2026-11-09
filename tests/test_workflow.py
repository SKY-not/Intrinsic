"""Configuration, native resolution and unified reporting regressions."""
import copy
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from intrinsic.config import DEFAULT_PATH, load_config, resolve_camera_config
from intrinsic.camera import HikCamera
from intrinsic.cli import main, parser
from intrinsic.session import Session, write_json
from intrinsic.solver import calibrate, apply_validation, validate, mark_reviews
from intrinsic.report import export_report
from test_intrinsic import make_session


class ConfigTests(unittest.TestCase):
    def test_default_path_independent_of_working_directory(self):
        before = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertEqual(parser().parse_args(["capture"]).config, str(DEFAULT_PATH))
                self.assertEqual(load_config()["board"]["tag_size_m"], .02)
            finally:
                os.chdir(before)

    def test_profile_identity_and_override_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"cameras").mkdir()
            write_json(root/"app.json", {"camera_dir": "cameras"})
            write_json(root/"cameras/A.json", {"serial": "A", "gain_auto": "continuous", "exposure_us": 1234})
            write_json(root/"cameras/B.json", {"serial": "B", "gain_db": 4})
            cfg = load_config(root/"app.json")
            a = resolve_camera_config(cfg, "A", root/"app.json")
            b = resolve_camera_config(cfg, "B", root/"app.json")
            self.assertEqual(a["camera"]["exposure_us"], 1234)
            self.assertEqual(a["camera"]["gain_auto"], "continuous")
            self.assertEqual(b["camera"]["gain_db"], 4)
            self.assertEqual(cfg["camera"]["gain_auto"], "off")
            manual = resolve_camera_config(cfg, "A", root/"app.json", overrides={"gain_auto": "off", "gain_db": 0})
            self.assertEqual(manual["camera"]["gain_auto"], "off")
            with self.assertRaisesRegex(ValueError, "serial"):
                resolve_camera_config(cfg, "A", root/"app.json", root/"cameras/B.json")

    def test_unknown_camera_default_and_legacy_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/"app.json", {})
            config = resolve_camera_config(load_config(), "NEW", root/"app.json")
            self.assertEqual(config["camera"]["resolution"], "max")
            for field in ("resolution", "lens_note", "gain_auto"):
                config["camera"].pop(field)
            session = Session.create(root, config, {"serial": "legacy"}, (960, 720))
            self.assertEqual(Session(session.path).config, config)


def geometry_camera(fail_node=None, ignore_node=None):
    nodes = {"Width": 400, "Height": 300, "OffsetX": 12, "OffsetY": 8,
             "BinningHorizontal": 2, "BinningVertical": 2,
             "DecimationHorizontal": 1, "DecimationVertical": 1}
    writes = []
    class Handle:
        def MV_CC_GetIntValueEx(self, node, value):
            value.nCurValue = nodes[node]
            if node in ("Width", "Height"):
                axis = "Horizontal" if node == "Width" else "Vertical"
                limit = (1002 if node == "Width" else 802) // nodes["Binning"+axis]
                value.nMin, value.nMax, value.nInc = 8, limit, 8
            elif node.startswith("Offset"):
                value.nMin, value.nMax, value.nInc = 0, 600, 1
            else:
                value.nMin, value.nMax, value.nInc = 1, 2, 1
            return 0
        def MV_CC_SetIntValueEx(self, node, value):
            writes.append((node, value))
            if node == fail_node:
                return 1
            if node != ignore_node:
                nodes[node] = value
            return 0
    camera = HikCamera.__new__(HikCamera)
    camera.sdk = types.SimpleNamespace(MVCC_INTVALUE_EX=types.SimpleNamespace)
    camera.cam = Handle()
    camera._geometry_original, camera._geometry_notes = {}, []
    return camera, nodes, writes


class ResolutionTests(unittest.TestCase):
    def test_native_maximum_steps_and_restoration(self):
        camera, nodes, writes = geometry_camera()
        original = nodes.copy()
        camera.configure_max_resolution()
        self.assertEqual((nodes["Width"], nodes["Height"]), (1000, 800))
        self.assertEqual(nodes["BinningHorizontal"], 1)
        self.assertEqual(nodes["BinningVertical"], 1)
        self.assertEqual((nodes["OffsetX"], nodes["OffsetY"]), (0, 0))
        errors = []
        camera.restore_geometry(errors)
        self.assertEqual(errors, [])
        self.assertEqual(nodes, original)

    def test_unsupported_or_ignored_setting_is_not_silent(self):
        for kwargs in ({"fail_node": "BinningHorizontal"}, {"ignore_node": "BinningHorizontal"}):
            camera, nodes, _ = geometry_camera(**kwargs)
            original = nodes.copy()
            with self.assertRaises(RuntimeError):
                camera.configure_max_resolution()
            errors = []
            camera.restore_geometry(errors)
            self.assertEqual(nodes, original)


class WorkflowTests(unittest.TestCase):
    def test_short_run_names_use_beijing_time_and_preserve_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Session.__new__(Session)
            session.path = Path(tmp)
            instant = datetime(2026, 10, 16, 16, 30, tzinfo=timezone.utc)
            with patch("intrinsic.session.datetime") as clock:
                clock.now.side_effect = lambda tz: instant.astimezone(tz)
                first = session.new_run("calibrate")
                marker = first / "keep.txt"
                marker.write_text("keep", encoding="utf-8")
                second = session.new_run("calibrate")
                third = session.new_run("calibrate")
                self.assertEqual(first.name, "calib_202610170030")
                self.assertEqual(second.name, "calib_202610170030_01")
                self.assertEqual(third.name, "calib_202610170030_02")
                self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.session = make_session(cls.tmp.name)
        cls.result = calibrate(cls.session)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_auto_required_off_and_failure(self):
        result = copy.deepcopy(self.result)
        self.assertTrue(apply_validation(self.session, result, "auto"))
        self.assertEqual(result["validation_status"], "evaluated")
        self.assertEqual(len(result["validation"]["views"]), 5)
        self.assertTrue(apply_validation(self.session, result, "off"))
        self.assertEqual(result["validation_status"], "skipped_off")
        self.assertNotIn("validation", result)
        training_only = types.SimpleNamespace(records=[r for r in self.session.records if r["role"] == "calibration"])
        self.assertTrue(apply_validation(training_only, result))
        self.assertEqual(result["validation_status"], "skipped_no_data")
        self.assertFalse(apply_validation(training_only, result, "required"))
        self.assertEqual(result["validation_status"], "missing_required")
        with patch("intrinsic.solver.validate", side_effect=ValueError("damaged validation image")):
            self.assertFalse(apply_validation(self.session, result))
        self.assertEqual(result["validation_status"], "failed")
        self.assertIn("damaged", result["validation_reason"])
        self.assertEqual(result["K"], self.result["K"])

    def test_one_command_one_report_and_failure_exit_code(self):
        for failed in (False, True):
            with patch("intrinsic.cli.calibrate", return_value=copy.deepcopy(self.result)), patch("intrinsic.cli.export_report") as export:
                if failed:
                    with patch("intrinsic.solver.validate", side_effect=ValueError("test failure")):
                        code = main(["calibrate", "--session", str(self.session.path)])
                else:
                    code = main(["calibrate", "--session", str(self.session.path)])
                self.assertEqual(code, 1 if failed else 0)
                self.assertEqual(export.call_count, 1)
                self.assertEqual(export.call_args.args[1]["validation_status"], "failed" if failed else "evaluated")

    def test_validation_outlier_is_flagged_in_csv(self):
        session = Session(self.session.path)
        row = next(r for r in session._records if r["role"] == "validation")
        row["observation"]["corners"][0][0] += 15
        result = copy.deepcopy(self.result)
        self.assertTrue(apply_validation(session, result))
        bad = next(v for v in result["validation"]["views"] if v["id"] == row["id"])
        self.assertEqual(bad["review_status"], "needs_review")
        output = session.new_run("test_review")
        export_report(session, result, output)
        with (output/"per_view.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        exported = next(r for r in rows if r["id"] == row["id"])
        self.assertEqual(exported["review_status"], "needs_review")
        self.assertNotIn("review", exported)
        self.assertIn("exceeds", exported["review_reason"])

    def test_review_boundary_and_unchecked(self):
        rows = [{"rms_px": 1.0}, {"rms_px": 1.01}, {}]
        mark_reviews(rows, 1.)
        self.assertEqual([r["review_status"] for r in rows], ["not_flagged", "needs_review", "not_checked"])
