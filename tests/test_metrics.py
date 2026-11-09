"""Residual statistics, report additions and read-only result comparison."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from intrinsic.cli import main
from intrinsic.report import anomaly_summary, compare_results, export_report
from intrinsic.session import read_json, write_json
from intrinsic.solver import add_result_metrics, calibrate, residual_metrics, validate, view_metrics
from test_intrinsic import make_session


class MetricsTests(unittest.TestCase):
    def test_hand_calculated_distances(self):
        m = residual_metrics([[3, 4], [0, 0], [0, -10]])
        self.assertEqual(m["mre_px"], 5)
        self.assertAlmostEqual(m["rmse_px"], np.sqrt(125 / 3))
        self.assertEqual(m["variance_px2"], 25)
        self.assertEqual(m["std_px"], 5)
        self.assertEqual(m["max_error_px"], 10)
        self.assertEqual(m["point_count"], 3)

    def test_edge_cases_and_point_weighting(self):
        self.assertIsNone(residual_metrics([[3, 4]])["std_px"])
        self.assertIsNone(view_metrics([]))
        for bad in ([], [[1, 2, 3]], [[np.nan, 0]]):
            with self.assertRaises(ValueError):
                residual_metrics(bad)
        views = [{"residuals": [[3, 4]]}, {"residuals": [[0, 0], [0, 0], [0, 0]]}]
        self.assertEqual(view_metrics(views)["mre_px"], 1.25)
        self.assertEqual(view_metrics(views)["rmse_px"], 2.5)


class ReportComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.session = make_session(cls.root)
        cls.result = calibrate(cls.session)
        cls.result["validation"] = validate(cls.session, cls.result)
        cls.result["validation_status"] = "evaluated"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_solver_outputs_unchanged_by_statistics(self):
        with patch("intrinsic.solver.residual_metrics", return_value={}):
            without = calibrate(self.session)
            heldout = validate(self.session, without)
        np.testing.assert_array_equal(without["K"], self.result["K"])
        np.testing.assert_array_equal(without["D"], self.result["D"])
        self.assertEqual(without["rms_px"], self.result["rms_px"])
        self.assertEqual(heldout["rms_px"], self.result["validation"]["rms_px"])
        self.assertAlmostEqual(self.result["rms_px"], self.result["metrics"]["rmse_px"], places=4)

    def test_report_csv_and_worst_corner(self):
        result = copy.deepcopy(self.result)
        result["views"][0]["residuals"][7] = [12, -9]
        original_ids = [v["id"] for v in result["views"]]
        original_excluded = result["excluded_frames"][:]
        add_result_metrics(result)
        ranked, worst = anomaly_summary(result["views"])
        self.assertEqual(worst["corner"], 7)
        self.assertEqual(worst["error_px"], 15)
        self.assertEqual(len(ranked), 3)
        self.assertEqual([v["rms_px"] for v in ranked], sorted([v["rms_px"] for v in result["views"]], reverse=True)[:3])
        output = self.session.new_run("metrics_test")
        export_report(self.session, result, output)
        saved = read_json(output / "calibration.json")
        compact = read_json(output / "intrinsics.json")
        self.assertEqual(set(compact), {"camera", "calibrated_at", "image_size", "K", "D", "distortion_order"})
        self.assertEqual(compact["K"], result["K"])
        self.assertEqual(compact["D"], result["D"])
        self.assertEqual(compact["camera"]["serial"], result["camera"]["serial"])
        self.assertTrue(compact["calibrated_at"].endswith("+08:00"))
        self.assertEqual(saved["metrics"], result["metrics"])
        self.assertEqual([v["id"] for v in saved["views"]], original_ids)
        self.assertEqual(saved["excluded_frames"], original_excluded)
        with (output / "per_view.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertAlmostEqual(float(rows[0]["variance_px2"]), result["views"][0]["metrics"]["variance_px2"])
        with (output / "residuals.csv").open(encoding="utf-8-sig", newline="") as f:
            corners = list(csv.DictReader(f))
        self.assertEqual(float(corners[7]["error_px"]), 15)
        html = (output / "report.html").read_text(encoding="utf-8")
        self.assertIn("ddof=1", html)
        self.assertIn("角点 7", html)
        self.assertIn(f'previews/{worst["id"]}.png', html)

    def test_legacy_compare_cli_no_fit_no_input_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = copy.deepcopy(self.result)
            for group in (old, old["validation"]):
                group.pop("metrics")
                group.pop("image_count")
                for view in group["views"]:
                    view.pop("metrics")
            path = root / "old.json"
            write_json(path, old)
            before = path.read_bytes()
            with patch("cv2.calibrateCameraExtended", side_effect=AssertionError("must not fit")), \
                 patch("cv2.solvePnPGeneric", side_effect=AssertionError("must not fit")):
                self.assertEqual(main(["compare", "--baseline", str(path), "--candidate", str(path),
                                       "--output", str(root / "comparison")]), 0)
            comparison = read_json(root / "comparison/comparison.json")
            self.assertTrue(comparison["validation_comparable"])
            self.assertEqual(comparison["delta"]["training_metrics"]["rmse_px"], 0)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                compare_results(path, path, root / "comparison")

    def test_different_sets_missing_validation_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left.json", root / "right.json"
            write_json(left, self.result)
            candidate = copy.deepcopy(self.result)
            removed = candidate["views"].pop()
            candidate["excluded_frames"] = [removed["id"]]
            candidate["validation"]["views"][0]["points"][0][0] += 1
            write_json(right, candidate)
            comparison = compare_results(left, right, root / "changed")
            self.assertFalse(comparison["validation_comparable"])
            self.assertIsNone(comparison["delta"]["validation_metrics"])
            self.assertEqual(len(comparison["warnings"]), 2)
            self.assertEqual(comparison["candidate"]["training_count"], len(candidate["views"]))
            candidate.pop("validation")
            write_json(right, candidate)
            comparison = compare_results(left, right, root / "missing")
            self.assertIsNone(comparison["candidate"]["validation_metrics"])
            for key, value in (("model", "fisheye"), ("image_size", [10, 10]), ("camera", {"serial": "other"})):
                bad = copy.deepcopy(self.result)
                bad[key] = value
                write_json(right, bad)
                with self.assertRaisesRegex(ValueError, key):
                    compare_results(left, right, root / "invalid")
                self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
