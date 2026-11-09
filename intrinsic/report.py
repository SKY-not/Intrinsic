"""JSON/CSV exports, unchanged overlays, and offline HTML report orchestration."""
from __future__ import annotations

import copy
import csv
import os
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import cv2
import numpy as np

from .session import Session, read_json, write_image, write_json
from .report_html import render_report_html, render_detector_comparison_html
from .solver import mark_reviews, add_result_metrics


METRICS = [("mre_px", "MRE (px)"), ("rmse_px", "RMSE (px)"),
           ("std_px", "标准差 (px)"), ("variance_px2", "方差 (px²)"),
           ("max_error_px", "最大误差 (px)")]


def write_calibration(result: dict, path: Path, comparison=False) -> None:
    """Keep computed values; configuration and observations have one source."""
    saved = copy.deepcopy(result)
    saved.pop("config", None)
    saved.pop("config_hash", None)  # Older results remain readable.
    if comparison:
        saved["source_session_id"] = Path(saved.pop("source_session")).name
        for key in ("camera", "image_size"):
            saved.pop(key, None)
        for view in saved["views"] + saved.get("validation", {}).get("views", []):
            view.pop("image", None)
            view.pop("points", None)
    write_json(path, saved)


def read_calibration(path: Path) -> dict:
    """Read both historical results and the compact Compare representation."""
    path = Path(path).resolve()
    result = read_json(path)
    if "source_session" not in result:
        root = next((p for p in path.parents if (p / "input_manifest.json").exists()), None)
        if root is None:
            raise ValueError("Compare calibration requires its input_manifest.json")
        dataset = next((d for d in read_json(root / "input_manifest.json")["datasets"]
                        if (root / d["directory"] / "methods" / path.parent.name).resolve() == path.parent), None)
        if dataset is None or dataset["source_session_id"] != result["source_session_id"]:
            raise ValueError("Calibration/source session mismatch")
        observations = {r["id"]: r["observation"] for r in read_json(path.parent / "detections.json")["frames"]}
        frames = {r["id"]: r for r in dataset["input_frames"]}
        result.update(source_session=str((root / dataset["source_session"]).resolve()),
                      camera=dataset["camera"], image_size=dataset["image_size"],
                      config={"board": dataset["board"], "quality": dataset["quality"],
                              "calibration": dataset["solver"]["calibration"]})
        for view in result["views"] + result.get("validation", {}).get("views", []):
            if view["role"] != frames[view["id"]]["role"]:
                raise ValueError("Calibration/frame role mismatch")
            view.update(points=observations[view["id"]]["corners"], image=frames[view["id"]]["image"])
    elif "config" not in result and (Path(result["source_session"]) / "session.json").exists():
        result["config"] = Session(result["source_session"]).config
    return result


def export_intrinsics(result: dict, output: Path) -> None:
    """Small deployment file; timestamp belongs to calibration, not export."""
    calibrated_at = datetime.fromisoformat(result["created_utc"]).astimezone(
        timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    camera = {key: result["camera"][key] for key in ("serial", "model") if key in result["camera"]}
    write_json(output / "intrinsics.json", {
        "camera": camera, "calibrated_at": calibrated_at,
        "image_size": result["image_size"], "K": result["K"], "D": result["D"],
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
    })


def number(value):
    return "—" if value is None else f"{value:.6f}"


def anomaly_summary(views):
    if not views:
        return [], None
    ranked = sorted(views, key=lambda v: (-v["rms_px"], v["id"]))[:3]
    worst_view = max(views, key=lambda v: v["metrics"]["max_error_px"])
    corner = int(np.argmax(np.linalg.norm(worst_view["residuals"], axis=1)))
    corner_id = worst_view.get("point_ids", range(len(worst_view["points"])))[corner]
    return ranked, {"id": worst_view["id"], "corner": corner_id,
                    "point": worst_view["points"][corner],
                    "error_px": worst_view["metrics"]["max_error_px"]}


def plot_svg(views: list[dict], size: list[int], residuals: bool = False) -> str:
    width, height = size
    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="角点覆盖与残差矢量">',
                f'<rect width="{width}" height="{height}" fill="#f1f5f9"/>']
    for i in range(1, 4):
        elements.append(f'<path d="M {width*i/4} 0 V {height}" stroke="#cbd5e1"/>')
    for i in range(1, 3):
        elements.append(f'<path d="M 0 {height*i/3} H {width}" stroke="#cbd5e1"/>')
    for view in views:
        color = "#047857" if view["role"] == "validation" else "#2563eb"
        for point, error in zip(view["points"], view["residuals"]):
            x, y = point
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{max(width/450, 1):.2f}" fill="{color}" opacity=".55"/>')
            if residuals:
                elements.append(f'<path d="M {x:.2f} {y:.2f} l {error[0]*30:.2f} {error[1]*30:.2f}" stroke="#dc2626" stroke-width="{max(width/1100, 1):.2f}"/>')
    return "".join(elements) + "</svg>"


def write_html_report(result: dict, output: Path) -> None:
    """Refresh HTML from saved results without rewriting data files or previews."""
    display = copy.deepcopy(result)
    if "config" not in display:
        display["config"] = Session(display["source_session"]).config
    add_result_metrics(display)
    training = display["views"]
    validation = display.get("validation", {}).get("views", [])
    mark_reviews(training + validation, display["config"]["calibration"]["review_rms_px"])
    _, worst = anomaly_summary(training)
    size = display["image_size"]
    plots = {"coverage": plot_svg(training, size), "residuals": plot_svg(training, size, True)}
    if validation:
        plots.update(validation_coverage=plot_svg(validation, size),
                     validation_residuals=plot_svg(validation, size, True))
    (Path(output) / "report.html").write_text(render_report_html(display, plots, worst), encoding="utf-8")


def export_report(session: Session, result: dict, output: Path) -> None:
    add_result_metrics(result)
    output.mkdir(parents=True, exist_ok=True)
    K, D = np.asarray(result["K"], float), np.asarray(result["D"], float)
    training = result["views"]
    validation = result.get("validation", {}).get("views", [])
    views = training + validation
    _, worst = anomaly_summary(training)
    mark_reviews(views, result["config"]["calibration"]["review_rms_px"])
    with (output / "per_view.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["id", "role", "rms_px", "max_error_px", "review_status", "review_reason"]
        fields += [key for key, _ in METRICS if key not in fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for v in views:
            record = {**v, **v["metrics"]}
            writer.writerow({k: record[k] for k in writer.fieldnames})
    with (output / "residuals.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "role", "corner", "u_px", "v_px", "du_px", "dv_px", "error_px"])
        for v in views:
            ids = v.get("point_ids", range(len(v["points"])))
            for i, p, delta in zip(ids, v["points"], v["residuals"]):
                writer.writerow([v["id"], v["role"], i, *p, *delta, float(np.linalg.norm(delta))])
    images_dir = output / "previews"
    images_dir.mkdir(exist_ok=True)
    rows = {r["id"]: r for r in session.records}
    maps = None
    if validation:
        size = tuple(result["image_size"])
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, size, 1, size)
        maps = cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_32FC1)
        result["validation"]["preview_camera_matrix"] = new_K.tolist()
        result["validation"]["preview_valid_roi"] = list(roi)
    for v in views:
        image = session.checked_image(rows[v["id"]])
        raw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        points = np.asarray(v["points"])
        projected = points - np.asarray(v["residuals"])
        radius = max(3, int(raw.shape[1] / 500))
        for point, predicted in zip(points, projected):
            cv2.circle(raw, tuple(np.rint(point).astype(int)), radius, (0, 200, 0), 1)
            cv2.drawMarker(raw, tuple(np.rint(predicted).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, radius*3, 1)
        if worst and v["id"] == worst["id"]:
            position = tuple(np.rint(worst["point"]).astype(int))
            cv2.circle(raw, position, radius * 4, (0, 165, 255), 2)
            cv2.putText(raw, f"max corner {worst['corner']}", position,
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 165, 255), 2)
        if v["role"] == "validation":
            corrected = cv2.remap(image, *maps, cv2.INTER_LINEAR)
            if corrected.ndim == 2:
                corrected = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
            raw = np.hstack((raw, corrected))
        scale = min(1, 1600 / raw.shape[1])
        preview = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        write_image(images_dir / f"{v['id']}.png", preview)
    write_html_report(result, output)
    write_calibration(result, output / "calibration.json")
    export_intrinsics(result, output)


def export_corner_comparison(source: Session, row: dict, observations: dict, output: Path, baseline=None) -> str:
    """One raw-pixel crop per scheme; the blue circles are the stated baseline."""
    image = source.checked_image(row)
    raw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    baseline = baseline or next(iter(observations))
    points = np.concatenate([obs.corners for obs in observations.values() if len(obs.corners)])
    lo = np.maximum(np.floor(points.min(axis=0)).astype(int) - 25, 0)
    hi = np.minimum(np.ceil(points.max(axis=0)).astype(int) + 26, image.shape[1::-1])
    ids = source.config["board"]["ids"]
    def indexed(obs):
        return {4 * ids.index(tag_id) + i: point for tag_id, quad in
                zip(obs.ids, obs.corners.reshape(-1, 4, 2)) for i, point in enumerate(quad)}
    reference = indexed(observations[baseline])
    panels = []
    for name, obs in observations.items():
        canvas = raw[lo[1]:hi[1], lo[0]:hi[0]].copy()
        candidate = indexed(obs)
        for i, point in reference.items():
            a = tuple(np.rint((point - lo) * 256).astype(int))
            cv2.circle(canvas, a, 4 * 256, (220, 100, 20), 1, cv2.LINE_AA, shift=8)
            if i in candidate:
                b = tuple(np.rint((candidate[i] - lo) * 256).astype(int))
                cv2.line(canvas, a, b, (0, 180, 0), 1, cv2.LINE_AA, shift=8)
        for i, point in candidate.items():
            b = np.rint(point - lo).astype(int)
            cv2.drawMarker(canvas, tuple(b), (0, 80, 240), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
            cv2.putText(canvas, str(i), tuple(b + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, .35, (50, 180, 30), 1, cv2.LINE_AA)
        panel = cv2.copyMakeBorder(canvas, 30, 0, 1, 1, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(panel, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1, cv2.LINE_AA)
        panels.append(panel)
    relative = f"corner_previews/{row['id']}.png"
    (output / "corner_previews").mkdir(exist_ok=True)
    write_image(output / relative, np.hstack(panels))
    return relative


def comparison_metadata(source, schemes, rows, selected, policy, directory="."):
    """Metadata shared by normal comparisons and migration of saved results."""
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for package in ("numpy", "opencv-python", "pyapriltags"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"source_session": str(source.path), "source_session_id": source.path.name,
        "directory": directory, "schemes_source": None,
        "camera": source.manifest["camera"], "image_size": source.manifest["image_size"],
        "board": source.config["board"], "quality": source.config["quality"],
        "solver": {"function": "cv2.calibrateCameraExtended", "model": "opencv_pinhole_brown5",
                   "flags": 0, "criteria": {"type": 3, "max_iterations": 100, "epsilon": 1e-10},
                   "calibration": source.config["calibration"]},
        "versions": versions, "schemes": copy.deepcopy(schemes), "selection_policy": policy,
        "common_frame_ids": selected,
        "input_frames": [{key: row[key] for key in ("id", "role", "image")} for row in rows],
        "source_validation_count": sum(row["role"] == "validation" for row in rows)}


def read_detector_comparison(output: Path) -> dict:
    """Assemble the report from one manifest and the per-method records."""
    output = Path(output).resolve()
    comparison = read_json(output / "comparison.json")
    manifests = {d["directory"]: d for d in read_json(output / "input_manifest.json")["datasets"]}
    datasets = []
    for summary in comparison["datasets"]:
        dataset = copy.deepcopy(manifests[summary["directory"]])
        dataset.update({key: summary[key] for key in ("baseline", "corner_comparisons", "findings")})
        dataset.update(schemes=[], detection={}, results={}, frames=[])
        for row in dataset["input_frames"]:
            dataset["frames"].append({**row, "preview": summary["corner_previews"].get(row["id"])})
        for name in dataset["methods"]:
            folder = output / dataset["directory"] / "methods" / name
            detections = read_json(folder / "detections.json")
            actual = detections["detector"]
            dataset["schemes"].append({"name": name, "backend": actual["backend"]})
            roles = {row["id"]: row["role"] for row in dataset["input_frames"]}
            if len(detections["frames"]) != len(roles) or {r["id"] for r in detections["frames"]} != set(roles):
                raise ValueError("Detection/input frame IDs mismatch")
            failures = [{"id": r["id"], "role": roles[r["id"]], "issues": r["observation"]["issues"]}
                        for r in detections["frames"] if not r["observation"]["valid"]]
            dataset["detection"][name] = {"detector": actual, "total": len(roles),
                "success_count": len(roles) - len(failures), "failed_frames": failures}
            result = copy.deepcopy(summary["results"][name])
            if result["status"] == "completed":
                calibration = read_calibration(folder / "calibration.json")
                result.update({k: calibration[k] for k in ("K", "D", "std_order", "std_intrinsics",
                    "warnings", "diagnostics", "review_frames", "validation_status", "validation_reason")})
                result.update(training_metrics=calibration["metrics"],
                    validation_metrics=calibration.get("validation", {}).get("metrics"),
                    per_view=[{"id": v["id"], "role": v["role"], "metrics": v["metrics"]}
                              for v in calibration["views"] + calibration.get("validation", {}).get("views", [])])
                result["validation_warnings"] = calibration.get("validation", {}).get("warnings", [])
            dataset["results"][name] = result
        datasets.append(dataset)
    comparison["datasets"] = datasets
    return comparison


def export_detector_comparison(datasets: list[dict], output: Path, notes=None, extra=None, preserve_csv=False) -> None:
    """Export arbitrary scheme combinations without fitting or copying images."""
    comparison = {"schema_version": 2, "datasets": [], **(extra or {})}
    comparison["notes"] = notes if notes is not None else [
        "各方案在相同原始帧上检测，标定板、质量检查、训练/验证用途及求解器配置保持一致。",
        "检测成功率只针对输入会话已保存的帧；图像质量警告不自动排除图像。",
        "角点差异不是外部真值误差，训练重投影误差与独立验证误差分别报告。",
        "参数估计标准差不是重复标定实验的波动；没有独立验证时不作精度合格判断。"]
    manifest = {"schema_version": 2, "datasets": []}
    for ds in datasets:
        entry = {key: copy.deepcopy(ds[key]) for key in (
            "source_session_id", "directory", "camera", "image_size", "board", "quality", "solver", "versions",
            "selection_policy", "common_frame_ids", "source_validation_count", "schemes_source")}
        entry["source_session"] = os.path.relpath(ds["source_session"], output).replace("\\", "/")
        entry["input_frames"] = [{k: r[k] for k in ("id", "role", "image")} for r in ds["input_frames"]]
        entry["methods"] = [scheme["name"] for scheme in ds["schemes"]]
        manifest["datasets"].append(entry)
        comparison["datasets"].append({"directory": ds["directory"],
            "results": {name: {k: r[k] for k in ("status", "reason") if k in r} for name, r in ds["results"].items()},
            "corner_previews": {r["id"]: r["preview"] for r in ds["frames"] if r.get("preview")},
            **{k: ds[k] for k in ("baseline", "corner_comparisons", "findings") if k in ds}})
        comparison["datasets"][-1].setdefault("findings", [])
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "comparison.json", comparison)
    if not preserve_csv:
        with (output / "per_view_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_session", "method", "id", "role", "detection_valid", "mre_px", "rmse_px", "max_error_px"])
            for ds in datasets:
                for name, result in ds["results"].items():
                    views = {v["id"]: v["metrics"] for v in result.get("per_view", [])}
                    failed = {r["id"] for r in ds["detection"][name]["failed_frames"]}
                    for row in ds["input_frames"]:
                        metrics = views.get(row["id"], {})
                        writer.writerow([ds["source_session"], name, row["id"], row["role"], row["id"] not in failed,
                                         *[metrics.get(k) for k in ("mre_px", "rmse_px", "max_error_px")]])
        with (output / "corner_differences.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_session", "method", "reference", "id", "role", "tag_id", "tag_corner",
                             "u_px", "v_px", "reference_u_px", "reference_v_px", "du_px", "dv_px", "distance_px"])
            for ds in datasets:
                names = list(ds["detection"])
                observations = {name: {r["id"]: r["observation"] for r in read_json(
                    output / ds["directory"] / "methods" / name / "detections.json")["frames"]} for name in names}
                reference = ds["baseline"]
                for name in names:
                    if name == reference:
                        continue
                    for row in ds["input_frames"]:
                        a, b = observations[name][row["id"]], observations[reference][row["id"]]
                        if not a["valid"] or not b["valid"]:
                            continue
                        for i, (p, q) in enumerate(zip(a["corners"], b["corners"])):
                            delta = np.asarray(p) - q
                            writer.writerow([ds["source_session"], name, reference, row["id"], row["role"],
                                ds["board"]["ids"][i // 4], i % 4, *p, *q, *delta, float(np.linalg.norm(delta))])
    (output / "comparison.html").write_text(render_detector_comparison_html(read_detector_comparison(output)), encoding="utf-8")


def compare_results(baseline_path: Path, candidate_path: Path, output: Path) -> dict:
    """Compare saved observations only; never fit or modify either input."""
    baseline, candidate = read_calibration(baseline_path), read_calibration(candidate_path)
    for key in ("camera", "image_size", "model"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Cannot compare different {key}")
    for result in (baseline, candidate):
        if result.get("model") != "opencv_pinhole_brown5" or result.get("schema_version") != 1:
            raise ValueError("Unsupported calibration result model/schema")
        if np.asarray(result["K"]).shape != (3, 3) or np.asarray(result["D"]).size != 5:
            raise ValueError("Invalid K/D shape")
        if not np.isfinite(result["K"]).all() or not np.isfinite(result["D"]).all():
            raise ValueError("Non-finite K/D")
        add_result_metrics(result)
    def observed(group):
        return {v["id"]: v["points"] for v in group.get("views", [])}

    bv, cv = baseline.get("validation", {}), candidate.get("validation", {})
    validation_comparable = bool(bv.get("views") and cv.get("views")) and observed(bv) == observed(cv)
    validation_comparable = validation_comparable and all(
        r.get("validation_status", "evaluated") == "evaluated" for r in (baseline, candidate))
    warnings = []
    if not validation_comparable:
        warnings.append("验证图不同、缺失或验证未成功，验证误差不能直接比较。")
    if observed(baseline) != observed(candidate):
        warnings.append("训练图不同或数量减少，不能仅凭训练误差下降判断效果更好。")
    if baseline.get("config") != candidate.get("config"):
        warnings.append("配置不同，请结合标定板及成像设置理解差异。")

    def summary(result):
        val = result.get("validation", {})
        return {"training_count": result["image_count"], "validation_count": len(val.get("views", [])),
                "excluded_frames": result.get("excluded_frames", []),
                "training_metrics": result["metrics"], "validation_metrics": val.get("metrics"),
                "K": result["K"], "D": np.asarray(result["D"]).ravel().tolist()}

    before, after = summary(baseline), summary(candidate)
    delta = {"K": (np.asarray(after["K"]) - np.asarray(before["K"])).tolist(),
             "D": (np.asarray(after["D"]) - np.asarray(before["D"])).tolist()}
    for group in ("training_metrics", "validation_metrics"):
        left, right = before[group], after[group]
        delta[group] = ({key: None if left[key] is None or right[key] is None else right[key] - left[key]
                         for key, _ in METRICS} if left and right and
                        (group == "training_metrics" or validation_comparable) else None)
    comparison = {"baseline_source": str(baseline_path.resolve()),
                  "candidate_source": str(candidate_path.resolve()), "baseline": before, "candidate": after,
                  "delta": delta, "validation_comparable": validation_comparable, "warnings": warnings}
    rows = []
    def row(label, left, right, difference=None):
        rows.append(f"<tr><td>{escape(label)}</td><td>{escape(str(left))}</td>"
                    f"<td>{escape(str(right))}</td><td>{escape(str(difference)) if difference is not None else '—'}</td></tr>")
    for key, label in (("training_count", "训练图数量"), ("validation_count", "验证图数量")):
        row(label, before[key], after[key], after[key] - before[key])
    row("排除名单", ", ".join(before["excluded_frames"]) or "无", ", ".join(after["excluded_frames"]) or "无")
    for group, label in (("training_metrics", "训练"), ("validation_metrics", "验证")):
        for key, metric_label in METRICS:
            row(f"{label} {metric_label}", number((before[group] or {}).get(key)),
                number((after[group] or {}).get(key)), number((delta[group] or {}).get(key)))
    for label, i, j in (("fx", 0, 0), ("fy", 1, 1), ("cx", 0, 2), ("cy", 1, 2)):
        row(f"{label} (px)", number(before["K"][i][j]), number(after["K"][i][j]), number(delta["K"][i][j]))
    for i, label in enumerate(("k1", "k2", "p1", "p2", "k3")):
        row(label, number(before["D"][i]), number(after["D"][i]), number(delta["D"][i]))
    notes = "".join(f"<li>{escape(w)}</li>" for w in warnings)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>标定结果对比</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left;overflow-wrap:anywhere}}</style>
<h1>标定结果对比</h1><p>只读取保存结果，未重新标定。差值 = 新结果 − 原结果。</p><ul>{notes}</ul>
<p>标准差、方差针对角点误差模长 dᵢ，采用 ddof=1；— 表示未计算或不可比较。像素误差不代表毫米测量精度。</p>
<table><tr><th>指标</th><th>原结果</th><th>新结果</th><th>差值</th></tr>{''.join(rows)}</table>
<p>原结果：{escape(str(baseline_path.resolve()))}<br>新结果：{escape(str(candidate_path.resolve()))}</p></html>'''
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "comparison.json", comparison)
    (output / "comparison.html").write_text(html, encoding="utf-8")
    return comparison
