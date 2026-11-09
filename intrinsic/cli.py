from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .board import Board, Detector, load_config, validate_config
from .config import DEFAULT_PATH, DEFAULT_COMPARE_PATH, detector_settings, load_compare_schemes
from .report import export_report, compare_results, export_corner_comparison, export_detector_comparison, comparison_metadata, write_calibration, read_calibration
from .session import Session, new_id, read_image, read_json, write_image, write_json
from .solver import calibrate, apply_validation, residual_metrics
from .quality import warning_text


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data"


def configured_detection(config, args):
    backend = detector_settings(config, args.detector)["backend"]
    choices = {"opencv": {"corner_refinement": getattr(args, "corner_refinement", None),
                          "window_size": getattr(args, "corner_window_size", None)},
               "pyapriltags": {"quad_decimate": getattr(args, "quad_decimate", None),
                               "refine_edges": getattr(args, "refine_edges", None)}, "opencv_charuco": {}}
    if any(value is not None for name, params in choices.items() if name != backend for value in params.values()):
        raise ValueError("CLI detector parameters must belong to the selected backend")
    parameters = {key: value for key, value in choices[backend].items() if value is not None}
    resolved = copy.deepcopy(config)
    resolved["detector"] = detector_settings(config, backend, parameters)
    return resolved


def detector_arguments(command):
    command.add_argument("--detector", choices=["auto", "opencv", "pyapriltags"], default=None,
                         help="覆盖 detector.backend；未指定时采用配置，旧配置默认为 opencv")
    command.add_argument("--corner-refinement", type=str.upper, choices=["NONE", "SUBPIX", "CONTOUR", "APRILTAG"])
    command.add_argument("--corner-window-size", type=int, help="OpenCV 精化窗口上限，单位 px")
    command.add_argument("--quad-decimate", type=float, help="pyapriltags 四边形检测降采样倍数")
    command.add_argument("--refine-edges", type=int, choices=[0, 1], help="pyapriltags 边缘精化开关")


def import_images(args):
    config = load_config(args.config)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    if not paths:
        raise ValueError("No supported images in the input directory")
    session = Session(args.session) if args.session else None
    if session:
        if session.readonly:
            raise ValueError("Cannot import images into a read-only experiment session")
        declared = copy.deepcopy(config)
        if "detector" not in session.config:
            declared.pop("detector", None)
        if args.config and declared != session.config:
            raise ValueError("Import configuration differs from immutable session configuration")
        config = session.config
        if session.manifest["camera"]["serial"] != args.camera_id:
            raise ValueError("Imported camera identity differs from session")
    config = configured_detection(config, args)
    detector = Detector(config)
    accepted, rejected = [], []
    for path in paths:
        try:
            image = read_image(path)
            if session and list(image.shape[1::-1]) != session.manifest["image_size"]:
                raise ValueError("Image resolution differs from session")
            observation = detector.detect(image)
            if not observation.valid:
                raise ValueError("; ".join(observation.issues))
            for warning in observation.warnings:
                print(f"{path.name} WARNING: {warning_text(warning)}")
            if session is None:
                session = Session.create(args.output, config,
                    {"serial": args.camera_id, "source": "import", "settings": "declared same camera/lens/ROI; not independently verified"}, image.shape[1::-1])
                print(f"会话：{session.path}")
            row = session.save(image, observation, args.role,
                               {"source_path": str(path.resolve()), "detector": detector.description})
            accepted.append({"source": str(path), "id": row["id"]})
        except (ValueError, OSError, cv2.error) as error:
            rejected.append({"source": str(path), "reason": str(error)})
            print(f"跳过 {path.name}: {error}")
    if session is None:
        raise ValueError("No valid calibration-board images found")
    write_json(session.path / f"import_{new_id()}.json", {"accepted": accepted, "rejected": rejected})
    print(f"导入 {len(accepted)} 张，跳过 {len(rejected)} 张。会话：{session.path}")
    if not accepted:
        raise ValueError("No new valid images imported")


def compare_detectors(args):
    """Compare configured schemes through shared Detector, Session and solver."""
    sources = [Session(path) for path in args.session]
    if any(Board.from_config(source.config).kind != "four_apriltag" for source in sources):
        raise ValueError("compare-detectors currently supports four-AprilTag sessions only")
    if len({source.path for source in sources}) != len(sources):
        raise ValueError("Duplicate input session")
    fixed = getattr(args, "frame_ids", None)
    if fixed and len(sources) != 1:
        raise ValueError("--frame-ids requires exactly one source session")
    schemes = load_compare_schemes(getattr(args, "schemes", None))
    output = Path(args.output).resolve()
    if any(output == source.path or source.path in output.parents for source in sources):
        raise ValueError("Comparison output must be outside original sessions")
    output.mkdir(parents=True, exist_ok=False)
    datasets = []
    for index, source in enumerate(sources, 1):
        if source.readonly:
            raise ValueError("Compare inputs must be original capture sessions")
        rows = source.records
        if fixed:
            if len(set(fixed)) != len(fixed) or set(fixed) - {r["id"] for r in rows}:
                raise ValueError("Invalid fixed frame IDs")
            rows = [row for row in rows if row["id"] in set(fixed)]
        detectors = {s["name"]: Detector(source.config, backend=s["backend"],
                     parameters=detector_settings({}, s["backend"], s["parameters"])[s["backend"]]) for s in schemes}
        detected = {name: {} for name in detectors}
        directory = "." if len(sources) == 1 else f"{index:02d}_{source.path.parent.name}_{source.path.name}"
        folder = output / directory
        (folder / "methods").mkdir(parents=True)
        for row in rows:
            image = source.checked_image(row)
            for name, detector in detectors.items():
                detected[name][row["id"]] = detector.detect(image)
        shared = {r["id"] for r in rows if all(detected[name][r["id"]].valid for name in detectors)}
        used = {r["id"] for r in rows} if fixed else shared
        selected = {role: [r["id"] for r in rows if r["role"] == role and r["id"] in used]
                    for role in ("calibration", "validation")}
        dataset = comparison_metadata(source, schemes, rows, selected,
                                      "fixed" if fixed else "common_success", directory)
        dataset["schemes_source"] = str(Path(getattr(args, "schemes", None) or DEFAULT_COMPARE_PATH).resolve())
        dataset.update(detection={}, results={}, frames=[], corner_comparisons={})
        baseline = next((s["name"] for s in schemes if s["backend"] == "opencv" and
                         detector_settings({}, s["backend"], s["parameters"])["opencv"]["corner_refinement"] == "SUBPIX"),
                        schemes[0]["name"])
        dataset["baseline"] = baseline
        for name, detector in detectors.items():
            method_folder = folder / "methods" / name
            method_folder.mkdir()
            failures = [{"id": r["id"], "role": r["role"], "issues": detected[name][r["id"]].issues}
                        for r in rows if not detected[name][r["id"]].valid]
            dataset["detection"][name] = {"detector": detector.description, "total": len(rows),
                "success_count": len(rows) - len(failures), "failed_frames": failures}
            detection_path = method_folder / "detections.json"
            detections = {"detector": detector.description,
                "frames": [{"id": r["id"], "observation": detected[name][r["id"]].to_dict()} for r in rows]}
            write_json(detection_path, detections)
            if fixed and failures:
                dataset["results"][name] = {"status": "not_run", "reason": "固定帧存在检测失败；未缩小集合，未运行标定。"}
                continue
            session = Session.view(source, detections, [r["id"] for r in rows if r["id"] in used])
            try:
                result = calibrate(session)
                apply_validation(session, result, "auto")
                write_calibration(result, method_folder / "calibration.json", comparison=True)
                dataset["results"][name] = {"status": "completed",
                    **{k: result[k] for k in ("K", "D", "std_order", "std_intrinsics", "warnings",
                                              "validation_status", "validation_reason")},
                    "training_metrics": result["metrics"],
                    "validation_metrics": result.get("validation", {}).get("metrics"),
                    "per_view": [{"id": v["id"], "role": v["role"], "metrics": v["metrics"]}
                                 for v in result["views"] + result.get("validation", {}).get("views", [])]}
            except (ValueError, RuntimeError, OSError, cv2.error) as error:
                dataset["results"][name] = {"status": "failed", "reason": str(error)}
        for row in rows:
            observations = {name: detected[name][row["id"]] for name in detectors}
            frame = {"id": row["id"], "role": row["role"], "common_success": row["id"] in shared,
                     "used_for_comparison": row["id"] in used}
            if any(len(obs.corners) for obs in observations.values()):
                frame["preview"] = export_corner_comparison(source, row, observations, folder, baseline)
            dataset["frames"].append(frame)
        for name in detectors:
            if name == baseline:
                continue
            deltas = [detected[name][r["id"]].corners.astype(float) - detected[baseline][r["id"]].corners
                      for r in rows if r["id"] in used and detected[name][r["id"]].valid and detected[baseline][r["id"]].valid]
            complete = bool(used) and len(deltas) == len(used)
            dataset["corner_comparisons"][name] = {baseline: {
                "compared_frame_count": len(deltas), "status": "complete" if complete else "incomplete",
                "metrics": residual_metrics(np.concatenate(deltas)) if complete else None}}
        datasets.append(dataset)
        export_detector_comparison(datasets, output)
    print(f"检测器对比报告：{output / 'comparison.html'}")
    if any(r["status"] != "completed" or r.get("validation_status") == "failed"
           for ds in datasets for r in ds["results"].values()):
        raise RuntimeError("One or more fits/validations failed; inspect comparison.json without changing the solver")
    return datasets


def parser():
    p = argparse.ArgumentParser(description="海康 USB / GigE 相机 AprilTag / ChArUco 内参标定")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="列出海康 USB / GigE 网口相机及网络地址")
    cap = commands.add_parser("capture", help="实时预览、标定/验证图采集")
    cap.add_argument("--serial", help="相机序列号；多台相机时必须指定")
    cap.add_argument("--config", default=str(DEFAULT_PATH), help="程序配置，默认使用项目 configs/pipeline.json")
    cap.add_argument("--camera-config", help="覆盖按序列号自动选择的相机配置")
    cap.add_argument("--resolution", choices=["max", "current"], help="max 全幅最大分辨率（默认）；current 沿用相机设置")
    cap.add_argument("--output", default=str(DEFAULT_ROOT), help="会话存储根目录")
    detector_arguments(cap)
    cap.add_argument("--exposure-us", type=float, help="固定曝光时间（微秒）")
    gain = cap.add_mutually_exclusive_group()
    gain.add_argument("--gain-db", type=float, help="固定增益（dB），关闭自动增益")
    gain.add_argument("--gain-auto", choices=["off", "continuous"], help="自动增益模式：off 关闭，continuous 持续自动调整")
    fit = commands.add_parser("calibrate", help="使用会话中的标定图求内参")
    fit.add_argument("--session", required=True)
    fit.add_argument("--exclude", nargs="*", default=[], metavar="FRAME_ID")
    fit.add_argument("--validation", choices=["auto", "required", "off"], default="auto", help="auto 有验证图就验证（默认）；required 必须验证；off 跳过")
    val = commands.add_parser("validate", help="固定内参，在留出图像上验证")
    val.add_argument("--session", required=True)
    val.add_argument("--result", required=True, help="calibration.json 路径")
    compare = commands.add_parser("compare", help="比较两份已有结果，不重新标定")
    compare.add_argument("--baseline", required=True, help="原 calibration.json")
    compare.add_argument("--candidate", required=True, help="新 calibration.json")
    compare.add_argument("--output", required=True, help="新的对比目录（不能已存在）")
    detection = commands.add_parser("compare-detectors", help="在相同原始帧上离线比较两种检测器；各会话独立标定")
    detection.add_argument("--session", nargs="+", required=True, help="一个或多个已有会话目录，保留原训练/验证用途")
    detection.add_argument("--schemes", default=str(DEFAULT_COMPARE_PATH), help="方案 JSON 列表；默认 configs/compare.json")
    detection.add_argument("--frame-ids", nargs="+", help="固定单会话帧集合；失败方案不缩小集合标定")
    detection.add_argument("--output", default=str(DEFAULT_ROOT / "test_results" / f"detector_compare_{new_id()}"), help="新的对比目录（不能已存在）")
    imp = commands.add_parser("import-images", help="导入同一相机、镜头、ROI 的离线图像")
    imp.add_argument("--images", required=True)
    imp.add_argument("--config")
    imp.add_argument("--camera-id", required=True, help="源相机序列号或独立身份；用户确认所有图片属于同一成像设置")
    imp.add_argument("--role", choices=["calibration", "validation"], default="calibration")
    imp.add_argument("--session", help="追加到已有会话（例如导入验证图）")
    imp.add_argument("--output", default=str(DEFAULT_ROOT))
    detector_arguments(imp)
    board = commands.add_parser("board-image", help="导出参考板图；打印时自行核实实际尺寸")
    board.add_argument("--config")
    board.add_argument("--output", required=True, help="PNG 文件路径")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "devices":
            from .camera import Runtime
            with Runtime() as runtime:
                devices = [{k: v for k, v in d.items() if not k.startswith("_")}
                           for d in runtime.devices(include_gige=True)]
                print(json.dumps(devices, ensure_ascii=False, indent=2))
                if not devices:
                    print("未发现海康 USB / GigE 相机。请检查相机供电、USB/网线连接和 MVS 驱动；网口相机还需检查网卡及 IP 配置。")
        elif args.command == "capture":
            from .capture import capture
            config = load_config(args.config)
            overrides = {}
            for key in ("exposure_us", "gain_db", "resolution"):
                if getattr(args, key) is not None:
                    overrides[key] = getattr(args, key)
            if args.gain_db is not None:
                overrides["gain_auto"] = "off"
            elif args.gain_auto is not None:
                overrides["gain_auto"] = args.gain_auto
                if args.gain_auto == "continuous":
                    overrides["gain_db"] = None
            config["camera"].update(overrides)
            config = configured_detection(config, args)
            validate_config(config)
            capture(config, args.output, args.serial, config_path=args.config,
                    profile_path=args.camera_config, camera_overrides=overrides, detector_backend=args.detector)
        elif args.command == "import-images":
            import_images(args)
        elif args.command == "compare-detectors":
            compare_detectors(args)
        elif args.command == "compare":
            comparison = compare_results(Path(args.baseline), Path(args.candidate), Path(args.output))
            for warning in comparison["warnings"]:
                print(f"提示：{warning}")
            print(f"对比报告：{Path(args.output) / 'comparison.html'}")
        elif args.command == "board-image":
            path = Path(args.output)
            if path.suffix.lower() != ".png":
                raise ValueError("Reference board must use lossless .png format")
            if path.exists():
                raise ValueError("Output already exists; choose a new filename")
            path.parent.mkdir(parents=True, exist_ok=True)
            config = load_config(args.config)
            board = Board.from_config(config)
            write_image(path, board.render())
            if board.kind == "charuco":
                b = config["board"]
                width, height = [n * b["square_length_m"] * 1000 for n in b["squares"]]
                print(f"参考图：{path}；ChArUco 棋盘区域 {width:g} x {height:g} mm，格长 {b['square_length_m']*1000:g} mm；打印后必须实测。")
            else:
                print(f"参考图：{path}；整板应打印为 {board.size*1000:g} mm，标签黑边宽 {board.tag_size*1000:g} mm；必须实测。")
        elif args.command in ("calibrate", "validate"):
            session = Session(args.session)
            if args.command == "calibrate":
                result = calibrate(session, args.exclude)
                validation_ok = apply_validation(session, result, args.validation)
            else:
                result = read_calibration(Path(args.result))
                result.setdefault("config", session.config)
                validation_ok = apply_validation(session, result, "required")
                result["validation_source_result"] = str(Path(args.result).resolve())
            output = session.new_run(args.command)
            export_report(session, result, output)
            print(f"标定 RMS：{result['rms_px']:.4f} px；{len(result['views'])} 张标定图。")
            if "validation" in result:
                print(f"留出图像验证 RMS：{result['validation']['rms_px']:.4f} px。")
            else:
                print(f"验证状态：{result['validation_status']}；{result['validation_reason']}")
            for warning in result["warnings"] + result.get("validation", {}).get("warnings", []):
                print(f"提示：{warning}")
            print(f"结果：{output / 'calibration.json'}\n报告：{output / 'report.html'}")
            print(f"简洁内参文件：{output / 'intrinsics.json'}")
            if not validation_ok:
                return 1
        return 0
    except KeyboardInterrupt:
        print("已停止。已保存的数据保持完整。", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, cv2.error) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
