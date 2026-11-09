"""Small OpenCV preview; saved image and observation always share ownership."""
from __future__ import annotations

import time
import copy
from contextlib import contextmanager

import cv2
import numpy as np

from .board import Detector
from .camera import HikCamera, LatestFrame, Runtime
from .session import Session
from .config import resolve_camera_config
from .quality import warning_text


DELETE_KEYS = (3014656, 65535, 127)


class CaptureReview:
    """Own one frozen candidate until the user explicitly keeps or excludes it."""

    def __init__(self, session):
        self.session, self.pending = session, None

    def stage(self, frame, observation, role, metadata):
        if self.pending is not None:
            raise ValueError("Resolve the current candidate before capturing another frame")
        if not observation.valid:
            raise ValueError("; ".join(observation.issues))
        self.pending = {"image": frame.image.copy(), "observation": copy.deepcopy(observation),
                        "role": role, "metadata": copy.deepcopy(metadata), "number": frame.number}
        for warning in observation.warnings:
            print("WARNING: " + warning_text(warning), flush=True)
        print("检测通过；Enter 保留，Delete 排除，Q / Esc 退出。", flush=True)

    def resolve(self, keep, reason):
        if self.pending is None:
            return None
        p = self.pending
        if keep:
            row = self.session.save(p["image"], p["observation"], p["role"],
                                    {**p["metadata"], "decision": "keep", "reason": reason})
            print(f"已保留 {row['id']} ({row['role']})", flush=True)
            result = row
        else:
            result = self.session.save_excluded(p["image"], p["observation"], p["role"], reason, p["metadata"])
            print(f"已排除：{result}", flush=True)
        # A failed write leaves the candidate available for retry.
        self.pending = None
        return result


@contextmanager
def configured_camera(config, serial, config_path, profile_path, overrides):
    with Runtime() as runtime:
        devices = [d for d in runtime.devices() if serial is None or d["serial"] == serial]
        if not devices:
            selected = f" with serial {serial}" if serial else ""
            raise ValueError(f"No matching USB/GigE camera found{selected}. Run devices and check the connection")
        if len(devices) != 1:
            raise ValueError("Multiple USB/GigE cameras found. Select one using devices and --serial")
        serial = devices[0]["serial"]
        config = resolve_camera_config(config, serial, config_path, profile_path, overrides)
        with HikCamera(runtime, serial, **config["camera"]) as camera:
            settings = camera.metadata["settings"]
            print(f"Camera {serial}: {settings['Width']} x {settings['Height']}; gain {camera.metadata['gain_auto_mode']}")
            if camera.metadata["transport"] == "gige":
                network = camera.metadata["network"]
                packet = camera.metadata["transport_settings"]["GevSCPSPacketSize"]
                print(f"GigE {network.get('ip', '?')} via {network.get('interface_ip', '?')}; packet size {packet} bytes")
            yield camera, config


def similar_view(points: np.ndarray, previous: list[np.ndarray], size) -> bool:
    scale = max(size)
    return any(np.sqrt(np.mean(np.sum((points - old)**2, axis=1))) / scale < .02 for old in previous)


def preview(image, observation, records, width, metadata=None, *, kind="four_apriltag", review_role=None):
    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = image.shape[:2]
    cells = set()
    for row in records:
        points = np.asarray(row["observation"]["corners"])
        cells.update(map(tuple, np.minimum((points / [w, h] * [4, 3]).astype(int), [3, 2]).tolist()))
    tint = display.copy()
    for x, y in cells:
        cv2.rectangle(tint, (int(x*w/4), int(y*h/3)), (int((x+1)*w/4), int((y+1)*h/3)), (60, 120, 30), -1)
    cv2.addWeighted(tint, .15, display, .85, 0, display)
    if observation.ids:
        ids = np.asarray(observation.ids, np.int32).reshape(-1, 1)
        if kind == "charuco":
            cv2.aruco.drawDetectedCornersCharuco(display, observation.corners.reshape(-1, 1, 2), ids)
        else:
            cv2.aruco.drawDetectedMarkers(display,
                [p.reshape(1, 4, 2) for p in observation.corners.reshape(-1, 4, 2)], ids)
    scale = min(1, width / w)
    display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    count = sum(r["role"] == "calibration" for r in records)
    status = f"REVIEW {review_role}: Enter keep | Delete exclude | Q/Esc exit" if review_role else "LIVE: Space calibration | V validation | Q/Esc exit"
    detected = f"ChArUco corners: {len(observation.ids)}" if kind == "charuco" else f"Tags: {observation.ids}"
    lines = [status, f"{detected} | {'READY' if observation.valid else 'INCOMPLETE'} | train={count}/30 val={len(records)-count}/5+ | coverage {len(cells)}/12"]
    if metadata and "gain_db" in metadata:
        lines.append(f"Exposure {metadata['exposure_us']:.0f} us | Gain {metadata['gain_db']:.2f} dB | Auto gain: {metadata['gain_auto']}")
    lines += observation.issues + [warning_text(w, display=True) for w in observation.warnings[:6]]
    if len(observation.warnings) > 6:
        lines.append(f"+ {len(observation.warnings)-6} warnings; see terminal for all details")
    canvas = cv2.copyMakeBorder(display, 0, 28 * len(lines) + 12, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, display.shape[0] + 25 + i*28), cv2.FONT_HERSHEY_SIMPLEX, .52,
                    (230, 230, 230) if i < 2 else (70, 200, 255), 1, cv2.LINE_AA)
    return canvas


def capture(config, root, serial=None, *, config_path=None, profile_path=None, camera_overrides=None,
            detector_backend=None):
    detector = Detector(config, backend=detector_backend)
    window = "HIK Intrinsic - " + ("ChArUco" if detector.board.kind == "charuco" else "AprilTag")
    session = None
    review = None
    reviewed_numbers = set()
    displayed = observation = None
    last_detection = 0.0
    print("采集前固定镜头对焦和光圈。Space 标定候选，V 验证候选；Enter 保留，Delete 排除，Q / Esc 退出。")
    print("请移动到画面中央、四边和四角，并绕横向、纵向倾斜；每次停稳后保存。")
    try:
        with configured_camera(config, serial, config_path, profile_path, camera_overrides) as (camera, config), LatestFrame(camera) as stream:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            while True:
                frame = stream.latest()
                now = time.monotonic()
                if (review is None or review.pending is None) and frame is not None and (displayed is None or frame.number != displayed.number) and now - last_detection >= 1/config["capture"]["detection_hz"]:
                    displayed = frame
                    observation = detector.detect(frame.image, check_quality=False)
                    last_detection = now
                    first_preview = session is None
                    if first_preview:
                        session = Session.create(root, config, camera.metadata, frame.image.shape[1::-1])
                        review = CaptureReview(session)
                        print(f"会话：{session.path}")
                    canvas = preview(frame.image, observation, session.records, config["capture"]["preview_width"], frame.metadata, kind=detector.board.kind)
                    cv2.imshow(window, canvas)
                    if first_preview:
                        cv2.resizeWindow(window, canvas.shape[1], canvas.shape[0])
                key = cv2.waitKeyEx(15)
                if key in (ord("q"), ord("Q"), 27) or (displayed is not None and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1):
                    break
                if review is not None and review.pending is not None:
                    if key in (10, 13) or key in DELETE_KEYS:
                        try:
                            review.resolve(key in (10, 13), "user_confirmed" if key in (10, 13) else "user_deleted")
                        except (ValueError, OSError, cv2.error) as error:
                            print(f"保存失败，候选图仍待确认：{error}")
                            continue
                        canvas = preview(displayed.image, observation, session.records, config["capture"]["preview_width"], displayed.metadata, kind=detector.board.kind)
                        cv2.imshow(window, canvas)
                    continue
                if key not in (32, ord("v"), ord("V")) or displayed is None:
                    continue
                if displayed.number in reviewed_numbers:
                    print("该帧已提交复核；请移动标定板并等待新图像。")
                    continue
                try:
                    observation = detector.detect(displayed.image)
                    role = "calibration" if key == 32 else "validation"
                    review.stage(displayed, observation, role, {**displayed.metadata, "detector": detector.description})
                except (ValueError, OSError, cv2.error) as error:
                    print(f"不能提交候选图：{error}")
                    continue
                reviewed_numbers.add(displayed.number)
                p = review.pending
                canvas = preview(p["image"], p["observation"], session.records, config["capture"]["preview_width"],
                                 p["metadata"], kind=detector.board.kind, review_role=role)
                cv2.imshow(window, canvas)
    finally:
        try:
            if review is not None and review.pending is not None:
                review.resolve(False, "exit_without_confirmation")
        finally:
            cv2.destroyAllWindows()
            if session is not None:
                print(f"数据保存在 {session.path}；接下来运行 calibrate --session。")
    return session
