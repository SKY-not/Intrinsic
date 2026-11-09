"""Small OpenCV preview; saved image and observation always share ownership."""
from __future__ import annotations

import time
from contextlib import contextmanager

import cv2
import numpy as np

from .board import Detector
from .camera import HikCamera, LatestFrame, Runtime
from .session import Session
from .config import resolve_camera_config


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


def preview(image, observation, records, width, metadata=None):
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
        cv2.aruco.drawDetectedMarkers(display,
            [p.reshape(1, 4, 2) for p in observation.corners.reshape(-1, 4, 2)],
            np.asarray(observation.ids, np.int32).reshape(-1, 1))
    scale = min(1, width / w)
    display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    count = sum(r["role"] == "calibration" for r in records)
    lines = [f"Space: calibration  V: validation  Q: quit | train={count}/30 val={len(records)-count}/5+",
             f"Tags: {observation.ids} | {'READY' if observation.valid else 'INCOMPLETE'} | coverage {len(cells)}/12"]
    if metadata and "gain_db" in metadata:
        lines.append(f"Exposure {metadata['exposure_us']:.0f} us | Gain {metadata['gain_db']:.2f} dB | Auto gain: {metadata['gain_auto']}")
    lines += observation.issues + observation.warnings
    canvas = cv2.copyMakeBorder(display, 0, 28 * len(lines) + 12, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, display.shape[0] + 25 + i*28), cv2.FONT_HERSHEY_SIMPLEX, .52,
                    (230, 230, 230) if i < 2 else (70, 200, 255), 1, cv2.LINE_AA)
    return canvas


def capture(config, root, serial=None, *, config_path=None, profile_path=None, camera_overrides=None,
            detector_backend=None):
    detector = Detector(config, backend=detector_backend)
    window = "HIK Intrinsic - AprilTag"
    session = None
    previous = []
    saved_numbers = set()
    displayed = observation = None
    last_detection = 0.0
    print("采集前固定镜头对焦和光圈。Space 保存标定图，V 保存独立验证图，Q 退出。")
    print("请移动到画面中央、四边和四角，并绕横向、纵向倾斜；每次停稳后保存。")
    try:
        with configured_camera(config, serial, config_path, profile_path, camera_overrides) as (camera, config), LatestFrame(camera) as stream:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            while True:
                frame = stream.latest()
                now = time.monotonic()
                if frame is not None and (displayed is None or frame.number != displayed.number) and now - last_detection >= 1/config["capture"]["detection_hz"]:
                    displayed = frame
                    observation = detector.detect(frame.image)
                    if observation.valid and similar_view(observation.corners, previous, frame.image.shape[1::-1]):
                        observation.warnings.append("similar saved view; change position/tilt")
                    last_detection = now
                    first_preview = session is None
                    if first_preview:
                        session = Session.create(root, config, camera.metadata, frame.image.shape[1::-1])
                        print(f"会话：{session.path}")
                    canvas = preview(frame.image, observation, session.records, config["capture"]["preview_width"], frame.metadata)
                    cv2.imshow(window, canvas)
                    if first_preview:
                        cv2.resizeWindow(window, canvas.shape[1], canvas.shape[0])
                key = cv2.waitKey(15) & 0xff
                if key in (ord("q"), ord("Q"), 27) or (displayed is not None and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1):
                    break
                if key not in (32, ord("v"), ord("V")) or displayed is None:
                    continue
                if displayed.number in saved_numbers:
                    print("该帧已保存；请移动标定板并等待新图像。")
                    continue
                if not observation.valid:
                    print("不能保存：" + "; ".join(observation.issues))
                    continue
                try:
                    row = session.save(displayed.image, observation,
                                       "calibration" if key == 32 else "validation",
                                       {**displayed.metadata, "detector": detector.description})
                    saved_numbers.add(displayed.number)
                    previous.append(observation.corners.copy())
                    print(f"已保存 {row['id']} ({row['role']})" + ("；" + "; ".join(observation.warnings) if observation.warnings else ""))
                except (ValueError, OSError) as error:
                    print(f"保存失败：{error}")
    finally:
        cv2.destroyAllWindows()
        if session is not None:
            print(f"数据保存在 {session.path}；接下来运行 calibrate --session。")
    return session
