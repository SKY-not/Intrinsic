"""Lazy MVS adapter. SDK lifetime and frame ownership are explicit."""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import importlib
from ipaddress import IPv4Address
import os
from pathlib import Path
import sys
import threading
import time
import warnings

import numpy as np

from .session import utc_now


_dll_directories = []


def load_sdk():
    if sys.platform != "win32":
        raise RuntimeError("Live HIK capture requires Windows and the MVS SDK; offline commands remain available")
    roots = [os.environ.get("MVS_SDK_PATH"), os.environ.get("MVCAM_COMMON_RUNENV"),
             r"C:\Program Files (x86)\MVS\Development", r"C:\Program Files\MVS\Development"]
    candidates = [p for root in roots if root for p in
                  (Path(root), Path(root) / "Samples/Python/MvImport")]
    binding = next((p for p in candidates if (p / "MvCameraControl_class.py").exists()), None)
    if binding is None:
        raise RuntimeError("MVS Python bindings not found. Install MVS Development or set MVS_SDK_PATH")
    bits = "Win64_x64" if ct.sizeof(ct.c_void_p) == 8 else "Win32_i86"
    runtime_candidates = [os.environ.get("MVS_DLL_PATH"),
                          str(Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime") / bits),
                          str(Path(r"C:\Program Files\Common Files\MVS\Runtime") / bits)]
    for folder in runtime_candidates:
        if folder and (Path(folder) / "MvCameraControl.dll").exists():
            _dll_directories.append(os.add_dll_directory(folder))
            # The installed vendor binding loads with winmode=0, which uses PATH.
            if folder not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            break
    if str(binding) not in sys.path:
        sys.path.insert(0, str(binding))
    try:
        return importlib.import_module("MvCameraControl_class")
    except (ImportError, OSError) as error:
        raise RuntimeError(f"Cannot load MVS SDK; check DLL path and Python/SDK bitness: {error}") from error


def check(code: int, operation: str):
    if code != 0:
        raise RuntimeError(f"MVS {operation} failed: 0x{code & 0xffffffff:08x}")


def decode(value) -> str:
    raw = bytes(value).split(b"\0", 1)[0]
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("latin1")


class Runtime:
    def __init__(self, sdk=None):
        self.sdk = sdk if sdk is not None else load_sdk()
        self.active = False

    def __enter__(self):
        check(self.sdk.MvCamera.MV_CC_Initialize(), "initialize")
        self.active = True
        return self

    def __exit__(self, *_):
        if self.active:
            code = self.sdk.MvCamera.MV_CC_Finalize()
            self.active = False
            if code:
                warnings.warn(f"MVS finalize returned 0x{code & 0xffffffff:08x}")

    def devices(self, *, include_gige: bool = True) -> list[dict]:
        """Enumerate USB and GigE cameras, with optional USB-only filtering."""
        device_list = self.sdk.MV_CC_DEVICE_INFO_LIST()
        layers = self.sdk.MV_USB_DEVICE
        if include_gige:
            layers |= self.sdk.MV_GIGE_DEVICE
        check(self.sdk.MvCamera.MV_CC_EnumDevices(layers, device_list),
              "enumerate USB/GigE devices" if include_gige else "enumerate USB devices")
        result = []
        for i in range(device_list.nDeviceNum):
            pointer = ct.cast(device_list.pDeviceInfo[i], ct.POINTER(self.sdk.MV_CC_DEVICE_INFO))
            info = self.sdk.MV_CC_DEVICE_INFO.from_buffer_copy(pointer.contents)
            network = {}
            if info.nTLayerType == self.sdk.MV_USB_DEVICE:
                details = info.SpecialInfo.stUsb3VInfo
                transport = "usb"
            elif include_gige and info.nTLayerType == self.sdk.MV_GIGE_DEVICE:
                details = info.SpecialInfo.stGigEInfo
                transport = "gige"
                network = {key: str(IPv4Address(getattr(details, field))) for key, field in (
                    ("ip", "nCurrentIp"), ("subnet_mask", "nCurrentSubNetMask"),
                    ("gateway", "nDefultGateWay"), ("interface_ip", "nNetExport"))}
            else:
                continue
            result.append({"serial": decode(details.chSerialNumber), "model": decode(details.chModelName),
                           "transport": transport, **network, "_info": info})
        return result


@dataclass
class Frame:
    image: np.ndarray
    number: int
    metadata: dict


class HikCamera:
    def __init__(self, runtime: Runtime, serial: str | None, exposure_us=None, gain_db=None, gain_auto="off", resolution="current", lens_note=""):
        if gain_auto not in ("off", "continuous") or (gain_auto == "continuous" and gain_db is not None):
            raise ValueError("Use either continuous automatic gain or fixed manual gain")
        self.sdk = runtime.sdk
        self.cam = self.sdk.MvCamera()
        self.created = self.opened = self.grabbing = False
        self._conversion_buffer = None
        self._transport_original = {}
        self.metadata = {}
        try:
            devices = runtime.devices()
            matches = [d for d in devices if serial is None or d["serial"] == serial]
            if len(matches) != 1:
                raise ValueError(f"Expected one selected USB/GigE camera, got {len(matches)}. Use devices and --serial")
            device = matches[0]
            transport = device.get("transport", "usb")
            check(self.cam.MV_CC_CreateHandle(device["_info"]), "create handle")
            self.created = True
            check(self.cam.MV_CC_OpenDevice(self.sdk.MV_ACCESS_Exclusive, 0), "open (close MVS preview if occupied)")
            self.opened = True
            transport_settings = self.configure_gige() if transport == "gige" else {}
            self._geometry_original = {}
            self._geometry_notes = []
            if resolution not in ("max", "current"):
                raise ValueError("Resolution must be max or current")
            if resolution == "max":
                self.configure_max_resolution()
            self._original = {}
            for node in ("TriggerMode", "ExposureAuto", "GainAuto"):
                self._original[node] = self.read_node(node, "enum")
            for node in ("ExposureTime", "Gain"):
                self._original[node] = self.read_node(node, "float")
            check(self.cam.MV_CC_SetEnumValue("TriggerMode", 0), "disable trigger")
            controls = [("ExposureAuto", "ExposureTime", exposure_us)]
            if gain_auto == "off":
                controls.append(("GainAuto", "Gain", gain_db))
            for auto, manual, value in controls:
                check(self.cam.MV_CC_SetEnumValue(auto, 0), f"lock {auto}")
                check(self.cam.MV_CC_SetFloatValue(manual, self._original[manual] if value is None else value), f"set {manual}")
                if self.read_node(auto, "enum") != 0:
                    raise RuntimeError(f"Failed to lock {auto}")
            if gain_auto == "continuous":
                check(self.cam.MV_CC_SetEnumValueByString("GainAuto", "Continuous"), "enable continuous automatic gain")
                if self.read_node("GainAuto", "enum") != self.sdk.MV_GAIN_MODE_CONTINUOUS:
                    raise RuntimeError("Failed to enable continuous automatic gain")
            nodes = {node: self.read_node(node, "int") for node in ("Width", "Height", "OffsetX", "OffsetY")}
            nodes.update({node: self.read_node(node, "float") for node in ("ExposureTime", "Gain")})
            nodes["PixelFormat"] = self.read_node("PixelFormat", "enum")
            nodes["ExposureAuto"] = self.read_node("ExposureAuto", "enum")
            nodes["GainAuto"] = self.read_node("GainAuto", "enum")
            for node in ("BinningHorizontal", "BinningVertical", "DecimationHorizontal", "DecimationVertical"):
                nodes[node] = self.read_node(node, "int", optional=True)
            self.metadata = {"serial": device["serial"], "model": device["model"], "source": f"hik_{transport}",
                             "transport": transport, "transport_settings": transport_settings,
                             "sdk_version": int(self.sdk.MvCamera.MV_CC_GetSDKVersion()), "settings": nodes,
                             "gain_auto_mode": gain_auto,
                             "resolution_policy": resolution, "geometry_notes": self._geometry_notes,
                             "lens_note": lens_note,
                             "pixel_storage": "lossless PNG after SDK conversion to Mono8/BGR8; no geometric correction"}
            if transport == "gige":
                self.metadata["network"] = {key: device[key] for key in
                    ("ip", "subnet_mask", "gateway", "interface_ip") if key in device}
            check(self.cam.MV_CC_StartGrabbing(), "start grabbing")
            self.grabbing = True
        except BaseException:
            self.close()
            raise

    def read_node(self, node: str, kind: str, optional: bool = False):
        structs = {"enum": ("MVCC_ENUMVALUE", "MV_CC_GetEnumValue", "nCurValue"),
                   "float": ("MVCC_FLOATVALUE", "MV_CC_GetFloatValue", "fCurValue"),
                   "int": ("MVCC_INTVALUE_EX", "MV_CC_GetIntValueEx", "nCurValue")}
        struct, method, attribute = structs[kind]
        value = getattr(self.sdk, struct)()
        code = getattr(self.cam, method)(node, value)
        if code and optional:
            return None
        check(code, f"read {node}")
        return getattr(value, attribute)

    def int_limits(self, node, optional=False):
        value = self.sdk.MVCC_INTVALUE_EX()
        code = self.cam.MV_CC_GetIntValueEx(node, value)
        if code and optional:
            return None
        check(code, f"read {node} bounds")
        return value

    def set_int(self, node, value):
        if self.read_node(node, "int") == value:
            return
        check(self.cam.MV_CC_SetIntValueEx(node, int(value)), f"set {node}={value}")
        if self.read_node(node, "int") != value:
            raise RuntimeError(f"Camera did not apply {node}={value}")

    def configure_gige(self):
        node = "GevSCPSPacketSize"
        bounds = self.int_limits(node)
        self._transport_original[node] = int(bounds.nCurValue)

        def supported(size):
            # Some SDK bindings return unsigned error codes instead of a negative size.
            return (0 < size <= 65535 and bounds.nMin <= size <= bounds.nMax
                    and (size - bounds.nMin) % max(1, int(bounds.nInc)) == 0)

        packet_size = int(self.cam.MV_CC_GetOptimalPacketSize())
        if not supported(packet_size):
            warnings.warn(f"MVS optimal GigE packet size unavailable ({packet_size:#x}); trying 1500 bytes")
            packet_size = 1500
        if not supported(packet_size):
            raise RuntimeError("Camera does not support the fallback GigE packet size of 1500 bytes")
        try:
            self.set_int(node, packet_size)
        except RuntimeError:
            if packet_size == 1500 or not supported(1500):
                raise
            warnings.warn(f"Cannot apply GigE packet size {packet_size}; trying 1500 bytes")
            self.set_int(node, 1500)
        return {node: self.read_node(node, "int")}

    def configure_max_resolution(self):
        # Snapshot all geometry before the first change, so partial failures restore it.
        optional = ("BinningHorizontal", "BinningVertical", "DecimationHorizontal", "DecimationVertical")
        for node in ("Width", "Height", "OffsetX", "OffsetY", *optional):
            value = self.int_limits(node, optional=node in optional)
            if value is None:
                self._geometry_notes.append(f"{node}: not exposed by device")
            else:
                self._geometry_original[node] = int(value.nCurValue)
        for node in ("OffsetX", "OffsetY"):
            self.set_int(node, 0)
        for node in ("Width", "Height"):
            self.set_int(node, self.int_limits(node).nMin)
        for node in optional:
            if node in self._geometry_original and self.read_node(node, "int") != 1:
                self.set_int(node, 1)
        for node in ("Width", "Height"):
            bounds = self.int_limits(node)
            increment = max(1, int(bounds.nInc))
            maximum = int(bounds.nMin + (bounds.nMax-bounds.nMin)//increment*increment)
            self.set_int(node, maximum)
        for node in ("Width", "Height"):
            bounds = self.int_limits(node)
            maximum = bounds.nMin + (bounds.nMax-bounds.nMin)//max(1, bounds.nInc)*max(1, bounds.nInc)
            if bounds.nCurValue != maximum:
                raise RuntimeError(f"Maximum resolution readback failed for {node}")

    def restore_geometry(self, errors):
        original = getattr(self, "_geometry_original", {})
        if not original:
            return
        def attempt(node, value):
            try:
                if self.read_node(node, "int") == value:
                    return
            except RuntimeError:
                pass
            code = self.cam.MV_CC_SetIntValueEx(node, int(value))
            if code:
                errors.append(f"restore {node}: {code:#x}")
        for node in ("OffsetX", "OffsetY"):
            if node in original:
                attempt(node, 0)
        for node in ("Width", "Height"):
            if node in original:
                try:
                    attempt(node, self.int_limits(node).nMin)
                except RuntimeError as error:
                    errors.append(str(error))
        for node in ("BinningHorizontal", "BinningVertical", "DecimationHorizontal", "DecimationVertical",
                     "Width", "Height", "OffsetX", "OffsetY"):
            if node in original:
                attempt(node, original[node])

    def _convert(self, frame) -> np.ndarray:
        info = frame.stFrameInfo
        width, height, pixel_type = int(info.nWidth), int(info.nHeight), int(info.enPixelType)
        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid SDK frame dimensions")
        if pixel_type == self.sdk.PixelType_Gvsp_Mono8:
            if info.nFrameLen < width * height:
                raise RuntimeError("Incomplete Mono8 frame")
            return np.ctypeslib.as_array(frame.pBufAddr, shape=(height * width,)).reshape(height, width).copy()
        mono_types = {getattr(self.sdk, n) for n in dir(self.sdk) if n.startswith("PixelType_Gvsp_Mono")}
        channels = 1 if pixel_type in mono_types else 3
        length = width * height * channels
        if self._conversion_buffer is None or len(self._conversion_buffer) != length:
            self._conversion_buffer = (ct.c_ubyte * length)()
        params = self.sdk.MV_CC_PIXEL_CONVERT_PARAM_EX()
        params.nWidth, params.nHeight = width, height
        params.pSrcData, params.nSrcDataLen = frame.pBufAddr, info.nFrameLen
        params.enSrcPixelType = pixel_type
        params.enDstPixelType = self.sdk.PixelType_Gvsp_Mono8 if channels == 1 else self.sdk.PixelType_Gvsp_BGR8_Packed
        params.pDstBuffer, params.nDstBufferSize = self._conversion_buffer, length
        check(self.cam.MV_CC_ConvertPixelTypeEx(params), "convert pixel format")
        if params.nDstLen != length:
            raise RuntimeError("Incomplete converted frame")
        shape = (height, width) if channels == 1 else (height, width, 3)
        return np.ctypeslib.as_array(self._conversion_buffer).reshape(shape).copy()

    def read(self, timeout_ms: int = 500) -> Frame:
        output = self.sdk.MV_FRAME_OUT()
        code = self.cam.MV_CC_GetImageBuffer(output, timeout_ms)
        if code & 0xffffffff == self.sdk.MV_E_NODATA:
            raise TimeoutError("Camera frame timeout")
        check(code, "get image buffer")
        try:
            if not output.pBufAddr:
                raise RuntimeError("SDK returned an empty frame")
            info = output.stFrameInfo
            image = self._convert(output)
            settings = self.metadata.get("settings")
            if settings and (image.shape[1] != settings["Width"] or image.shape[0] != settings["Height"] or info.enPixelType != settings["PixelFormat"]):
                raise RuntimeError("Camera geometry/pixel format changed during capture; start a new session")
            controls = {}
            if settings:
                exposure = getattr(info, "fExposureTime", 0)
                gain = getattr(info, "fGain", float("nan"))
                if np.isfinite(exposure) and exposure > 0 and np.isfinite(gain) and gain >= 0:
                    controls = {"exposure_us": float(exposure), "gain_db": float(gain), "controls_source": "sdk_frame_info"}
                else:
                    gain = self.read_node("Gain", "float") if self.metadata.get("gain_auto_mode") == "continuous" else settings["Gain"]
                    controls = {"exposure_us": settings["ExposureTime"], "gain_db": float(gain),
                                "controls_source": "node_readback_after_frame" if self.metadata.get("gain_auto_mode") == "continuous" else "locked_settings"}
                controls["gain_auto"] = self.metadata.get("gain_auto_mode", "off")
            return Frame(image, int(info.nFrameNum), {"frame_number": int(info.nFrameNum),
                         "device_timestamp_ticks": (int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow),
                         "received_utc": utc_now(), "pixel_type": int(info.enPixelType), **controls})
        finally:
            check(self.cam.MV_CC_FreeImageBuffer(output), "free image buffer")

    def close(self):
        errors = []
        if self.grabbing:
            code = self.cam.MV_CC_StopGrabbing()
            self.grabbing = False
            if code:
                errors.append(f"stop: {code:#x}")
        if self.opened:
            self.restore_geometry(errors)
            # Restore temporary acquisition controls; no persistent camera UserSet is written.
            original = getattr(self, "_original", {})
            for node in ("ExposureAuto", "GainAuto"):
                if node in original:
                    code = self.cam.MV_CC_SetEnumValue(node, 0)
                    if code:
                        errors.append(f"disable {node} for restore: {code:#x}")
            for node in ("ExposureTime", "Gain", "ExposureAuto", "GainAuto", "TriggerMode"):
                if node in original:
                    method = self.cam.MV_CC_SetFloatValue if node in ("ExposureTime", "Gain") else self.cam.MV_CC_SetEnumValue
                    code = method(node, original[node])
                    if code:
                        errors.append(f"restore {node}: {code:#x}")
            for node, value in getattr(self, "_transport_original", {}).items():
                code = self.cam.MV_CC_SetIntValueEx(node, value)
                if code:
                    errors.append(f"restore {node}: {code:#x}")
            code = self.cam.MV_CC_CloseDevice()
            self.opened = False
            if code:
                errors.append(f"close: {code:#x}")
        if self.created:
            code = self.cam.MV_CC_DestroyHandle()
            self.created = False
            if code:
                errors.append(f"destroy: {code:#x}")
        if errors:
            warnings.warn("MVS cleanup: " + "; ".join(errors))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class LatestFrame:
    """One producer, one latest-frame slot; no unbounded image queue."""
    def __init__(self, camera: HikCamera):
        self.camera = camera
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame = None
        self._error = None
        self._thread = threading.Thread(target=self._run, name="hik-capture", daemon=True)

    def _run(self):
        last_success = time.monotonic()
        while not self._stop.is_set():
            try:
                frame = self.camera.read()
                last_success = time.monotonic()
                with self._lock:
                    self._frame = frame
            except TimeoutError:
                if time.monotonic() - last_success > 5:
                    self._error = RuntimeError("No frames for five seconds; check camera power, USB/network connection, IP configuration, exposure and camera state")
                    return
            except Exception as error:
                self._error = error
                return

    def latest(self) -> Frame | None:
        if self._error is not None:
            raise self._error
        with self._lock:
            return self._frame

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()  # read() has a bounded SDK timeout; release camera only after this.
