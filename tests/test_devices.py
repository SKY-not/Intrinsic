"""Device discovery through ctypes structures without loading the vendor DLL."""
import contextlib
import ctypes as ct
import io
import json
import types
import unittest
from unittest.mock import patch

from intrinsic.camera import Runtime
from intrinsic.cli import main, parser


def discovery_sdk(error=0):
    class UsbInfo(ct.Structure):
        _fields_ = [("chModelName", ct.c_char * 32), ("chSerialNumber", ct.c_char * 32)]

    class GigEInfo(ct.Structure):
        _fields_ = [("nCurrentIp", ct.c_uint32), ("nCurrentSubNetMask", ct.c_uint32),
                    ("nDefultGateWay", ct.c_uint32), ("chModelName", ct.c_char * 32),
                    ("chSerialNumber", ct.c_char * 32), ("nNetExport", ct.c_uint32)]

    class SpecialInfo(ct.Union):
        _fields_ = [("stUsb3VInfo", UsbInfo), ("stGigEInfo", GigEInfo)]

    class DeviceInfo(ct.Structure):
        _fields_ = [("nTLayerType", ct.c_uint32), ("SpecialInfo", SpecialInfo)]

    usb, gige, unknown = DeviceInfo(), DeviceInfo(), DeviceInfo()
    usb.nTLayerType, gige.nTLayerType, unknown.nTLayerType = 4, 1, 8
    usb.SpecialInfo.stUsb3VInfo = UsbInfo(b"MV-USB", b"USB001")
    gige.SpecialInfo.stGigEInfo = GigEInfo(
        0xC0A80164, 0xFFFFFF00, 0, b"MV-CU013-A0GC", b"GIGE001", 0xC0A8010A)
    backing = [usb, gige, unknown]
    calls = []

    def enumerate_devices(layers, output):
        calls.append(layers)
        if error:
            return error
        selected = [device for device in backing if device.nTLayerType & layers]
        output.nDeviceNum = len(selected)
        output.pDeviceInfo = [ct.pointer(device) for device in selected]
        return 0

    sdk = types.SimpleNamespace(
        MV_USB_DEVICE=4, MV_GIGE_DEVICE=1,
        MV_CC_DEVICE_INFO_LIST=types.SimpleNamespace, MV_CC_DEVICE_INFO=DeviceInfo,
        MvCamera=types.SimpleNamespace(
            MV_CC_Initialize=lambda: 0, MV_CC_Finalize=lambda: calls.append("finalize") or 0,
            MV_CC_EnumDevices=enumerate_devices))
    return sdk, backing, calls


class DeviceTests(unittest.TestCase):
    def test_mixed_devices_and_network_addresses(self):
        sdk, _, calls = discovery_sdk()
        devices = Runtime(sdk).devices(include_gige=True)
        self.assertEqual(calls, [5])
        public = [{key: value for key, value in device.items() if key != "_info"} for device in devices]
        self.assertEqual(public, [
            {"serial": "USB001", "model": "MV-USB", "transport": "usb"},
            {"serial": "GIGE001", "model": "MV-CU013-A0GC", "transport": "gige",
             "ip": "192.168.1.100", "subnet_mask": "255.255.255.0",
             "gateway": "0.0.0.0", "interface_ip": "192.168.1.10"}])

    def test_default_includes_gige_and_usb_only_filter_is_available(self):
        sdk, _, calls = discovery_sdk()
        self.assertEqual([device["serial"] for device in Runtime(sdk).devices()], ["USB001", "GIGE001"])
        self.assertEqual([device["serial"] for device in Runtime(sdk).devices(include_gige=False)], ["USB001"])
        self.assertEqual(calls, [sdk.MV_USB_DEVICE | sdk.MV_GIGE_DEVICE, sdk.MV_USB_DEVICE])

    def test_gige_only_and_info_survives_sdk_memory_changes(self):
        sdk, backing, _ = discovery_sdk()
        backing[:] = [backing[1]]
        devices = Runtime(sdk).devices(include_gige=True)
        ct.memset(ct.byref(backing[0]), 0, ct.sizeof(backing[0]))
        self.assertEqual(devices[0]["serial"], "GIGE001")
        self.assertEqual(bytes(devices[0]["_info"].SpecialInfo.stGigEInfo.chSerialNumber), b"GIGE001")

    def test_cli_prints_public_json_and_finalizes(self):
        sdk, _, calls = discovery_sdk()
        stdout = io.StringIO()
        with patch("intrinsic.camera.load_sdk", return_value=sdk), contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["devices"]), 0)
        devices = json.loads(stdout.getvalue())
        self.assertEqual(devices[1]["transport"], "gige")
        self.assertEqual(devices[1]["ip"], "192.168.1.100")
        self.assertNotIn("_info", devices[0])
        self.assertNotIn("_info", devices[1])
        self.assertEqual(calls, [5, "finalize"])

    def test_cli_empty_list_has_usb_and_gige_guidance(self):
        sdk, backing, calls = discovery_sdk()
        backing.clear()
        stdout = io.StringIO()
        with patch("intrinsic.camera.load_sdk", return_value=sdk), contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["devices"]), 0)
        self.assertTrue(stdout.getvalue().startswith("[]\n"))
        self.assertIn("USB / GigE", stdout.getvalue())
        self.assertIn("IP", stdout.getvalue())
        self.assertEqual(calls, [5, "finalize"])
        self.assertIn("GigE", parser().format_help())

    def test_cli_enumeration_error_is_reported_and_finalized(self):
        sdk, _, calls = discovery_sdk(error=0x80000001)
        stderr = io.StringIO()
        with patch("intrinsic.camera.load_sdk", return_value=sdk), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["devices"]), 1)
        self.assertIn("enumerate USB/GigE devices failed: 0x80000001", stderr.getvalue())
        self.assertEqual(calls, [5, "finalize"])
