"""GigE selection, packet setup and cleanup through a simulated MVS SDK."""
import contextlib
import io
import types
import unittest
from unittest.mock import patch

from intrinsic.camera import HikCamera, Runtime
from intrinsic.capture import configured_camera
from intrinsic.config import load_config
from test_devices import discovery_sdk
from test_intrinsic import auto_gain_runtime


def capture_sdk(optimal=9000, rejected_packets=(), start_error=0):
    discovery, devices, enumeration = discovery_sdk()
    base, nodes, calls = auto_gain_runtime()
    nodes["GevSCPSPacketSize"] = 2048

    class Handle(base.sdk.MvCamera):
        MV_CC_Initialize = staticmethod(discovery.MvCamera.MV_CC_Initialize)
        MV_CC_Finalize = staticmethod(discovery.MvCamera.MV_CC_Finalize)
        MV_CC_EnumDevices = staticmethod(discovery.MvCamera.MV_CC_EnumDevices)

        def MV_CC_CreateHandle(self, info):
            details = (info.SpecialInfo.stGigEInfo if info.nTLayerType == discovery.MV_GIGE_DEVICE
                       else info.SpecialInfo.stUsb3VInfo)
            calls.append(("select", bytes(details.chSerialNumber).decode()))
            return 0

        def MV_CC_OpenDevice(self, *args):
            calls.append(("open", None))
            return 0

        def MV_CC_GetOptimalPacketSize(self):
            calls.append(("probe", None))
            return optimal

        def MV_CC_GetIntValueEx(self, name, value):
            code = super().MV_CC_GetIntValueEx(name, value)
            if code:
                return code
            value.nMin, value.nMax, value.nInc = {
                "GevSCPSPacketSize": (576, 9000, 4), "Width": (8, 1280, 8),
                "Height": (8, 1024, 8), "OffsetX": (0, 512, 1), "OffsetY": (0, 512, 1),
            }.get(name, (1, 1, 1))
            return 0

        def MV_CC_SetIntValueEx(self, name, value):
            calls.append((name, value))
            if name == "GevSCPSPacketSize" and value in rejected_packets:
                return 1
            nodes[name] = value
            return 0

        def MV_CC_StartGrabbing(self):
            calls.append(("start", None))
            return start_error

        def MV_CC_StopGrabbing(self):
            calls.append(("stop", None))
            return 0

    sdk = types.SimpleNamespace(**{**vars(discovery), **vars(base.sdk), "MvCamera": Handle})
    return sdk, devices, nodes, calls, enumeration


class GigECaptureTests(unittest.TestCase):
    def test_selected_gige_sets_packet_before_streaming_and_restores_settings(self):
        sdk, _, nodes, calls, _ = capture_sdk()
        original = nodes.copy()
        with Runtime(sdk) as runtime, HikCamera(runtime, "GIGE001", resolution="max") as camera:
            self.assertEqual(camera.metadata["source"], "hik_gige")
            self.assertEqual(camera.metadata["transport"], "gige")
            self.assertEqual(camera.metadata["network"]["ip"], "192.168.1.100")
            self.assertEqual(camera.metadata["transport_settings"], {"GevSCPSPacketSize": 9000})
            self.assertEqual((nodes["Width"], nodes["Height"]), (1280, 1024))
            self.assertIn(("select", "GIGE001"), calls)
            self.assertLess(calls.index(("open", None)), calls.index(("probe", None)))
            self.assertLess(calls.index(("GevSCPSPacketSize", 9000)), calls.index(("start", None)))
        self.assertEqual(nodes, original)
        self.assertLess(calls.index(("stop", None)), calls.index(("GevSCPSPacketSize", 2048)))
        self.assertEqual(calls[-2:], [("close", None), ("destroy", None)])

    def test_capture_entry_selects_a_single_gige_camera_automatically(self):
        sdk, devices, nodes, _, enumeration = capture_sdk()
        devices[:] = [devices[1]]
        original = nodes.copy()
        stdout = io.StringIO()
        with patch("intrinsic.camera.load_sdk", return_value=sdk), contextlib.redirect_stdout(stdout):
            with configured_camera(load_config(), None, None, None, None) as (camera, config):
                self.assertEqual(camera.metadata["serial"], "GIGE001")
                self.assertEqual(config["camera"]["resolution"], "max")
        self.assertIn("GigE 192.168.1.100 via 192.168.1.10; packet size 9000 bytes", stdout.getvalue())
        self.assertEqual(enumeration, [5, 5, "finalize"])
        self.assertEqual(nodes, original)

    def test_usb_capture_keeps_its_source_without_probing_network(self):
        sdk, _, nodes, calls, _ = capture_sdk()
        original = nodes.copy()
        with Runtime(sdk) as runtime, HikCamera(runtime, "USB001") as camera:
            self.assertEqual(camera.metadata["source"], "hik_usb")
            self.assertNotIn("network", camera.metadata)
            self.assertEqual(camera.metadata["transport_settings"], {})
        self.assertNotIn(("probe", None), calls)
        self.assertEqual(nodes, original)

    def test_ambiguous_or_missing_selection_does_not_open_a_camera(self):
        for serial, message in ((None, "Multiple USB/GigE"), ("missing", "serial missing")):
            with self.subTest(serial=serial):
                sdk, _, _, calls, enumeration = capture_sdk()
                with patch("intrinsic.camera.load_sdk", return_value=sdk), self.assertRaisesRegex(ValueError, message):
                    with configured_camera(load_config(), serial, None, None, None):
                        self.fail("An ambiguous or missing device must not open")
                self.assertEqual(calls, [])
                self.assertEqual(enumeration, [5, "finalize"])

    def test_bad_packet_probe_including_unsigned_sdk_errors_falls_back(self):
        for optimal in (0, -2147483647, 0x80000001, 9001):
            with self.subTest(optimal=optimal):
                sdk, _, nodes, _, _ = capture_sdk(optimal=optimal)
                original = nodes.copy()
                with Runtime(sdk) as runtime, self.assertWarnsRegex(UserWarning, "trying 1500"):
                    with HikCamera(runtime, "GIGE001") as camera:
                        self.assertEqual(camera.metadata["transport_settings"]["GevSCPSPacketSize"], 1500)
                self.assertEqual(nodes, original)

    def test_packet_set_failure_falls_back_and_restores_original(self):
        sdk, _, nodes, calls, _ = capture_sdk(rejected_packets=(9000,))
        original = nodes.copy()
        with Runtime(sdk) as runtime, self.assertWarnsRegex(UserWarning, "Cannot apply GigE packet size 9000"):
            with HikCamera(runtime, "GIGE001"):
                self.assertEqual(nodes["GevSCPSPacketSize"], 1500)
        self.assertEqual(nodes, original)
        self.assertLess(calls.index(("GevSCPSPacketSize", 9000)), calls.index(("GevSCPSPacketSize", 1500)))

    def test_packet_setup_failure_closes_the_open_device(self):
        sdk, _, nodes, calls, _ = capture_sdk(rejected_packets=(9000, 1500))
        original = nodes.copy()
        with Runtime(sdk) as runtime, self.assertWarns(UserWarning), self.assertRaisesRegex(RuntimeError, "GevSCPSPacketSize"):
            HikCamera(runtime, "GIGE001")
        self.assertEqual(nodes, original)
        self.assertNotIn(("start", None), calls)
        self.assertEqual(calls[-2:], [("close", None), ("destroy", None)])

    def test_start_failure_restores_geometry_controls_and_packet_size(self):
        sdk, _, nodes, calls, _ = capture_sdk(start_error=1)
        original = nodes.copy()
        with Runtime(sdk) as runtime, self.assertRaisesRegex(RuntimeError, "start grabbing"):
            HikCamera(runtime, "GIGE001", resolution="max", exposure_us=5000, gain_db=0)
        self.assertEqual(nodes, original)
        self.assertEqual(calls[-2:], [("close", None), ("destroy", None)])
