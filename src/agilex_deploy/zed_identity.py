"""Read the video module's identity before using its images for robot control.

Protocol reference: Stereolabs zed-open-capture, videocapture.cpp,
commit 93739154903a66c112ee52c3c6fbfd83b7218617, getSerialNumber and
ll_SPI_FlashProgramRead. The UVC SET_CUR carries a flash READ command; this
module contains no flash-programming, camera reset, or sensor-setting operation.
"""

import ctypes
import os
import time
from pathlib import Path


class _Query(ctypes.Structure):
    _fields_ = [
        ("unit", ctypes.c_uint8),
        ("selector", ctypes.c_uint8),
        ("query", ctypes.c_uint8),
        ("size", ctypes.c_uint16),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


def read_video_serial(device):
    """Read identity from this V4L2 device, not a separate HID device or USB label."""
    libc = ctypes.CDLL(None, use_errno=True)
    request = (3 << 30) | (ctypes.sizeof(_Query) << 16) | (ord("u") << 8) | 0x21
    descriptor = os.open(str(device), os.O_RDWR | os.O_NONBLOCK)

    def query(code, buffer):
        value = _Query(4, 2, code, len(buffer), buffer)
        if libc.ioctl(descriptor, request, ctypes.byref(value)) < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(device))

    try:
        length = (ctypes.c_uint8 * 2)()
        query(0x85, length)  # UVC_GET_LEN
        size = int.from_bytes(bytes(length), "little")
        if size not in (64, 384):
            raise RuntimeError(f"Unsupported ZED control size: {size}")
        for _ in range(5):
            buffer = (ctypes.c_uint8 * size)()
            buffer[0:3] = [0x51, 0xA1, 0x03]  # Vendor flash read, not program.
            buffer[5:9] = (0x18000).to_bytes(4, "big")
            buffer[9:11] = (0x9000 + 64).to_bytes(2, "big")
            buffer[11:13] = (64).to_bytes(2, "big")
            query(0x01, buffer)  # UVC_SET_CUR: issue the read request.
            time.sleep(0.002)
            query(0x81, buffer)  # UVC_GET_CUR: retrieve its result.
            digits = bytes(buffer[19:23]).hex()
            if bytes(buffer[17:19]) == b"OV" and digits.isdecimal() and int(digits) > 0:
                return str(int(digits))
        raise RuntimeError(f"No valid ZED video serial returned by {device}")
    finally:
        os.close(descriptor)


def video_candidates():
    """Do not rely on by-id: several cameras can share the OV0001 USB label."""
    candidates = []
    for entry in sorted(Path("/sys/class/video4linux").glob("video*")):
        if (
            "ZED" in (entry / "name").read_text()
            and (entry / "index").read_text().strip() == "0"
        ):
            candidates.append(Path("/dev") / entry.name)
    return candidates


def select_video_device(expected_serial, device=None):
    expected_serial = str(expected_serial)
    if not expected_serial.isdecimal() or int(expected_serial) <= 0:
        raise ValueError("A positive calibrated ZED serial is required")
    expected_serial = str(int(expected_serial))
    matches, evidence = [], []
    for candidate in [Path(device)] if device is not None else video_candidates():
        try:
            serial = read_video_serial(candidate)
        except (OSError, RuntimeError) as error:
            evidence.append(f"{candidate}: {error}")
            continue
        evidence.append(f"{candidate}: serial {serial}")
        if serial == expected_serial:
            matches.append(candidate.resolve())
    if len(matches) != 1:
        raise RuntimeError(
            f"Require exactly one ZED video device with serial {expected_serial}; "
            f"found {len(matches)}. Observed: {'; '.join(evidence) or 'no ZED devices'}"
        )
    return matches[0], expected_serial
