"""Linux-isolated generated Python worker; robot access is a parent-owned RPC pipe.

No credentials enter this process. Landlock restricts filesystem access and
seccomp denies networking, process creation, and cross-process access. Fail closed
if either kernel mechanism is unavailable; Python restrictions are not a sandbox.
"""

import ctypes
import errno
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import traceback

import numpy as np  # Load extension libraries before restricting the filesystem.
import cv2

from agilex_deploy.workshop_client import RobotClient, APIError

cv2.setNumThreads(1)


def isolate(workdir):
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Unsupported sandbox architecture")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise RuntimeError("Landlock ABI >=3 required")
    access = (1 << (16 if abi >= 5 else 15)) - 1

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    rules = Ruleset(access)
    fd = libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Landlock create failed")
    try:
        # The checkout itself and home directory are deliberately not allowed.
        for path, allowed in [
            (Path("/usr"), 13),
            (Path("/lib"), 13),
            (Path("/lib64"), 13),
            (Path(sys.prefix), 13),
            (Path("/dev/null"), 6),
            (Path(workdir), (access & ~((1 << 6) | (1 << 11) | (1 << 15)))),
        ]:
            if not path.exists():
                continue
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Rule(allowed, parent)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), "Landlock path rule failed")
            finally:
                os.close(parent)
        if libc.syscall(446, fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "Landlock restrict failed")
    finally:
        os.close(fd)
    seccomp = ctypes.CDLL("libseccomp.so.2")
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = seccomp.seccomp_init(0x7FFF0000)  # Default allow; deny below with EPERM.
    if not ctx:
        raise RuntimeError("seccomp initialization failed")
    try:
        for name in (
            "socket",
            "socketpair",
            "connect",
            "bind",
            "listen",
            "accept",
            "accept4",
            "execve",
            "execveat",
            "fork",
            "vfork",
            "clone",
            "clone3",
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "pidfd_getfd",
            "pidfd_open",
            "kill",
            "tkill",
            "tgkill",
            "mount",
            "umount2",
            "unshare",
            "setns",
            "bpf",
            "io_uring_setup",
            "open_by_handle_at",
            "reboot",
        ):
            nr = seccomp.seccomp_syscall_resolve_name(name.encode())
            if (
                nr >= 0
                and seccomp.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, nr, 0) != 0
            ):
                raise RuntimeError("seccomp rule failed")
        if seccomp.seccomp_load(ctx) != 0:
            raise RuntimeError("seccomp load failed")
    finally:
        seccomp.seccomp_release(ctx)
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024**2, 10 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    wire = sys.stdout

    def send(value):
        wire.write(json.dumps(value, allow_nan=False) + "\n")
        wire.flush()

    request = json.loads(sys.stdin.readline())

    class PipeRobot(RobotClient):
        def __init__(self):
            pass

        def call(self, method, **params):
            send(dict(type="call", method=method, params=params))
            result = json.loads(sys.stdin.readline())
            if "error" in result:
                raise APIError("EXECUTION_ERROR", result["error"])
            return result["result"]

    class Output(io.TextIOBase):
        def write(self, value):
            if value:
                send(dict(type="output", text=value[:8000]))
            return len(value)

        def flush(self):
            pass

    try:
        isolate(Path.cwd())
        send(dict(type="sandbox_ready"))
        sys.stdout = sys.stderr = Output()
        namespace = {"__name__": "generated_policy", "np": np, "cv2": cv2}
        exec(compile(request["code"], "<generated-policy>", "exec"), namespace)
        if not callable(namespace.get("run")):
            raise ValueError("Code must define run(robot)")
        result = namespace["run"](PipeRobot())
        send(dict(type="result", result=result))
    except BaseException:
        send(dict(type="error", error=traceback.format_exc(limit=8)))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
