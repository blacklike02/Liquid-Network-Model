"""Own and terminate a stage's process tree, including orphaned descendants."""
from __future__ import annotations

import os
import signal
import subprocess


class ProcessTree:
    """A stage starts behind an stdin gate until ownership is established."""
    def __init__(self, command, **kwargs):
        self.process = None
        self.job = None
        if os.name == "nt":
            self._create_windows_job()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt", **kwargs)
            if self.job is not None:
                self._assign_windows_job()
            self.process.stdin.write(b"START\n")
            self.process.stdin.close()
        except BaseException:
            self.close()
            raise

    def _create_windows_job(self):
        import ctypes as c
        from ctypes import wintypes as w

        class Basic(c.Structure):
            _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64),
                ("flags", w.DWORD), ("min_ws", c.c_size_t), ("max_ws", c.c_size_t),
                ("active", w.DWORD), ("affinity", c.c_size_t),
                ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class IO(c.Structure):
            _fields_ = [(name, c.c_uint64) for name in ("read_ops", "write_ops", "other_ops",
                                                       "read_bytes", "write_bytes", "other_bytes")]

        class Extended(c.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", c.c_size_t),
                ("job_memory", c.c_size_t), ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]

        self.api = c.WinDLL("kernel32", use_last_error=True)
        for name, restype, argtypes in (
            ("CreateJobObjectW", w.HANDLE, [c.c_void_p, w.LPCWSTR]),
            ("SetInformationJobObject", w.BOOL, [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]),
            ("AssignProcessToJobObject", w.BOOL, [w.HANDLE, w.HANDLE]),
            ("OpenProcess", w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]),
            ("CloseHandle", w.BOOL, [w.HANDLE]),
        ):
            function = getattr(self.api, name)
            function.restype, function.argtypes = restype, argtypes
        self.job = self.api.CreateJobObjectW(None, None)
        if not self.job:
            raise c.WinError(c.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.job, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.api.CloseHandle(self.job)
            self.job = None
            raise error

    def _assign_windows_job(self):
        import ctypes as c
        handle = self.api.OpenProcess(0x0101, False, self.process.pid)  # SET_QUOTA | TERMINATE
        if not handle:
            raise c.WinError(c.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.job, handle):
                raise c.WinError(c.get_last_error())
        finally:
            self.api.CloseHandle(handle)

    def close(self):
        if self.job is not None:
            self.api.CloseHandle(self.job)
            self.job = None
        elif os.name != "nt" and self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=10)
            if self.process.stdin is not None:
                self.process.stdin.close()
