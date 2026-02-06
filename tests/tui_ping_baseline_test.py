from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_pty(master_fd: int, timeout_seconds: float) -> str:
    deadline = time.time() + timeout_seconds
    chunks: list[str] = []
    while time.time() < deadline:
        ready, _, _ = select.select([master_fd], [], [], 0.1)
        if not ready:
            continue
        try:
            data = os.read(master_fd, 8192)
        except OSError:
            break
        if not data:
            break
        chunks.append(data.decode("utf-8", errors="ignore"))
    return "".join(chunks)


class TUIPingBaselineTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_TUI_BASELINE_TEST", "").lower() in {"1", "true", "yes", "on"},
        "Set RUN_TUI_BASELINE_TEST=1 to run Textual integration baseline.",
    )
    def test_ping_has_no_traceback(self) -> None:
        env = os.environ.copy()
        env["ASSISTANT_UI"] = "textual"
        env["ASSISTANT_TEST_MODE"] = "1"
        env.setdefault("PYTHONUNBUFFERED", "1")

        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=str(PROJECT_ROOT),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
        os.close(slave_fd)

        transcript = ""
        ping_sent = False
        try:
            transcript += _read_pty(master_fd, 2.0)
            if proc.poll() is None:
                try:
                    os.write(master_fd, b"Ping\r")
                    ping_sent = True
                except OSError:
                    pass

            for _ in range(20):
                transcript += _read_pty(master_fd, 0.25)
                if proc.poll() is not None:
                    break

            transcript += _read_pty(master_fd, 0.5)
            if proc.poll() is None:
                try:
                    os.write(master_fd, b"\x03")
                except OSError:
                    pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            os.close(master_fd)

        self.assertTrue(ping_sent, "Could not send Ping to TUI session.")
        self.assertNotIn("Traceback (most recent call last)", transcript)
        self.assertNotIn("RuntimeError: cannot enter context", transcript)
        self.assertNotIn("Exception in callback", transcript)


if __name__ == "__main__":
    unittest.main()
