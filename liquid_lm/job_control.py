"""Small file-based bridge between native training and an unattended job."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid


def write_json(path, value):
    """Publish complete snapshots; readers never observe a partial JSON file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # Windows readers/antivirus may briefly hold a file without delete sharing.
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, limit=65536):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"JSON snapshot exceeds {limit} bytes")
    return json.loads(data.decode("utf-8-sig"))


class TrainingControl:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        if self.directory != Path.cwd().resolve():
            raise ValueError("Managed training must run in its isolated stage directory")
        self.started = time.monotonic()

    @classmethod
    def from_environment(cls):
        directory = os.environ.get("ECHO_JOB_CONTROL_DIR")
        return cls(directory) if directory else None

    def check_output(self, path):
        if not Path(path).resolve().is_relative_to(self.directory):
            raise ValueError("Managed checkpoint outputs must stay inside the stage directory")

    def stop_reason(self):
        try:
            return read_json(self.directory / "stop.json")["reason"]
        except FileNotFoundError:
            return None

    def update(self, **metrics):
        write_json(self.directory / "training_progress.json",
                   dict(elapsed_seconds=time.monotonic() - self.started, **metrics))
