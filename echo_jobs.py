"""Launch unattended local recipes; inspect status/results or request a stop.

Only the controller monitors stages. No LLM calls or active chat are involved.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

from liquid_lm.job_control import read_json, write_json
from liquid_lm.job_process import ProcessTree

ROOT = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
TERMINAL = {"completed", "deadline", "stopped", "failed"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def redact(text):
    text = text.replace(str(ROOT), "<workspace>")
    return re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+", "<user-profile>", text)


def validate_recipe(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("Recipe must be a JSON object")
    if set(recipe) - {"version", "budget_seconds", "finalization_seconds", "stop_grace_seconds", "stages"}:
        raise ValueError("Unknown recipe fields")
    if recipe.get("version") != 1:
        raise ValueError("Recipe version must be 1")
    if len(json.dumps(recipe).encode("utf-8")) > 200000:
        raise ValueError("Recipe exceeds 200000 bytes")
    recipe = json.loads(json.dumps(recipe))
    for key, default in (("budget_seconds", None), ("finalization_seconds", 120), ("stop_grace_seconds", 60)):
        value = recipe.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        recipe[key] = value
    if recipe["budget_seconds"] <= recipe["finalization_seconds"] + recipe["stop_grace_seconds"]:
        raise ValueError("Budget must exceed finalization reserve plus stop grace")
    stages = recipe.get("stages")
    if not isinstance(stages, list) or not 1 <= len(stages) <= 32:
        raise ValueError("Recipe requires 1..32 stages")
    names, finalizing = set(), False
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) - {"name", "phase", "command", "summary"}:
            raise ValueError("Invalid stage fields")
        name = stage.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,47}", name) or name.lower() in names:
            raise ValueError("Stage names must be unique portable directory names")
        if name.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1,10)), *(f"LPT{i}" for i in range(1,10))}:
            raise ValueError("Reserved Windows stage name")
        names.add(name.lower())
        phase = stage.get("phase")
        if phase not in ("train", "finalize") or (finalizing and phase == "train"):
            raise ValueError("All train stages must precede finalize stages")
        finalizing |= phase == "finalize"
        command = stage.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x or "\0" in x for x in command):
            raise ValueError("Stage command must be a nonempty argument array")
        if len(json.dumps(command)) > 32768:
            raise ValueError("Stage command is too large")
        summary = stage.get("summary", "training_progress.json")
        if not isinstance(summary, str) or not summary or Path(summary).is_absolute() or Path(summary).drive or ".." in Path(summary).parts:
            raise ValueError("Summary must be a relative path inside the stage")
        stage["summary"] = summary
    if len(json.dumps(recipe, ensure_ascii=False, indent=2).encode("utf-8")) > 200000:
        raise ValueError("Normalized recipe exceeds 200000 bytes")
    return recipe


def job_directory(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if resolved == ROOT or not resolved.is_relative_to(ROOT):
        raise ValueError("Job directories must stay inside the workspace")
    # Reject junctions as well as symlinks before writing through an ancestor.
    for ancestor in (path, *path.parents):
        if ancestor == ROOT.parent:
            break
        if ancestor.exists() and (ancestor.is_symlink() or getattr(ancestor.lstat(), "st_file_attributes", 0) & 0x400):
            raise ValueError("Job directory contains a reparse point or symlink")
    return resolved


def expand(command, directory, stage_dir):
    values = {"{python}": sys.executable, "{repo}": str(ROOT), "{run}": str(directory), "{stage}": str(stage_dir)}
    result = []
    for argument in command:
        for key, value in values.items():
            argument = argument.replace(key, value)
        result.append(argument)
    return result


def pump_output(pipe, log, errors):
    try:
        with pipe, log:
            while True:
                chunk = pipe.readline(65536)
                if not chunk:
                    break
                log.write(redact(chunk))
                log.flush()
    except Exception as exc:
        errors.append(str(exc))


def collect_summary(stage_dir, name):
    path = (stage_dir / name).resolve()
    if not path.is_relative_to(stage_dir):
        raise ValueError("Stage summary escaped its directory")
    try:
        value = read_json(path, limit=4096)
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError("Stage summary must be a JSON object")
    return json.loads(redact(json.dumps(value, ensure_ascii=False, allow_nan=False)))


def publish_result(directory, state):
    state = dict(state, finished_utc=utc_now(),
                 exit_code={"completed":0, "failed":1, "deadline":124, "stopped":130}[state["status"]])
    rows = ["# Echo unattended job", "", f"Status: {state['status']}",
            f"Reason: {state['reason']}", f"Elapsed: {state['elapsed_seconds']:.3f} s", ""]
    for stage in state["stages"]:
        rows.extend([f"- {stage['name']}: {stage['status']}; exit={stage['exit_code']}; log: {stage['log']}"])
    rows.extend(["", "Structured measurements are in result.json. Missing summaries mean no metrics were supplied.", ""])
    (directory / "report.md").write_text("\n".join(rows), encoding="utf-8")
    write_json(directory / "result.json", state)
    write_json(directory / "status.json", state)


def controller(directory):
    # A second controller cannot execute an existing recipe again.
    with (directory / "controller.lock").open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    recipe = validate_recipe(read_json(directory / "recipe.json", limit=262144))
    started = time.monotonic()
    hard_deadline = started + recipe["budget_seconds"]
    train_end = hard_deadline - recipe["finalization_seconds"]
    soft_deadline = train_end - recipe["stop_grace_seconds"]
    state = dict(status="running", reason="running", controller_pid=os.getpid(),
                 started_utc=utc_now(), elapsed_seconds=0, stages=[],
                 budget_seconds=recipe["budget_seconds"],
                 recipe_sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True).encode("utf-8")).hexdigest())
    tree = None
    deadline_reached = False
    try:
        for stage in recipe["stages"]:
            if (directory / "stop.json").exists():
                state.update(status="stopped", reason="user_stop")
                break
            now = time.monotonic()
            if stage["phase"] == "train" and (deadline_reached or now >= soft_deadline):
                deadline_reached = True
                continue
            if now >= hard_deadline:
                state.update(status="deadline", reason="total_deadline")
                break
            stage_dir = directory / stage["name"]
            stage_dir.mkdir()
            command = expand(stage["command"], directory, stage_dir)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                ECHO_JOB_CONTROL_DIR=str(stage_dir),
                TRITON_CACHE_DIR=str(directory / "cache/triton"),
                TORCHINDUCTOR_CACHE_DIR=str(directory / "cache/inductor"))
            log = (stage_dir / "output.log").open("w", encoding="utf-8")
            try:
                tree = ProcessTree([sys.executable, "-u", str(SCRIPT), "_stage", *command],
                    cwd=stage_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            process = tree.process
            errors = []
            reader = threading.Thread(target=pump_output, args=(
                io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace"), log, errors), daemon=True)
            reader.start()
            stage_start = time.monotonic()
            reason = None
            stop_started = None
            forced = False
            row = dict(name=stage["name"], phase=stage["phase"], status="running", exit_code=None,
                       log=f"{stage['name']}/output.log")
            state["stages"].append(row)
            while process.poll() is None:
                now = time.monotonic()
                # User cancellation supersedes any previously requested deadline.
                requested = "user_stop" if (directory / "stop.json").exists() else None
                if requested is None and stage["phase"] == "train" and now >= soft_deadline:
                    requested = "deadline"
                if requested and requested != reason:
                    reason = requested
                    stop_started = now if stop_started is None else stop_started
                    write_json(stage_dir / "stop.json", dict(reason=reason))
                limit = train_end if stage["phase"] == "train" else hard_deadline
                if now >= limit or (stop_started is not None and now-stop_started >= recipe["stop_grace_seconds"]):
                    reason = reason or "deadline"
                    forced = True
                    tree.close()
                    break
                state.update(active_stage=stage["name"], elapsed_seconds=now-started,
                             updated_utc=utc_now(), stop_requested=reason)
                write_json(directory / "status.json", state)
                time.sleep(.25)
            exit_code = process.wait()
            # Close ownership even after success, killing abandoned descendants.
            tree.close()
            tree = None
            reader.join(timeout=5)
            if reader.is_alive() or errors:
                raise RuntimeError("Stage log collection failed: " + "; ".join(errors))
            row.update(exit_code=exit_code, elapsed_seconds=time.monotonic()-stage_start,
                       forced=forced, summary=collect_summary(stage_dir, stage["summary"]))
            with (stage_dir / "output.log").open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell()-2048))
                row["log_tail"] = stream.read().decode("utf-8", errors="replace")
            if (directory / "stop.json").exists():
                row["status"] = "stopped"
                state.update(status="stopped", reason="user_stop")
                break
            if reason == "deadline":
                row["status"] = "deadline"
                deadline_reached = True
                if forced:
                    state.update(status="deadline", reason="forced_deadline")
                    break
                if exit_code:
                    row["status"] = "failed"
                    state.update(status="failed", reason="stage_failed_during_stop")
                    break
            elif exit_code:
                row["status"] = "failed"
                state.update(status="failed", reason="stage_failed")
                break
            else:
                row["status"] = "completed"
        else:
            state.update(status="deadline" if deadline_reached else "completed",
                         reason="training_deadline" if deadline_reached else "all_stages_completed")
    except BaseException as exc:
        state.update(status="failed", reason="controller_error", error=redact(str(exc)))
        if state["stages"] and state["stages"][-1]["status"] == "running":
            state["stages"][-1]["status"] = "failed"
    finally:
        if tree is not None:
            tree.close()
        state.update(elapsed_seconds=time.monotonic()-started)
        state["deadline_overrun_seconds"] = max(0, time.monotonic()-hard_deadline)
        state.pop("active_stage", None)
        publish_result(directory, state)
    return 1 if state["status"] == "failed" else 0


def launch(recipe_path, directory, startup_seconds=5):
    recipe = validate_recipe(read_json(recipe_path, limit=262144))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # Never overwrite or implicitly resume an existing job.
    write_json(directory / "recipe.json", recipe)
    write_json(directory / "status.json", dict(status="starting", created_utc=utc_now()))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    try:
        with (directory / "controller.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", str(SCRIPT), "_run", str(directory)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=env, close_fds=True, **kwargs)
    except OSError as exc:
        publish_result(directory, dict(status="failed", reason="launch_failed", elapsed_seconds=0,
                       error=redact(str(exc)), stages=[]))
        raise
    end = time.monotonic() + startup_seconds
    while time.monotonic() < end:
        state = read_json(directory / "status.json", limit=262144)
        if "active_stage" in state or state["status"] in TERMINAL:
            return state
        if process.poll() is not None:
            raise RuntimeError("Controller exited before startup; inspect controller.log")
        time.sleep(.1)
    # Do not leave an uncertain launch running without an explicit stop request.
    write_json(directory / "stop.json", dict(reason="user_stop", requested_utc=utc_now()))
    raise TimeoutError("Startup was not confirmed; stop requested; inspect job status")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_stage":
        if sys.stdin.buffer.readline() != b"START\n":
            return 1
        return subprocess.call(argv[1:], stdin=subprocess.DEVNULL)
    if argv and argv[0] == "_run":
        try:
            return controller(job_directory(argv[1]))
        except (OSError, ValueError, KeyError) as exc:
            print(redact(str(exc)), file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--recipe", type=Path, required=True)
    launch_parser.add_argument("--run-dir", default=None)
    for name in ("status", "result", "stop"):
        sub.add_parser(name).add_argument("run_dir")
    args = parser.parse_args(argv)
    try:
        if args.action == "launch":
            name = args.run_dir or f"runs/job_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
            directory = job_directory(name)
            state = launch(args.recipe, directory)
            print(json.dumps(dict(run_dir=directory.relative_to(ROOT).as_posix(), **state), ensure_ascii=False))
            return int(state["status"] == "failed")
        directory = job_directory(args.run_dir)
        if args.action == "stop" and not (directory / "result.json").exists():
            if not (directory / "recipe.json").is_file():
                raise ValueError("Not an Echo job directory")
            write_json(directory / "stop.json", dict(reason="user_stop", requested_utc=utc_now()))
        path = directory / ("result.json" if (directory / "result.json").exists() else "status.json")
        state = read_json(path, limit=262144)
        if state["status"] not in TERMINAL:
            if args.action == "stop":
                state["stop_requested"] = "user_stop"
            if args.action == "status" and state.get("active_stage"):
                state["progress"] = collect_summary(directory / state["active_stage"], "training_progress.json")
            updated = state.get("updated_utc", state.get("created_utc"))
            if updated:
                age = (datetime.now(timezone.utc)-datetime.fromisoformat(updated)).total_seconds()
                state["heartbeat_age_seconds"] = round(age, 1)
                state["controller_unresponsive"] = age > 15
        if args.action == "result" and state["status"] not in TERMINAL:
            print(json.dumps(dict(status="not_finished", job=state), ensure_ascii=False))
            return 2
        print(json.dumps(state, ensure_ascii=False))
        return int(state["status"] == "failed")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
