"""Real subprocess tests of detached execution, cancellation and checkpoint stops."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import echo_jobs as jobs
from liquid_lm.job_control import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def recipe(stages, budget=12, reserve=1, grace=.4):
    return dict(version=1, budget_seconds=budget, finalization_seconds=reserve,
                stop_grace_seconds=grace, stages=stages)


def stage(name, source, phase="train"):
    return dict(name=name, phase=phase, command=["{python}", "-c", source])


def wait_for(predicate, seconds=8):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(.04)
    raise AssertionError("Timed out waiting for a test-owned job")


def process_running(pid):
    if os.name == "nt":
        import ctypes as c
        from ctypes import wintypes as w
        api = c.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        api.OpenProcess.restype = w.HANDLE
        api.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        api.WaitForSingleObject.restype = w.DWORD
        api.CloseHandle.argtypes = [w.HANDLE]
        handle = api.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return api.WaitForSingleObject(handle, 0) == 258
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        status = Path(f"/proc/{pid}/stat")
        return not status.exists() or status.read_text().split()[2] != "Z"
    except ProcessLookupError:
        return False


class UnattendedJobTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "runs/unattended_controller_20261021/scratch"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="job_test_", dir=parent)
        self.directory = Path(self.temp.name)
        self.jobs = []

    def tearDown(self):
        try:
            for directory in self.jobs:
                if not (directory / "result.json").exists():
                    write_json(directory / "stop.json", dict(reason="user_stop"))
                    wait_for(lambda: (directory / "result.json").exists(), seconds=8)
                pid = read_json(directory / "result.json", limit=262144).get("controller_pid")
                if pid is not None:
                    wait_for(lambda: not process_running(pid))
        finally:
            self.temp.cleanup()

    def command(self, *arguments, timeout=10):
        return subprocess.run([sys.executable, str(ROOT / "echo_jobs.py"), *map(str, arguments)],
            cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", timeout=timeout)

    def launch(self, value):
        directory = self.directory / f"run_{len(self.jobs)}"
        path = self.directory / f"recipe_{len(self.jobs)}.json"
        write_json(path, value)
        self.jobs.append(directory)
        result = self.command("launch", "--recipe", path, "--run-dir", directory)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return directory, json.loads(result.stdout)

    def result(self, directory):
        wait_for(lambda: (directory / "result.json").exists())
        return read_json(directory / "result.json", limit=262144)

    def test_recipe_and_directory_guards_reject_ambiguous_jobs(self):
        good = recipe([stage("train", "print('ok')")])
        self.assertEqual(jobs.validate_recipe(good)["version"], 1)
        with self.assertRaises(ValueError):
            jobs.validate_recipe([])
        invalid_summary = self.directory / "summary"
        invalid_summary.mkdir()
        (invalid_summary / "metrics.json").write_text('{"loss":NaN}', encoding="utf-8")
        with self.assertRaises(ValueError):
            jobs.collect_summary(invalid_summary, "metrics.json")
        for field, value in (("budget_seconds", float("nan")), ("stop_grace_seconds", True),
                             ("finalization_seconds", -1), ("budget_seconds", 1)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                jobs.validate_recipe(dict(good, **{field:value}))
        for stages in ([stage("../old", "pass")], [stage("NUL", "pass")],
                       [stage("A", "pass"), stage("a", "pass")],
                       [stage("sample", "pass", "finalize"), stage("train", "pass")]):
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                jobs.validate_recipe(recipe(stages))
        with self.assertRaises(ValueError):
            jobs.job_directory(ROOT.parent / "outside")
        self.assertEqual(jobs.expand(["{python}", "print({'a': 1})", "{run}/x"], self.directory, self.directory),
                         [sys.executable, "print({'a': 1})", str(self.directory)+"/x"])

    def test_launcher_exits_while_job_runs_then_collects_sequential_results(self):
        source = ("from pathlib import Path; import time, json; "
                  "Path('ready').write_text('ready'); "
                  "exec(\"while not Path('finish').exists(): time.sleep(.02)\"); "
                  "Path('training_progress.json').write_text(json.dumps({'validation_loss':1.25}))")
        directory, initial = self.launch(recipe([stage("train", source),
            stage("sample", "from pathlib import Path; Path('generated.txt').write_text('fixed sample')", "finalize")]))
        self.assertEqual(initial["status"], "running")
        wait_for(lambda: (directory / "train/ready").exists())
        self.assertFalse((directory / "result.json").exists())
        duplicate = self.command("_run", directory)
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertFalse((directory / "result.json").exists())
        unfinished = self.command("result", directory)
        self.assertEqual(unfinished.returncode, 2)
        self.assertEqual(json.loads(unfinished.stdout)["status"], "not_finished")
        (directory / "train/finish").touch()
        result = self.result(directory)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([x["status"] for x in result["stages"]], ["completed", "completed"])
        self.assertEqual(result["stages"][0]["summary"]["validation_loss"], 1.25)
        self.assertEqual((directory / "sample/generated.txt").read_text(), "fixed sample")
        self.assertTrue((directory / "report.md").is_file())
        collected = self.command("result", directory)
        self.assertEqual(collected.returncode, 0)
        self.assertEqual(json.loads(collected.stdout)["status"], "completed")
        again = self.command("launch", "--recipe", self.directory / "recipe_0.json", "--run-dir", directory)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(read_json(directory / "result.json"), result)

    def test_failure_preserves_artifacts_and_does_not_run_queued_stage(self):
        directory, _ = self.launch(recipe([stage("train", "import time; time.sleep(.4); "
            "from pathlib import Path; Path('best.pt').write_bytes(b'checkpoint'); "
            "print(Path.cwd()); raise SystemExit(7)"), stage("sample", "raise AssertionError('queued')", "finalize")]))
        result = self.result(directory)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"][0]["exit_code"], 7)
        self.assertEqual((directory / "train/best.pt").read_bytes(), b"checkpoint")
        self.assertFalse((directory / "sample").exists())
        self.assertNotIn(str(ROOT), (directory / "train/output.log").read_text())

    def test_stop_terminates_descendants_and_cancels_finalization(self):
        source = ("import os, json, subprocess, sys, time; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "Path('pids.json').write_text(json.dumps([os.getpid(),child.pid])); time.sleep(60)")
        directory, _ = self.launch(recipe([stage("train", source), stage("sample", "pass", "finalize")]))
        wait_for(lambda: (directory / "train/pids.json").exists())
        pids = read_json(directory / "train/pids.json")
        self.assertTrue(all(process_running(pid) for pid in pids))
        stopped = self.command("stop", directory)
        self.assertEqual(stopped.returncode, 0)
        result = self.result(directory)
        self.assertEqual(result["status"], "stopped")
        self.assertTrue(result["stages"][0]["forced"])
        self.assertFalse((directory / "sample").exists())
        wait_for(lambda: all(not process_running(pid) for pid in pids))

    def test_cooperative_deadline_finishes_reporting_without_manual_marker(self):
        source = ("from pathlib import Path; import time,json; "
            "exec(\"while not Path('stop.json').exists(): time.sleep(.01)\"); "
            "Path('training_progress.json').write_text(json.dumps({'checkpoint':'best.pt'}))")
        directory, _ = self.launch(recipe([stage("train", source),
            stage("sample", "from pathlib import Path; Path('done').touch()", "finalize")], budget=3, reserve=1, grace=.4))
        result = self.result(directory)
        self.assertEqual(result["status"], "deadline")
        self.assertFalse(result["stages"][0]["forced"])
        self.assertTrue((directory / "sample/done").is_file())

    def test_uncooperative_deadline_is_bounded_and_skips_finalization(self):
        directory, _ = self.launch(recipe([stage("train", "import time; time.sleep(60)"),
            stage("sample", "pass", "finalize")], budget=2, reserve=.6, grace=.3))
        result = self.result(directory)
        self.assertEqual(result["status"], "deadline")
        self.assertTrue(result["stages"][0]["forced"])
        self.assertFalse((directory / "sample").exists())
        self.assertLess(result["elapsed_seconds"], 6)

    def test_failure_during_cooperative_deadline_is_not_hidden(self):
        source = ("from pathlib import Path; import time; "
            "exec(\"while not Path('stop.json').exists(): time.sleep(.01)\"); raise SystemExit(9)")
        directory, _ = self.launch(recipe([stage("train", source), stage("sample", "pass", "finalize")],
                                         budget=2, reserve=.6, grace=.3))
        result = self.result(directory)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["stages"][0]["exit_code"], 9)
        self.assertFalse((directory / "sample").exists())

    def test_finalization_is_bounded_by_total_budget(self):
        directory, _ = self.launch(recipe([stage("sample", "import time; time.sleep(60)", "finalize")],
                                         budget=2, reserve=.6, grace=.3))
        result = self.result(directory)
        self.assertEqual(result["status"], "deadline")
        self.assertEqual(result["exit_code"], 124)
        self.assertTrue(result["stages"][0]["forced"])
        self.assertLess(result["elapsed_seconds"], 6)

    def test_native_cpu_training_saves_last_before_cooperative_stop(self):
        work = self.directory / "native"
        work.mkdir()
        corpus = self.directory / "data"
        corpus.mkdir()
        (corpus / "tiny.txt").write_text("A small dog ran home. "*200, encoding="utf-8")
        write_json(work / "stop.json", dict(reason="deadline"))
        command = [sys.executable, str(ROOT / "liquid_text_model.py"), "--data", str(corpus),
            "--device", "cpu", "--tokenizer", "char", "--min-chars", "1",
            "--embedding-dim", "8", "--hidden-size", "8", "--num-layers", "1",
            "--batch-size", "2", "--seq-len", "8", "--state-carry", "1",
            "--epochs", "3", "--steps-per-epoch", "1", "--eval-batches", "1",
            "--grad-accumulation", "1", "--amp", "off", "--generate", "0",
            "--checkpoint", "best.pt", "--last-checkpoint", "last.pt", "--save-every", "99"]
        output = subprocess.run(command, cwd=work, env=dict(os.environ,
            ECHO_JOB_CONTROL_DIR=str(work), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1"),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", timeout=35)
        self.assertEqual(output.returncode, 0, output.stdout[-3000:])
        summary = read_json(work / "training_progress.json")
        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["stop_reason"], "deadline")
        self.assertEqual(summary["epoch"], 1)
        self.assertEqual(summary["processed_tokens"], 16)
        self.assertTrue((work / summary["best_checkpoint"]).is_file())
        self.assertTrue((work / summary["last_checkpoint"]).is_file())
        self.assertFalse((work / "generated.txt").exists())


if __name__ == "__main__":
    unittest.main()
