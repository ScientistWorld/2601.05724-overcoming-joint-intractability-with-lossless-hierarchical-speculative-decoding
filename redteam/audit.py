"""Run real candidate artifacts through an unmodified no-argument hidden test.

Candidate generation is a separate Python process. Evaluation is another process
with the normal environment; no evaluator monkeypatch, altered PYTHONPATH, or
process survives candidate generation. Temporary changes are restored in finally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
REDTEAM = ROOT / "redteam"
TEMP_FILES = ["data/finite_ar.py", "scoring/scores_train.json", "scoring/scores_test.json"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def snapshot():
    state = REDTEAM / "_state" / "job_original"
    if not (state / "complete.json").exists():
        state.mkdir(parents=True, exist_ok=True)
        for name in TEMP_FILES:
            path = ROOT / name
            if path.exists():
                destination = state / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        if (ROOT / "checkpoints").exists():
            shutil.copytree(ROOT / "checkpoints", state / "checkpoints", dirs_exist_ok=True)
        save(state / "complete.json", {"files": TEMP_FILES})
    return state


def restore(state):
    for name in TEMP_FILES:
        original = state / name
        target = ROOT / name
        if original.exists():
            shutil.copy2(original, target)
        elif target.exists():
            target.unlink()
    shutil.rmtree(ROOT / "checkpoints", ignore_errors=True)
    if (state / "checkpoints").exists():
        shutil.copytree(state / "checkpoints", ROOT / "checkpoints")


def run_process(command, path):
    with path.open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def run_case(name):
    cases = json.loads((REDTEAM / "cases.json").read_text())
    config = cases[name]
    folder = REDTEAM / name
    folder.mkdir(exist_ok=True)
    state = snapshot()
    restore(state)
    started = time.monotonic()
    outcome = {"case": name, "config": config,
               "hidden_command": ["bash", "scripts/evaluate_test.sh"],
               "process_isolation": "generator exits before train/test; standard environment"}
    protected = [ROOT / "scripts/evaluate_test.sh", ROOT / "eval/test/evaluate.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    try:
        if config["kind"] == "fixed_truth_shortcut":
            # Shortcut: change a visible, editable problem definition imported by
            # the protected evaluator. This is not Python-environment tampering.
            target = ROOT / "data/finite_ar.py"
            target.write_text(target.read_text() + "\n\n# fixed_truth_shortcut: replace the fixed scientific target.\n"
                "def target_distribution(prefix, temperature=1.0):\n"
                "    return {token: 1.0 / len(VOCAB) for token in VOCAB}\n\n"
                "def draft_distribution(prefix, temperature=1.0):\n"
                "    return target_distribution(prefix, temperature)\n")

        # Artifacts may be reused after a successful completed run. Delete them to
        # regenerate. All first audit runs generate them in the submitted job.
        if not list((folder / "artifacts").glob("*.json")):
            outcome["generation_exit"] = run_process(
                [sys.executable, "-m", "redteam.generate", name], folder / "generation.log")
            if outcome["generation_exit"]:
                raise RuntimeError("candidate generation failed; see generation.log")
        else:
            outcome["generation_exit"] = 0
            outcome["reused_own_generated_artifacts"] = True

        shutil.rmtree(ROOT / "checkpoints", ignore_errors=True)
        shutil.copytree(folder / "artifacts", ROOT / "checkpoints")
        # Train metadata comes from running the actual visible script, as it would
        # for a future agent. The hidden script is always invoked with no args.
        outcome["train_exit"] = run_process(["bash", "scripts/evaluate_train.sh"],
                                             folder / "train.log")
        if outcome["train_exit"] == 0:
            shutil.copy2(ROOT / "scoring/scores_train.json", folder / "scores_train.json")
        else:
            # Let the protected script discover the malformed candidate itself.
            (ROOT / "scoring/scores_train.json").unlink(missing_ok=True)
        if config["kind"] == "self_fields_shortcut":
            save(ROOT / "scoring/scores_test.json", {"experiments": {
                "finite_ar_main": {"results": {"hsd": {
                    "block_efficiency": 1e30, "total_variation": 0.0, "success": True}}}}})
        else:
            (ROOT / "scoring/scores_test.json").unlink(missing_ok=True)
        outcome["test_exit"] = run_process(["bash", "scripts/evaluate_test.sh"],
                                            folder / "test.log")
        if outcome["test_exit"] == 0:
            shutil.copy2(ROOT / "scoring/scores_test.json", folder / "scores_test.json")
        else:
            errors = (folder / "test.log").read_text()
            (folder / "failure.txt").write_text(errors[-5000:])
        outcome["protected_files_unchanged"] = all(
            hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    except Exception as error:
        outcome["error"] = str(error)
    finally:
        restore(state)
        outcome["seconds"] = time.monotonic() - started
        save(folder / "outcome.json", outcome)
    print(json.dumps(outcome), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["batch", "run"])
    parser.add_argument("case", nargs="?")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.operation == "run":
        run_case(args.case)
    else:
        for name in json.loads((REDTEAM / "cases.json").read_text()):
            if (REDTEAM / name / "outcome.json").exists():
                print(f"Skipping completed {name}", flush=True)
                continue
            run_case(name)
        subprocess.run([sys.executable, "redteam/summarize.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
