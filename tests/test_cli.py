import json
import subprocess
import sys


def test_cli_writes_one_jsonl_trajectory(tmp_path):
    output = tmp_path / "trajectory.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "agentenv_forge",
            "run",
            "--task",
            "text-normalization-001",
            "--action",
            "correct",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["task_id"] == "text-normalization-001"
    assert record["conditions"]["action_variant"] == "correct"
    assert record["reward"]["total"] == 1.0
    assert "expected_content" not in record
