import argparse
from pathlib import Path

from .runner import run_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agentenv_forge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one deterministic episode")
    run.add_argument("--task", required=True)
    run.add_argument("--action", required=True, choices=("correct", "wrong"))
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trajectory = run_episode(args.task, args.action, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(trajectory.model_dump_json() + "\n")
    print(
        f"task={trajectory.task_id} action={args.action} "
        f"reward={trajectory.reward.total:.1f} output={args.output}"
    )
    return 0
