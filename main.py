#!/usr/bin/env python3
"""
main.py — CLI entry point for the LLM agent demo.

Usage:
    python main.py                          # run key_door scenario
    python main.py --scenario gem_collector
    python main.py --scenario maze
    python main.py --scenario all           # run all scenarios
    python main.py --scenario key_door --save-log run.json
    python main.py --quiet                  # suppress per-step output
"""

import argparse
import json
import os
import sys

from src.agent import LLMAgent
from src.scenarios import SCENARIOS, SCENARIO_MAP


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Agent in a Virtual World — Humanoid Intern Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_MAP.keys()) + ["all"],
        default="key_door",
        help="Scenario to run (default: key_door)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Anthropic model string to use",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output (only show final summary)",
    )
    parser.add_argument(
        "--save-log",
        metavar="FILE",
        help="Save full episode log as JSON to FILE",
    )
    args = parser.parse_args()

    # ── API key check ─────────────────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    agent = LLMAgent(model=args.model, verbose=not args.quiet)

    to_run = SCENARIOS if args.scenario == "all" else [SCENARIO_MAP[args.scenario]]
    all_results = []

    for scenario in to_run:
        if not args.quiet:
            print(f"\nScenario : {scenario.name}")
            print(f"Objective: {scenario.description}")

        world = scenario.build()
        result = agent.run_episode(world)
        result["scenario"] = scenario.name
        all_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│  RESULTS                                                │")
    print("├──────────────────┬───────────┬─────────────────────────┤")
    print("│  Scenario        │  Outcome  │  Steps                  │")
    print("├──────────────────┼───────────┼─────────────────────────┤")
    for r in all_results:
        status = "✓ SUCCESS" if r["success"] else "✗ FAILED "
        steps_str = f"{r['steps']} / {r['max_steps']}"
        print(f"│  {r['scenario']:16s}│  {status}  │  {steps_str:23s}│")
    print("└──────────────────┴───────────┴─────────────────────────┘")

    # ── Save log ──────────────────────────────────────────────────────────────
    if args.save_log:
        with open(args.save_log, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull episode log saved to: {args.save_log}")


if __name__ == "__main__":
    main()
