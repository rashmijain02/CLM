#!/usr/bin/env python3
"""Run typed CLM questions and candidate ranking against synthetic scenarios."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import requests

from clm import CLMClient, CLMError, SystemOneResponse
from scenarios import RANKING_SCENARIO, SCENARIOS, Scenario


def load_environment(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def run_scenario(client: CLMClient, scenario: Scenario, temperature: float) -> SystemOneResponse:
    return client.system_one(
        state=scenario.state,
        questions=scenario.questions,
        temperature=temperature,
    )


def response_payload(scenario: Scenario, response: SystemOneResponse) -> dict:
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "model": response.model,
        "latency_ms": response.latency_ms,
        "usage": asdict(response.usage),
        "answers": {question_id: asdict(answer) for question_id, answer in response.answers.items()},
    }


def print_scenario(scenario: Scenario, response: SystemOneResponse) -> None:
    print(f"\n=== {scenario.name}: {scenario.description} ===")
    print(f"model={response.model} latency_ms={response.latency_ms} input_tokens={response.usage.input_tokens}")
    for question_id, answer in response.answers.items():
        data = asdict(answer)
        details = ", ".join(f"{key}={value}" for key, value in data.items() if key != "type")
        print(f"  {question_id} [{data['type']}]: {details}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all", "rank"],
        default="all",
        help="Synthetic scenario to execute (default: all).",
    )
    parser.add_argument("--base-url", help="Override CLM_BASE_URL.")
    parser.add_argument("--model", help="Override CLM_MODEL.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main() -> int:
    client_root = Path(__file__).resolve().parent
    load_environment(client_root / ".env")
    args = build_parser().parse_args()

    client = CLMClient(
        base_url=args.base_url or os.environ.get("CLM_BASE_URL"),
        api_key=os.environ.get("CLM_API_KEY"),
        model=args.model or os.environ.get("CLM_MODEL", "clm-latest"),
    )

    try:
        if not client.health():
            print(
                f"CLM server is not healthy at {client.base_url}. "
                "Start the mock or real server described in SETUP_GUIDE.md.",
                file=sys.stderr,
            )
            return 2

        if args.scenario == "rank":
            ranked = client.rank(
                RANKING_SCENARIO.context,
                RANKING_SCENARIO.question,
                RANKING_SCENARIO.candidates,
                temperature=args.temperature,
            )
            if args.json:
                print(json.dumps({"scenario": "rank", "ranked": ranked}, indent=2))
            else:
                print("\n=== candidate ranking ===")
                for item in ranked:
                    print(f"  {item['rank']}. {item['candidate']} ({item['prob']:.4f})")
            return 0

        selected = SCENARIOS.values() if args.scenario == "all" else [SCENARIOS[args.scenario]]
        results = []
        for scenario in selected:
            response = run_scenario(client, scenario, args.temperature)
            results.append(response_payload(scenario, response))
            if not args.json:
                print_scenario(scenario, response)
        if args.json:
            print(json.dumps(results, indent=2))
        return 0
    except CLMError as error:
        print(f"CLM API error: {error}", file=sys.stderr)
        return 1
    except requests.RequestException as error:
        print(f"Could not reach the CLM server at {client.base_url}: {error}", file=sys.stderr)
        return 1
    finally:
        client._s.close()


if __name__ == "__main__":
    raise SystemExit(main())

