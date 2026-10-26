from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

CLIENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLIENT_ROOT))

from main import response_payload, run_scenario  # noqa: E402
from scenarios import RANKING_SCENARIO, SCENARIOS  # noqa: E402
from clm import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage  # noqa: E402


class ScenarioTests(unittest.TestCase):
    def test_every_scenario_covers_all_question_types(self):
        for scenario in SCENARIOS.values():
            self.assertEqual(
                {question.to_dict()["type"] for question in scenario.questions.values()},
                {"noul", "choice", "score"},
                scenario.name,
            )

    def test_ranking_scenario_has_distinct_candidates(self):
        self.assertGreaterEqual(len(RANKING_SCENARIO.candidates), 3)
        self.assertEqual(len(RANKING_SCENARIO.candidates), len(set(RANKING_SCENARIO.candidates)))

    def test_run_scenario_passes_typed_questions_to_client(self):
        client = Mock()
        client.system_one.return_value = Mock()
        scenario = SCENARIOS["customer-support"]

        result = run_scenario(client, scenario, temperature=0.7)

        self.assertIs(result, client.system_one.return_value)
        client.system_one.assert_called_once_with(
            state=scenario.state,
            questions=scenario.questions,
            temperature=0.7,
        )

    def test_response_payload_serializes_all_answer_types(self):
        scenario = SCENARIOS["customer-support"]
        response = SystemOneResponse(
            model="test-model",
            answers={
                "urgent": NoulAnswer(0.8),
                "department": ChoiceAnswer("billing", 0.7, {"billing": 0.85, "technical": 0.15}),
                "frustration": ScoreAnswer(
                    1.5,
                    0.4,
                    {"0": 0.1, "1": 0.3, "2": 0.6},
                    {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                ),
            },
            usage=Usage(billing_units=3, input_tokens=42, output_tokens=0),
            latency_ms=12.5,
        )

        payload = response_payload(scenario, response)

        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["answers"]["urgent"]["noul"], 0.8)
        self.assertEqual(payload["answers"]["department"]["choice"], "billing")
        self.assertEqual(payload["answers"]["frustration"]["score"], 1.5)
        self.assertEqual(payload["usage"]["input_tokens"], 42)


if __name__ == "__main__":
    unittest.main()

