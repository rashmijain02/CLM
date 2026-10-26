"""Synthetic scenarios used by the CLM client example application."""
from __future__ import annotations

from dataclasses import dataclass

from clm import Choice, Noul, Question, Score


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    state: object
    questions: dict[str, Question]


@dataclass(frozen=True)
class RankingScenario:
    context: object
    question: str
    candidates: list[str]


SCENARIOS = {
    "customer-support": Scenario(
        name="customer-support",
        description="Route and prioritize a synthetic customer complaint.",
        state={
            "customer_tier": "business",
            "message": "We were charged twice for invoice INV-1042 and payroll closes in two hours.",
            "previous_contacts": 2,
            "service_status": "operational",
        },
        questions={
            "urgent": Noul(instructions="Does this case require urgent attention?"),
            "department": Choice(
                instructions="Which department should own this case?",
                criteria={
                    "billing": "Invoice charges, duplicate payments, refunds, and credits.",
                    "technical": "Product defects, outages, and integration failures.",
                    "account": "Contracts, account access, and subscription changes.",
                },
            ),
            "frustration": Score(
                instructions="Estimate the customer's frustration.",
                criteria=["Calm", "Concerned", "Frustrated", "Very angry"],
            ),
        },
    ),
    "fraud-triage": Scenario(
        name="fraud-triage",
        description="Triage a synthetic card transaction.",
        state={
            "transaction_amount_usd": 1840,
            "merchant": "Northwind Electronics",
            "country": "US",
            "cardholder_country": "US",
            "device_seen_before": False,
            "transactions_last_hour": 7,
            "cardholder_note": "I am traveling and buying a laptop.",
        },
        questions={
            "suspicious": Noul(instructions="Is this transaction suspicious enough to investigate?"),
            "next_action": Choice(
                instructions="What is the best immediate action?",
                criteria={
                    "approve": "Approve without additional verification.",
                    "verify": "Request step-up verification from the cardholder.",
                    "block": "Block the transaction and freeze the card.",
                },
            ),
            "risk": Score(
                instructions="Assign a transaction risk level.",
                criteria=["Low", "Moderate", "High", "Critical"],
            ),
        },
    ),
    "incident-response": Scenario(
        name="incident-response",
        description="Classify and route a synthetic production incident.",
        state={
            "service": "checkout-api",
            "environment": "production",
            "symptoms": "HTTP 500 rate increased from 0.2% to 18% after deployment.",
            "affected_regions": ["us-east", "eu-west"],
            "customer_impact": "Some customers cannot complete purchases.",
            "rollback_available": True,
        },
        questions={
            "customer_impacting": Noul(instructions="Is this incident actively affecting customers?"),
            "owner": Choice(
                instructions="Which team should lead the response?",
                criteria={
                    "application": "Checkout application and deployment owners.",
                    "database": "Database reliability and data platform engineers.",
                    "network": "Network, DNS, load balancing, and edge infrastructure.",
                },
            ),
            "severity": Score(
                instructions="Assign the incident severity.",
                criteria=["SEV-4 minor", "SEV-3 limited", "SEV-2 major", "SEV-1 critical"],
            ),
        },
    ),
}


RANKING_SCENARIO = RankingScenario(
    context={
        "service": "checkout-api",
        "observation": "Errors began immediately after release 2026.10.6 and rollback is available.",
    },
    question="Rank the proposed first actions by how likely they are to restore service safely.",
    candidates=[
        "Roll back release 2026.10.6 and monitor the error rate.",
        "Restart every production database at the same time.",
        "Wait one hour without making changes.",
        "Disable all customer accounts.",
    ],
)

