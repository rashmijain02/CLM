# CLMClient example project

This project demonstrates the `clm.CLMClient` API with synthetic customer
support, fraud-triage, incident-response, and candidate-ranking scenarios.

It supports two local modes:

1. **GPU-free mock mode** on native Windows. This verifies installation,
   request formatting, API connectivity, and response handling. Mock
   probabilities are lexical test data and are not real CLM predictions.
2. **Real CLM mode** in WSL2/Linux with an NVIDIA GPU. This runs Qwen3-8B
   through vLLM and the released CLM projection head.

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete installation and
troubleshooting instructions.

## Automated setup skill

The reusable agent skill is in
[`clm-local-setup`](.github/skills/clm-local-setup/SKILL.md). The independently
runnable automation is in
[`scripts/clm-local-setup`](scripts/clm-local-setup/README.md), so no skill or
agent runtime is required to install or test CLM.

Windows client:

```powershell
.\scripts\clm-local-setup\setup-windows.ps1 -Mode Client
```

Windows with a WSL2 real server:

```powershell
.\scripts\clm-local-setup\setup-windows.ps1 -Mode WSL -Distribution Ubuntu-24.04
```

Native Linux:

```bash
bash scripts/clm-local-setup/setup-linux.sh
```

## Quick start: Windows mock mode

Install Python 3.12 if it is not already available:

```powershell
winget install --id Python.Python.3.12 --exact --scope user
```

Close and reopen PowerShell, then run:

```powershell
cd H:\masterswork\gitrash\CLM\CLMClient
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Start the mock server:

```powershell
.\.venv\Scripts\python.exe ..\tools\playground_mock.py
```

In a second PowerShell window:

```powershell
cd H:\masterswork\gitrash\CLM\CLMClient
.\.venv\Scripts\python.exe .\main.py --scenario all
.\.venv\Scripts\python.exe .\main.py --scenario rank
```

The playground is available at <http://127.0.0.1:8700/>.

## Commands

Run one typed scenario:

```powershell
.\.venv\Scripts\python.exe .\main.py --scenario fraud-triage
```

Return JSON:

```powershell
.\.venv\Scripts\python.exe .\main.py --scenario incident-response --json
```

Use another endpoint or model:

```powershell
.\.venv\Scripts\python.exe .\main.py `
  --base-url http://localhost:8700 `
  --model clm-latest `
  --scenario customer-support
```

Run the tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Evaluation notebooks

- [`notebooks/email_classification_three_way_comparison.ipynb`](notebooks/email_classification_three_way_comparison.ipynb)
  runs the same six-class email evaluation against CLM, local Laya, and an
  optional Microsoft-Decision-1 Foundry deployment. Each backend has an
  independent resumable checkpoint, so Microsoft results can be added later.
- [`notebooks/clm_email_classification_evaluation.ipynb`](notebooks/clm_email_classification_evaluation.ipynb)
  contains the original CLM-only evaluation and fine-tuning analysis.

Install the notebook dependencies from
[`notebooks/requirements.txt`](notebooks/requirements.txt) before running them.

## Project layout

```text
CLMClient/
|-- .env.example             endpoint and model configuration
|-- setup.ps1                repeatable Windows client installation
|-- requirements.txt         lightweight client/mock dependencies
|-- scenarios.py             synthetic states, questions, and candidates
|-- main.py                  example command-line application
|-- tests/test_scenarios.py  focused client-example tests
|-- .github/skills/clm-local-setup/  agent orchestration instructions only
|-- scripts/clm-local-setup/         standalone setup, server, verification, and tests
`-- SETUP_GUIDE.md           mock and real-server setup guide
```
