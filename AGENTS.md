# AGENTS.md

## Cursor Cloud specific instructions

This is a Python AIOps pipeline for Salesforce org monitoring. It has no automated test suite and no linter configuration.

### Running the application

- **From `src/` directory:** `cd src && python pipeline.py --dry-run` (imports use relative paths like `from ingest.salesforce_collector import ...`).
- **From workspace root with PYTHONPATH:** `PYTHONPATH=src python src/pipeline.py --dry-run`.
- Individual modules have `__main__` demo blocks and can be run standalone:
  - `PYTHONPATH=src python src/detect/anomaly_detector.py` — trains an Isolation Forest on synthetic baseline data and runs detection against sample snapshot in `data/snapshot_latest.json`.
  - `PYTHONPATH=src python src/correlate/deployment_correlator.py` — prints a demo correlation result (no live API needed).
  - `PYTHONPATH=src python src/act/notifier.py` — prints a demo Slack payload structure (no webhook needed).

### External service credentials

The full pipeline (`Ingest → Detect → Correlate → Act`) requires a `.env` file with Salesforce OAuth credentials (`SF_CLIENT_ID`, `SF_CLIENT_SECRET`, `SF_DOMAIN`). Without these, the Ingest step fails. Optional credentials: `GITHUB_TOKEN`/`GITHUB_REPO_OWNER`/`GITHUB_REPO_NAME` (correlation), `SLACK_WEBHOOK_URL` (alerts), `JIRA_BASE_URL`/`JIRA_EMAIL`/`JIRA_API_TOKEN` (ticketing).

### Key caveats

- There is no `requirements.txt` in the base repo; one was added in the dev-env setup PR. Dependencies: `requests`, `python-dotenv`, `numpy`, `scikit-learn`.
- The `src/` subdirectories have no `__init__.py` files; Python path must include `src/` for imports to resolve.
- `data/` is gitignored but `data/snapshot_latest.json` is committed as a sample fixture for local testing.
- No linter, formatter, or test framework is configured in this project.
