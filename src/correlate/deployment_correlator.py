"""
src/correlate/deployment_correlator.py
Links detected anomalies to recent Salesforce deployments via GitHub API.
This is the "wow moment" — showing that a bad commit caused the incident.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()


# ── Data model ────────────────────────────────────────────────────────────────

class DeploymentEvent:
    def __init__(self, commit_sha: str, commit_message: str, author: str,
                 deployed_at: str, workflow_run_url: str, branch: str):
        self.commit_sha       = commit_sha
        self.commit_message   = commit_message
        self.author           = author
        self.deployed_at      = deployed_at
        self.workflow_run_url = workflow_run_url
        self.branch           = branch

    def to_dict(self):
        return self.__dict__


class CorrelationResult:
    def __init__(self, anomaly_timestamp: str, matched_deployment: Optional[DeploymentEvent],
                 time_delta_minutes: Optional[float], confidence: str, reasoning: str):
        self.anomaly_timestamp    = anomaly_timestamp
        self.matched_deployment   = matched_deployment
        self.time_delta_minutes   = time_delta_minutes
        self.confidence           = confidence  # "high" | "medium" | "low" | "none"
        self.reasoning            = reasoning

    def to_dict(self):
        return {
            "anomaly_timestamp":  self.anomaly_timestamp,
            "matched_deployment": self.matched_deployment.to_dict() if self.matched_deployment else None,
            "time_delta_minutes": self.time_delta_minutes,
            "confidence":         self.confidence,
            "reasoning":          self.reasoning,
        }


# ── Correlator ────────────────────────────────────────────────────────────────

class DeploymentCorrelator:
    """
    Fetches recent GitHub Actions workflow runs for your Salesforce DX pipeline
    and correlates them with anomaly timestamps.
    """

    CORRELATION_WINDOW_MINUTES = 60  # Look back up to 60 min before anomaly

    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner   = os.getenv("GITHUB_REPO_OWNER")
        self.repo_name    = os.getenv("GITHUB_REPO_NAME")
        self.workflow_id  = os.getenv("GITHUB_WORKFLOW_ID", "deploy.yml")

    def _gh_headers(self):
        return {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
        }

    # ── Fetch recent deployments ──────────────────────────────────────────────

    def get_recent_deployments(self, hours_back: int = 4) -> List[DeploymentEvent]:
        """
        Query GitHub Actions for successful deployment runs in the past N hours.
        """
        url = (
            f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            f"/actions/workflows/{self.workflow_id}/runs"
        )
        params = {
            "status": "success",
            "per_page": 30,
        }
        resp = requests.get(url, headers=self._gh_headers(), params=params)
        resp.raise_for_status()
        runs = resp.json().get("workflow_runs", [])

        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        deployments = []

        for run in runs:
            completed_at = run.get("updated_at", "")
            if not completed_at:
                continue
            run_time = datetime.fromisoformat(completed_at.replace("Z", ""))
            if run_time < cutoff:
                continue

            deployments.append(DeploymentEvent(
                commit_sha=run["head_sha"][:8],
                commit_message=run["head_commit"]["message"].split("\n")[0] if run.get("head_commit") else "N/A",
                author=run["head_commit"]["author"]["name"] if run.get("head_commit") else "N/A",
                deployed_at=completed_at,
                workflow_run_url=run["html_url"],
                branch=run["head_branch"],
            ))

        print(f"🔍 Found {len(deployments)} deployment(s) in last {hours_back}h")
        return deployments

    # ── Correlate ─────────────────────────────────────────────────────────────

    def correlate(self, anomaly_timestamp: str,
                  deployments: Optional[List[DeploymentEvent]] = None) -> CorrelationResult:
        """
        Find the deployment most likely to have caused an anomaly detected
        at `anomaly_timestamp` (ISO 8601 UTC string).
        """
        if deployments is None:
            deployments = self.get_recent_deployments()

        anomaly_dt = datetime.fromisoformat(anomaly_timestamp.replace("Z", ""))
        window_start = anomaly_dt - timedelta(minutes=self.CORRELATION_WINDOW_MINUTES)

        candidates = []
        for dep in deployments:
            dep_dt = datetime.fromisoformat(dep.deployed_at.replace("Z", ""))
            if window_start <= dep_dt <= anomaly_dt:
                delta = (anomaly_dt - dep_dt).total_seconds() / 60
                candidates.append((delta, dep))

        if not candidates:
            return CorrelationResult(
                anomaly_timestamp=anomaly_timestamp,
                matched_deployment=None,
                time_delta_minutes=None,
                confidence="none",
                reasoning="No deployments found in the 60-minute window before the anomaly.",
            )

        # Closest deployment = most likely culprit
        candidates.sort(key=lambda x: x[0])
        best_delta, best_dep = candidates[0]

        confidence = "high" if best_delta <= 15 else ("medium" if best_delta <= 30 else "low")

        reasoning = (
            f"Deployment of commit #{best_dep.commit_sha} "
            f"by {best_dep.author} completed {round(best_delta, 1)} minutes "
            f"before the anomaly was detected. "
            f"Branch: {best_dep.branch}. "
            f"Commit: '{best_dep.commit_message}'"
        )

        return CorrelationResult(
            anomaly_timestamp=anomaly_timestamp,
            matched_deployment=best_dep,
            time_delta_minutes=round(best_delta, 1),
            confidence=confidence,
            reasoning=reasoning,
        )

    def correlate_batch(self, anomalies: list) -> List[CorrelationResult]:
        """
        Correlate a list of Anomaly objects (from anomaly_detector.py) in one pass,
        fetching deployments once.
        """
        deployments = self.get_recent_deployments()
        results = []
        for anomaly in anomalies:
            result = self.correlate(anomaly.timestamp, deployments)
            results.append(result)
        return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    correlator = DeploymentCorrelator()

    # Simulate an anomaly 25 minutes after a fake deployment
    fake_anomaly_time = datetime.utcnow().isoformat()

    print(f"\n🔗 Correlating anomaly at: {fake_anomaly_time}")

    # For demo without live GitHub: show what a result looks like
    fake_dep = DeploymentEvent(
        commit_sha="abc1234",
        commit_message="feat: optimize ContactTrigger with bulkified SOQL",
        author="priya.sharma",
        deployed_at=(datetime.utcnow() - timedelta(minutes=22)).isoformat() + "Z",
        workflow_run_url="https://github.com/your-org/salesforce-app/actions/runs/12345",
        branch="main",
    )

    result = CorrelationResult(
        anomaly_timestamp=fake_anomaly_time,
        matched_deployment=fake_dep,
        time_delta_minutes=22.0,
        confidence="medium",
        reasoning=(
            f"Deployment of commit #{fake_dep.commit_sha} by {fake_dep.author} "
            f"completed 22.0 minutes before the anomaly. "
            f"Commit: '{fake_dep.commit_message}'"
        ),
    )

    print(f"\n📌 Correlation Result:")
    print(json.dumps(result.to_dict(), indent=2))
