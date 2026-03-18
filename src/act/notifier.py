"""
src/act/notifier.py
Automated actions: Slack alert + Jira ticket creation when anomalies are confirmed.
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

SEVERITY_EMOJI = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🟢",
}


# ── Slack Notifier ────────────────────────────────────────────────────────────

class SlackNotifier:
    """Sends rich Slack block-kit messages to your DevOps channel."""

    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.channel     = os.getenv("SLACK_CHANNEL", "#salesforce-ops")

    def send_anomaly_alert(self, anomalies: list, correlation_result=None):
        """
        Send a formatted Slack message with anomaly + deployment correlation info.
        anomalies: list of Anomaly objects from anomaly_detector.py
        correlation_result: CorrelationResult from deployment_correlator.py
        """
        if not anomalies:
            return

        top = anomalies[0]
        emoji = SEVERITY_EMOJI.get(top.severity, "⚠️")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Salesforce Org Anomaly Detected — {top.severity.upper()}",
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Time:*\n{timestamp}"},
                    {"type": "mrkdwn", "text": f"*Anomalies Found:*\n{len(anomalies)}"},
                ]
            },
            {"type": "divider"},
        ]

        # Top anomaly details
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Top Anomaly: `{top.metric_name}`*\n"
                    f"Current: `{top.current_value}` | "
                    f"Expected: `{top.expected_range[0]} – {top.expected_range[1]}`\n"
                    f"_{top.description}_"
                )
            }
        })

        # Deployment correlation
        if correlation_result and correlation_result.matched_deployment:
            dep = correlation_result.matched_deployment
            conf_emoji = {"high": "🎯", "medium": "🔍", "low": "❓"}.get(correlation_result.confidence, "❓")
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{conf_emoji} *Likely Root Cause — Deployment Detected*\n"
                        f"• Commit: `#{dep.commit_sha}` by *{dep.author}*\n"
                        f"• Branch: `{dep.branch}`\n"
                        f"• Message: _{dep.commit_message}_\n"
                        f"• Deployed: `{dep.time_delta_minutes}` min before anomaly\n"
                        f"• Confidence: *{correlation_result.confidence.upper()}*"
                    )
                },
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Workflow Run"},
                    "url": dep.workflow_run_url,
                    "action_id": "view_run",
                }
            })

        # All anomalies summary
        if len(anomalies) > 1:
            anomaly_lines = "\n".join(
                [f"• {SEVERITY_EMOJI.get(a.severity,'⚠️')} `{a.metric_name}` — {a.current_value}" for a in anomalies[1:4]]
            )
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Other anomalies:*\n{anomaly_lines}"}
            })

        # Action buttons
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "🎫 Create Jira Ticket"},
                    "style": "primary",
                    "action_id": "create_jira",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📊 View Event Monitoring"},
                    "url": "https://your-org.salesforce.com/lightning/setup/EventMonitoring/home",
                    "action_id": "view_em",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "⏮ Suggest Rollback"},
                    "style": "danger",
                    "action_id": "suggest_rollback",
                },
            ]
        })

        payload = {"channel": self.channel, "blocks": blocks}
        resp = requests.post(self.webhook_url, json=payload)
        if resp.status_code == 200:
            print(f"✅ Slack alert sent to {self.channel}")
        else:
            print(f"❌ Slack error: {resp.status_code} — {resp.text}")
        return resp


# ── Jira Ticket Creator ───────────────────────────────────────────────────────

class JiraTicketCreator:
    """Auto-creates a Jira incident ticket with full AIOps context."""

    def __init__(self):
        self.base_url   = os.getenv("JIRA_BASE_URL")       # e.g. https://your-org.atlassian.net
        self.email      = os.getenv("JIRA_EMAIL")
        self.api_token  = os.getenv("JIRA_API_TOKEN")
        self.project    = os.getenv("JIRA_PROJECT_KEY", "OPS")
        self.issue_type = os.getenv("JIRA_ISSUE_TYPE", "Incident")

    def _auth(self):
        return (self.email, self.api_token)

    def create_incident(self, anomalies: list, correlation_result=None) -> Optional[str]:
        """
        Create a Jira incident ticket. Returns the ticket URL or None on failure.
        """
        if not anomalies:
            return None

        top = anomalies[0]
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        summary = f"[AIOps] Salesforce Org Anomaly — {top.metric_name} ({top.severity.upper()}) {timestamp}"

        description_lines = [
            "h2. 🚨 Anomaly Summary",
            f"*Detected at:* {timestamp}",
            f"*Total anomalies:* {len(anomalies)}",
            "",
            "h3. Top Anomaly",
            f"*Metric:* {top.metric_name}",
            f"*Current Value:* {top.current_value}",
            f"*Expected Range:* {top.expected_range[0]} – {top.expected_range[1]}",
            f"*Description:* {top.description}",
            "",
        ]

        if correlation_result and correlation_result.matched_deployment:
            dep = correlation_result.matched_deployment
            description_lines += [
                "h2. 🔗 Correlated Deployment",
                f"*Commit SHA:* #{dep.commit_sha}",
                f"*Author:* {dep.author}",
                f"*Branch:* {dep.branch}",
                f"*Commit Message:* {dep.commit_message}",
                f"*Deployed:* {dep.time_delta_minutes} minutes before anomaly",
                f"*Confidence:* {correlation_result.confidence.upper()}",
                f"*Workflow Run:* {dep.workflow_run_url}",
                "",
            ]

        if len(anomalies) > 1:
            description_lines.append("h2. 📊 All Anomalies")
            for a in anomalies:
                description_lines.append(f"* [{a.severity.upper()}] {a.metric_name}: {a.current_value}")

        description_lines += [
            "",
            "h2. 📋 Suggested Actions",
            "# Review the correlated deployment and consider rollback",
            "# Check Salesforce Event Monitoring for detailed logs",
            "# Monitor org limits for the next 30 minutes",
            "# Update this ticket with RCA findings",
        ]

        payload = {
            "fields": {
                "project":     {"key": self.project},
                "summary":     summary,
                "description": "\n".join(description_lines),
                "issuetype":   {"name": self.issue_type},
                "priority":    {"name": "High" if top.severity in ("critical", "high") else "Medium"},
                "labels":      ["aiops", "salesforce", "automated"],
            }
        }

        url = f"{self.base_url}/rest/api/2/issue"
        resp = requests.post(url, json=payload, auth=self._auth(),
                             headers={"Content-Type": "application/json"})

        if resp.status_code == 201:
            issue_key = resp.json()["key"]
            ticket_url = f"{self.base_url}/browse/{issue_key}"
            print(f"✅ Jira ticket created: {ticket_url}")
            return ticket_url
        else:
            print(f"❌ Jira error: {resp.status_code} — {resp.text}")
            return None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📢 Testing Slack notification (dry run — no real webhook needed)...")
    print("Set SLACK_WEBHOOK_URL and JIRA_* env vars to use live integrations.")
    print("\nPayload structure preview:")

    sample_blocks = {
        "channel": "#salesforce-ops",
        "blocks": [
            "header: 🔴 Salesforce Org Anomaly Detected — CRITICAL",
            "section: DailyApiRequests at 87.5% (expected 10–40%)",
            "divider",
            "section: 🎯 Root Cause — commit #abc1234 by priya.sharma (22 min ago)",
            "actions: [Create Jira] [View Event Monitoring] [Suggest Rollback]",
        ]
    }
    print(json.dumps(sample_blocks, indent=2))
