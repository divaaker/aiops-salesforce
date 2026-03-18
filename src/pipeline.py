"""
src/pipeline.py
Main AIOps pipeline orchestrator.
Run this to execute the full: Ingest → Detect → Correlate → Act loop.

Usage:
    python src/pipeline.py              # single run
    python src/pipeline.py --watch 60  # poll every 60 seconds
"""

import time
import json
import argparse
import os
from datetime import datetime

from ingest.salesforce_collector import SalesforceCollector
from detect.anomaly_detector import AnomalyDetector
from correlate.deployment_correlator import DeploymentCorrelator
from act.notifier import SlackNotifier, JiraTicketCreator


# ── Bootstrap baseline (first-run helper) ─────────────────────────────────────

def bootstrap_baseline(collector: SalesforceCollector, detector: AnomalyDetector,
                        snapshots: int = 20, interval_sec: int = 30):
    """
    Collect `snapshots` data points to build an initial baseline model.
    Call this once before going into watch mode.
    """
    print(f"\n🔄 Bootstrapping baseline with {snapshots} snapshots...")
    for i in range(snapshots):
        print(f"  Snapshot {i+1}/{snapshots}")
        snapshot = collector.collect_all(hours_back=1)
        detector.add_to_baseline(snapshot)
        if i < snapshots - 1:
            time.sleep(interval_sec)

    detector.train()
    print("✅ Baseline ready!\n")


# ── Single pipeline run ───────────────────────────────────────────────────────

def run_once(collector, detector, correlator, slack, jira, dry_run=False):
    print(f"\n{'='*60}")
    print(f"🚀 Pipeline run at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}")

    # Step 1: Ingest
    print("\n📥 Step 1: Ingesting Salesforce data...")
    snapshot = collector.collect_all(hours_back=1)

    # Save raw snapshot for debugging
    os.makedirs("data", exist_ok=True)
    with open("data/snapshot_latest.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    # Step 2: Detect
    print("\n🤖 Step 2: Running anomaly detection...")
    if not detector.is_trained:
        print("⚠️  Model not trained yet — adding to baseline instead.")
        detector.add_to_baseline(snapshot)
        return

    anomalies = detector.detect(snapshot)
    print(f"   → Found {len(anomalies)} anomaly/anomalies")

    if not anomalies:
        print("✅ Org looks healthy. No action needed.")
        return

    for a in anomalies[:3]:
        print(f"   [{a.severity.upper()}] {a.metric_name}: {a.current_value} (score: {a.raw_score})")

    # Step 3: Correlate
    print("\n🔗 Step 3: Correlating with deployments...")
    correlation = correlator.correlate(anomalies[0].timestamp)
    if correlation.matched_deployment:
        print(f"   → MATCH [{correlation.confidence.upper()}]: {correlation.reasoning}")
    else:
        print(f"   → No recent deployment correlated ({correlation.reasoning})")

    # Step 4: Act
    print("\n📢 Step 4: Firing notifications...")

    if dry_run:
        print("   [DRY RUN] Would send Slack + create Jira ticket")
        print(f"   Anomalies: {[a.metric_name for a in anomalies]}")
        print(f"   Correlation confidence: {correlation.confidence}")
        return

    # Only alert on medium+ severity
    critical_anomalies = [a for a in anomalies if a.severity in ("critical", "high", "medium")]
    if critical_anomalies:
        slack.send_anomaly_alert(critical_anomalies, correlation)
        jira.create_incident(critical_anomalies, correlation)
    else:
        print("   Low severity anomalies — no alert fired (within tolerance).")

    print("\n✅ Pipeline complete!")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AIOps for Salesforce")
    parser.add_argument("--watch", type=int, default=0,
                        help="Poll interval in seconds (0 = run once)")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Bootstrap baseline before starting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect but don't send alerts")
    args = parser.parse_args()

    # Init components
    collector   = SalesforceCollector()
    detector    = AnomalyDetector()
    correlator  = DeploymentCorrelator()
    slack       = SlackNotifier()
    jira        = JiraTicketCreator()

    if args.bootstrap:
        bootstrap_baseline(collector, detector)
    else:
        # Load pre-trained model if available
        print("💡 Tip: Run with --bootstrap to build a baseline first.")

    if args.watch > 0:
        print(f"\n👀 Watching org every {args.watch}s — press Ctrl+C to stop\n")
        try:
            while True:
                run_once(collector, detector, correlator, slack, jira, dry_run=args.dry_run)
                print(f"\n⏳ Next check in {args.watch}s...")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n👋 Stopped.")
    else:
        run_once(collector, detector, correlator, slack, jira, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
