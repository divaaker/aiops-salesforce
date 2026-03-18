"""
src/detect/anomaly_detector.py
AI-powered anomaly detection for Salesforce org metrics.
Uses Isolation Forest (unsupervised) — no labeled data needed.
"""

import json
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Anomaly:
    timestamp: str
    metric_name: str
    current_value: float
    expected_range: tuple
    severity: str          # "low" | "medium" | "high" | "critical"
    description: str
    raw_score: float


# ── Key metrics to monitor ────────────────────────────────────────────────────

MONITORED_LIMITS = [
    "DailyApiRequests",
    "DailyAsyncApexExecutions",
    "DailyBulkApiRequests",
    "HourlyODataCallout",
    "HourlyShortTermEntitlement",
    "DataStorageMB",
    "FileStorageMB",
    "ConcurrentAsyncGetReportInstances",
]

SOQL_HEAVY_EVENT_TYPES = ["ApexExecution", "ApexSoap", "API"]


# ── Detector ──────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for Salesforce org metrics.

    Workflow:
        1. Feed it multiple historical snapshots to build a baseline.
        2. Call detect(snapshot) with a new snapshot to flag anomalies.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.baseline_history: List[Dict] = []
        self.is_trained = False

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, snapshot: dict) -> Dict[str, float]:
        """Extract numeric features from a snapshot dict."""
        features = {}

        # Limits: use UsedPct
        for limit_name in MONITORED_LIMITS:
            val = snapshot.get("limits", {}).get(limit_name, {}).get("UsedPct", 0.0)
            features[f"limit_{limit_name}"] = val

        # Event log aggregates
        event_logs = snapshot.get("event_logs", {})
        for etype in SOQL_HEAVY_EVENT_TYPES:
            rows = event_logs.get(etype, [])
            if rows:
                soql_counts = [int(r.get("NUMBER_SOQL_QUERIES", 0)) for r in rows]
                cpu_times   = [int(r.get("CPU_TIME", 0)) for r in rows]
                features[f"{etype}_avg_soql"]   = np.mean(soql_counts)
                features[f"{etype}_max_soql"]   = np.max(soql_counts)
                features[f"{etype}_avg_cpu_ms"] = np.mean(cpu_times)
            else:
                features[f"{etype}_avg_soql"]   = 0.0
                features[f"{etype}_max_soql"]   = 0.0
                features[f"{etype}_avg_cpu_ms"] = 0.0

        return features

    def _features_to_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dict to ordered list."""
        # Sort keys for consistent ordering
        return [features[k] for k in sorted(features.keys())]

    # ── Training ──────────────────────────────────────────────────────────────

    def add_to_baseline(self, snapshot: dict):
        """Add a historical snapshot to build training data."""
        features = self._extract_features(snapshot)
        self.baseline_history.append(features)

    def train(self):
        """Train the Isolation Forest on accumulated baseline history."""
        if len(self.baseline_history) < 10:
            print("⚠️  Need at least 10 baseline snapshots for reliable detection.")

        X = [self._features_to_vector(f) for f in self.baseline_history]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        print(f"✅ Model trained on {len(self.baseline_history)} baseline snapshots")

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect(self, snapshot: dict) -> List[Anomaly]:
        """
        Detect anomalies in a new snapshot.
        Returns a list of Anomaly objects sorted by severity.
        """
        if not self.is_trained:
            raise RuntimeError("Call train() before detect()")

        features = self._extract_features(snapshot)
        vector = self._features_to_vector(features)
        X_scaled = self.scaler.transform([vector])

        # Isolation Forest score: lower = more anomalous (-1 = anomaly, 1 = normal)
        score = self.model.decision_function(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]

        anomalies = []

        if prediction == -1:  # Global anomaly detected
            # Find which individual metrics are most deviated
            for key, value in features.items():
                history_vals = [h[key] for h in self.baseline_history]
                mean = np.mean(history_vals)
                std  = np.std(history_vals) or 1.0
                z_score = abs((value - mean) / std)

                if z_score > 2.0:  # More than 2 std deviations
                    severity = self._score_severity(z_score, value, key)
                    anomalies.append(Anomaly(
                        timestamp=snapshot.get("timestamp", datetime.utcnow().isoformat()),
                        metric_name=key,
                        current_value=value,
                        expected_range=(round(mean - 2*std, 2), round(mean + 2*std, 2)),
                        severity=severity,
                        description=self._describe(key, value, mean, z_score),
                        raw_score=round(score, 4),
                    ))

        # Sort: critical first
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 4))
        return anomalies

    def _score_severity(self, z_score: float, value: float, key: str) -> str:
        """Map z-score and metric context to severity level."""
        # Limits that are near 100% are always critical
        if "limit_" in key and value > 90:
            return "critical"
        if "limit_" in key and value > 75:
            return "high"
        if z_score > 4:
            return "critical"
        if z_score > 3:
            return "high"
        if z_score > 2:
            return "medium"
        return "low"

    def _describe(self, key: str, value: float, mean: float, z_score: float) -> str:
        direction = "above" if value > mean else "below"
        return (
            f"{key} is {round(value, 1)} — {round(z_score, 1)}σ {direction} "
            f"the baseline mean of {round(mean, 1)}."
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: load a snapshot and run detection
    import os

    snapshot_path = "data/snapshot_latest.json"
    if not os.path.exists(snapshot_path):
        print("❌ No snapshot found. Run the ingest step first.")
        exit(1)

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    # For demo: create a minimal dummy detector with fake baseline
    detector = AnomalyDetector()

    # Simulate 20 baseline snapshots (in real usage, these come from history)
    import copy, random
    for _ in range(20):
        fake = copy.deepcopy(snapshot)
        # Randomize limits slightly to simulate normal variation
        for k in fake["limits"]:
            fake["limits"][k]["UsedPct"] = max(0, fake["limits"][k].get("UsedPct", 10) + random.uniform(-5, 5))
        detector.add_to_baseline(fake)

    detector.train()

    # Simulate a spike for demo purposes
    demo_snapshot = copy.deepcopy(snapshot)
    demo_snapshot["limits"].setdefault("DailyApiRequests", {})["UsedPct"] = 87.5  # Spike!

    anomalies = detector.detect(demo_snapshot)

    if anomalies:
        print(f"\n🚨 {len(anomalies)} anomaly/anomalies detected!\n")
        for a in anomalies:
            print(f"  [{a.severity.upper()}] {a.metric_name}")
            print(f"    → {a.description}")
            print(f"    → Expected range: {a.expected_range}")
            print()
    else:
        print("✅ No anomalies detected — org looks healthy!")
