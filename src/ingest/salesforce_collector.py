import os, json, requests, csv, io
from datetime import datetime, timedelta

class SalesforceCollector:
    def __init__(self):
        self.instance_url  = None
        self.access_token  = None
        self.client_id     = os.environ.get("SF_CLIENT_ID")
        self.client_secret = os.environ.get("SF_CLIENT_SECRET")
        self.domain        = os.environ.get("SF_DOMAIN", "devgroup2-dev-ed.develop.my.salesforce.com")

    def authenticate(self):
        resp = requests.post(f"https://{self.domain}/services/oauth2/token", data={
            "grant_type":    "client_credentials",
            "client_id":     self.client_id,
            "client_secret": self.client_secret,
        })
        resp.raise_for_status()
        data = resp.json()
        self.access_token = data["access_token"]
        self.instance_url = data["instance_url"]
        print(f"Authenticated to: {self.instance_url}")
        return self

    def _headers(self):
        return {"Authorization": f"Bearer {self.access_token}"}

    def get_org_limits(self):
        resp = requests.get(f"{self.instance_url}/services/data/v60.0/limits/", headers=self._headers())
        resp.raise_for_status()
        limits = {}
        for name, vals in resp.json().items():
            if isinstance(vals, dict) and "Max" in vals:
                max_val = vals["Max"]
                remaining = vals["Remaining"]
                used = max_val - remaining
                pct = round((used / max_val) * 100, 2) if max_val > 0 else 0
                limits[name] = {"Max": max_val, "Remaining": remaining, "UsedPct": pct}
        print(f"Fetched {len(limits)} org limits")
        return limits

    def get_event_log_files(self, hours_back=24):
        since = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = f"SELECT Id, EventType, LogDate, LogFile FROM EventLogFile WHERE LogDate >= {since} ORDER BY LogDate DESC"
        resp = requests.get(f"{self.instance_url}/services/data/v60.0/query/", headers=self._headers(), params={"q": query})
        resp.raise_for_status()
        records = resp.json().get("records", [])
        print(f"Found {len(records)} event log file(s)")
        return records

    def download_event_log(self, log_record):
        resp = requests.get(self.instance_url + log_record["LogFile"], headers=self._headers())
        resp.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(resp.text)))
        print(f"  -> {log_record['EventType']}: {len(rows)} rows")
        return rows

    def collect_all(self, hours_back=24):
        self.authenticate()
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "limits": self.get_org_limits(),
            "event_logs": {},
        }
        for lf in self.get_event_log_files(hours_back):
            etype = lf["EventType"]
            rows = self.download_event_log(lf)
            snapshot["event_logs"].setdefault(etype, []).extend(rows)
        return snapshot

if __name__ == "__main__":
    collector = SalesforceCollector()
    snapshot = collector.collect_all(hours_back=24)
    os.makedirs(os.path.expanduser("~/aiops-salesforce/data"), exist_ok=True)
    path = os.path.expanduser("~/aiops-salesforce/data/snapshot_latest.json")
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\nSnapshot saved to {path}")
    print(f"Limits: {len(snapshot['limits'])}")
    print(f"Event types: {list(snapshot['event_logs'].keys())}")
