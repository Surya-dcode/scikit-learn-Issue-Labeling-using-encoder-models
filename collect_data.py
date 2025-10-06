#!/usr/bin/env python3
"""
Scikit-learn GitHub Issues Data Collection (flexible version)

- Non-interactive: pass --token/--target-count/--max-pages (or use env GITHUB_TOKEN)
- Looser defaults to collect more issues faster:
    * By default, accepts ANY labeled issue (no target-label restriction)
    * No min title/body length by default
    * max-pages default bumped to 300
- Optional strict filters can be re-enabled via flags

Examples:
  export GITHUB_TOKEN=ghp_xxx
  python collect_data.py --target-count 2000 --max-pages 400 --jsonl-out data/raw_issues.jsonl

  # With explicit token and stricter filtering:
  python collect_data.py --token ghp_xxx --target-count 2000 --require-target-labels \
      --min-title 10 --min-body 20
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

# --------------------------- logging ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sklearn_data_collection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _s(x: Optional[str]) -> str:
    """Safe string strip."""
    return (x or "").strip()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SklearnIssueCollector:
    def __init__(
        self,
        github_token: Optional[str] = None,
        require_target_labels: bool = False,
        min_title: int = 0,
        min_body: int = 0,
    ):
        self.base_url = "https://api.github.com/repos/scikit-learn/scikit-learn/issues"
        self.session = requests.Session()

        if github_token:
            self.session.headers.update(
                {"Authorization": f"Bearer {github_token}", "Accept": "application/vnd.github+json"}
            )
            logger.info("GitHub token configured - higher rate limits available")
        else:
            logger.warning("No GitHub token provided - limited to 60 requests/hour")

        # Filters
        self.require_target_labels = require_target_labels
        self.min_title = max(0, int(min_title))
        self.min_body = max(0, int(min_body))

        # Optional target labels (used only if require_target_labels=True)
        self.target_labels = {
            "Bug",
            "Enhancement",
            "Documentation",
            "Question",
            "Feature Request",
            "Performance",
            "API",
            "Good first issue",
            "Help wanted",
            "Maintenance",
        }
        self.target_labels_lower = {l.lower() for l in self.target_labels}

        os.makedirs("data", exist_ok=True)

    # -------------- Rate limit helpers --------------
    def check_rate_limit(self) -> Dict:
        try:
            r = self.session.get("https://api.github.com/rate_limit", timeout=60)
            if r.status_code == 200:
                return r.json().get("rate", {})
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
        return {}

    def wait_for_rate_limit_reset(self, rate_limit_info: Dict):
        remaining = int(rate_limit_info.get("remaining", 1) or 1)
        reset_time = int(rate_limit_info.get("reset", time.time()) or time.time())
        if remaining <= 5:
            wait_time = reset_time - int(time.time()) + 5
            if wait_time > 0:
                logger.info(f"Rate limit low ({remaining} remaining). Waiting {wait_time}s for reset...")
                time.sleep(wait_time)

    # -------------- Fetch --------------
    def fetch_issues_page(self, page: int, per_page: int = 100) -> List[Dict]:
        params = {"state": "all", "per_page": per_page, "page": page, "sort": "created", "direction": "desc"}
        try:
            r = self.session.get(self.base_url, params=params, timeout=60)
            if r.status_code == 403:
                rate = self.check_rate_limit()
                logger.warning(f"Rate limited. Remaining: {rate.get('remaining', 0)}")
                self.wait_for_rate_limit_reset(rate)
                return self.fetch_issues_page(page, per_page)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {page}: {e}")
            return []

    # -------------- Filter / clean --------------
    def is_valid_issue(self, issue: Dict) -> bool:
        # Skip PRs
        if issue.get("pull_request"):
            return False

        labels = (issue.get("labels") or [])
        if not labels:
            return False  # accept ONLY labeled issues

        title = _s(issue.get("title"))
        body = _s(issue.get("body"))

        if len(title) < self.min_title or len(body) < self.min_body:
            return False

        if self.require_target_labels:
            issue_label_names = {_s(lbl.get("name")) for lbl in labels if isinstance(lbl, dict)}
            has_target = any((l.lower() in self.target_labels_lower) for l in issue_label_names)
            if not has_target:
                return False

        return True

    def clean_issue_data(self, issue: Dict) -> Dict:
        labels_raw = (issue.get("labels") or [])
        names = []
        for lbl in labels_raw:
            name = _s(lbl.get("name") if isinstance(lbl, dict) else None)
            if name:
                names.append(name)

        body = _s(issue.get("body")).replace("\r\n", "\n").replace("\r", "\n")
        if len(body) > 2000:
            body = body[:2000] + "..."

        return {
            "number": issue.get("number"),
            "title": _s(issue.get("title")),
            "body": body,
            "labels": names,
            "state": _s(issue.get("state")),
            "created_at": _s(issue.get("created_at")),
            "updated_at": _s(issue.get("updated_at")),
            "comments": issue.get("comments", 0) or 0,
            "url": _s(issue.get("html_url")),
        }

    # -------------- Persistence --------------
    def save_checkpoint(self, issues: List[Dict], filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"data/sklearn_issues_checkpoint_{now_stamp()}.json"
        payload = {"timestamp": datetime.now().isoformat(), "total_issues": len(issues), "issues": issues}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Checkpoint saved: {len(issues)} issues -> {filename}")
        return filename

    @staticmethod
    def append_jsonl(rec: Dict, path: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -------------- Main loop --------------
    def collect_issues(
        self,
        target_count: int = 2000,
        max_pages: int = 300,
        sleep_s: float = 0.4,
        jsonl_out: Optional[str] = None,
    ) -> List[Dict]:
        logger.info(f"Starting collection of {target_count} scikit-learn issues...")
        rate = self.check_rate_limit()
        logger.info(f"Rate limit status: {rate.get('remaining', 'Unknown')}/{rate.get('limit', 'Unknown')}")

        valid: List[Dict] = []
        page = 1
        empty_pages = 0

        if jsonl_out:
            open(jsonl_out, "w", encoding="utf-8").close()
            logger.info(f"Will append cleaned issues to JSONL: {jsonl_out}")

        while len(valid) < target_count and page <= max_pages:
            logger.info(f"Fetching page {page}... (Valid issues so far: {len(valid)})")
            items = self.fetch_issues_page(page)
            if not items:
                empty_pages += 1
                if empty_pages >= 3:
                    logger.warning("Multiple empty pages, stopping collection")
                    break
                page += 1
                continue
            empty_pages = 0

            page_valid = 0
            for raw in items:
                try:
                    if self.is_valid_issue(raw):
                        cleaned = self.clean_issue_data(raw)
                        valid.append(cleaned)
                        page_valid += 1
                        if jsonl_out:
                            self.append_jsonl(cleaned, jsonl_out)
                        if len(valid) >= target_count:
                            break
                except Exception as e:
                    logger.error(f"Skip issue #{raw.get('number')} due to error: {e}")

            logger.info(f"Page {page}: {page_valid} valid issues found")

            if page % 5 == 0:
                self.save_checkpoint(valid)

            time.sleep(sleep_s)
            page += 1

        logger.info(f"Collection complete! Gathered {len(valid)} valid issues")
        return valid

    # -------------- Analysis / save --------------
    def analyze_labels(self, issues: List[Dict]) -> pd.DataFrame:
        counts: Dict[str, int] = {}
        for it in issues:
            for L in (it.get("labels") or []):
                counts[L] = counts.get(L, 0) + 1
        total = max(len(issues), 1)
        return pd.DataFrame(
            [{"label": k, "count": v, "percentage": v * 100.0 / total} for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
        )

    def save_final_dataset(self, issues: List[Dict], prefix: Optional[str] = None) -> Dict[str, str]:
        ts = now_stamp()
        base = prefix or f"data/sklearn_issues_final_{ts}"

        json_path = f"{base}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)

        df = pd.DataFrame(issues)
        csv_path = f"{base}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")

        la = self.analyze_labels(issues)
        analysis_path = f"data/sklearn_label_analysis_{ts}.csv"
        la.to_csv(analysis_path, index=False)

        logger.info("Final dataset saved:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  CSV: {csv_path}")
        logger.info(f"  Label Analysis: {analysis_path}")

        print(f"\n🎉 Data Collection Complete!")
        print(f"📊 Total Issues: {len(issues)}")
        print(f"📁 Files saved in 'data/' directory")
        print(f"\n📈 Top Labels:")
        if not la.empty:
            print(la.head(10).to_string(index=False))
        else:
            print("(no labels)")

        return {"json": json_path, "csv": csv_path, "analysis": analysis_path}


# --------------------------- CLI ---------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect labeled scikit-learn GitHub issues.")
    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN"), help="GitHub token (or set GITHUB_TOKEN)")
    ap.add_argument("--target-count", type=int, default=2000, help="Number of valid issues to collect")
    ap.add_argument("--max-pages", type=int, default=300, help="Max pages to fetch (100 items/page)")
    ap.add_argument("--sleep-s", type=float, default=0.4, help="Delay between page fetches")
    ap.add_argument("--jsonl-out", default=None, help="Optional JSONL path to append cleaned issues during collection")
    ap.add_argument("--final-prefix", default=None, help="Prefix for final JSON/CSV outputs (default includes timestamp)")

    # Filtering controls
    ap.add_argument("--require-target-labels", action="store_true", help="Only keep issues with specific target labels")
    ap.add_argument("--min-title", type=int, default=0, help="Minimum title length (default 0)")
    ap.add_argument("--min-body", type=int, default=0, help="Minimum body length (default 0)")
    return ap.parse_args()


def main():
    args = parse_args()

    if not args.token:
        logger.warning("No token supplied (flag or GITHUB_TOKEN). Proceeding with unauthenticated rate limits.")

    collector = SklearnIssueCollector(
        github_token=args.token,
        require_target_labels=args.require_target_labels,
        min_title=args.min_title,
        min_body=args.min_body,
    )

    try:
        issues = collector.collect_issues(
            target_count=args.target_count,
            max_pages=args.max_pages,
            sleep_s=args.sleep_s,
            jsonl_out=args.jsonl_out,
        )
        if issues:
            collector.save_final_dataset(issues, prefix=args.final_prefix)
        else:
            print("❌ No issues collected!")
    except KeyboardInterrupt:
        print("\n⚠️  Collection interrupted by user")
        if "issues" in locals() and issues:
            path = collector.save_checkpoint(issues, "data/sklearn_issues_interrupted.json")
            print(f"💾 Saved {len(issues)} issues to {path}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
