"""Collect a creation-date cohort of public scikit-learn issues with GitHub CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = "scikit-learn/scikit-learn"
PAGE_SIZE = 100
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def months_between(start: str, end: str):
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    while (year, month) <= (end_year, end_month):
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        first = f"{year:04d}-{month:02d}-01"
        last = (datetime(next_year, next_month, 1, tzinfo=timezone.utc) - timedelta(days=1)).date().isoformat()
        yield f"{year:04d}-{month:02d}", first, last
        year, month = next_year, next_month


def search_page(first: str, last: str, page: int) -> dict:
    query = f"repo:{REPO} is:issue created:{first}..{last}"
    command = ["gh", "api", "-X", "GET", "search/issues", "-f", f"q={query}",
               "-f", f"per_page={PAGE_SIZE}", "-f", f"page={page}",
               "-f", "sort=created", "-f", "order=asc"]
    for attempt in range(3):
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode == 0:
            return json.loads(run.stdout)
        if attempt == 2:
            raise RuntimeError(f"GitHub search failed for {first}, page {page}: {run.stderr.strip()}")
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def compact(item: dict) -> dict:
    return {
        "issue_number": item["number"],
        "title": item.get("title") or "",
        "body": item.get("body") or "",
        "created_at": item["created_at"],
        "updated_at": item.get("updated_at"),
        "closed_at": item.get("closed_at"),
        "state": item.get("state"),
        "labels": [label["name"] for label in item.get("labels", [])],
        "source_url": item["html_url"],
    }


def normalize_text(value: str) -> str:
    """Redact emails and standardize line endings in public issue text."""
    value = (EMAIL.sub("[email redacted]", value).replace("\r\n", "\n")
             .replace("\r", "\n").replace("\t", "    "))
    return "\n".join(line.rstrip() for line in value.split("\n"))


def redact_contact(record: dict) -> dict:
    return {**record, "title": normalize_text(record["title"]),
            "body": normalize_text(record["body"])}


def target(labels: list[str]) -> str | None:
    normalized = {label.strip().casefold() for label in labels}
    bug = "bug" in normalized
    feature = bool(normalized & {"enhancement", "new feature"})
    if bug and feature:
        return "Ambiguous"
    if bug:
        return "Bug"
    if feature:
        return "Feature Request"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01", help="First month, YYYY-MM")
    parser.add_argument("--end", default="2026-10", help="Last month, YYYY-MM")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--refresh", action="store_true", help="Refetch every month instead of using the cache")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    previous_audit_path = args.out / "collection_audit.json"
    previous_audit = json.loads(previous_audit_path.read_text()) if previous_audit_path.exists() else None
    month_dir = args.out / "monthly_raw"
    month_dir.mkdir(exist_ok=True)
    all_records = []
    monthly_counts = {}
    used_cache = False
    fetched_new = False
    for key, first, last in months_between(args.start, args.end):
        cache = month_dir / f"{key}.json"
        if cache.exists() and not args.refresh:
            payload = json.loads(cache.read_text())
            used_cache = True
        else:
            fetched_new = True
            response = search_page(first, last, 1)
            if response.get("incomplete_results"):
                raise RuntimeError(f"GitHub returned incomplete results for {key}")
            count = response["total_count"]
            if count > 1000:
                raise RuntimeError(f"{key} has over 1,000 results; split this month into days")
            items = response["items"]
            for page in range(2, math.ceil(count / PAGE_SIZE) + 1):
                time.sleep(2.1)
                response = search_page(first, last, page)
                if response.get("incomplete_results"):
                    raise RuntimeError(f"GitHub returned incomplete results for {key}, page {page}")
                items.extend(response["items"])
            if len(items) != count:
                raise RuntimeError(f"{key}: expected {count} records, received {len(items)}")
            payload = [compact(item) for item in items]
            cache.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            time.sleep(2.1)
        monthly_counts[key] = len(payload)
        all_records.extend(redact_contact(record) for record in payload)
        print(f"{key}: {len(payload)} issues", flush=True)

    ids = [record["issue_number"] for record in all_records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate issue numbers across monthly pages")
    for record in all_records:
        month = record["created_at"][:7]
        if month not in monthly_counts:
            raise RuntimeError(f"Issue {record['issue_number']} is outside requested months")

    retrieved_at = (previous_audit["retrieved_at"] if used_cache and not fetched_new and previous_audit
                    else datetime.now(timezone.utc).isoformat())
    labeled = []
    removed = {"ambiguous": 0, "neither_target": 0}
    for record in all_records:
        label = target(record["labels"])
        if label == "Ambiguous":
            removed["ambiguous"] += 1
        elif label is None:
            removed["neither_target"] += 1
        else:
            labeled.append({**record, "issue_type": label})
    if len(labeled) != len(all_records) - sum(removed.values()):
        raise RuntimeError("Filtering counts do not reconcile")
    raw_path = args.out / "scikit_learn_issues_date_cohort_raw.json"
    raw_path.write_text(json.dumps({"repository": REPO, "retrieved_at": retrieved_at,
                                    "start_month": args.start, "end_month": args.end,
                                    "monthly_counts": monthly_counts,
                                    "records": all_records}, ensure_ascii=False, indent=2) + "\n")
    csv_path = args.out / "scikit_learn_issues_date_cohort_labeled.csv"
    fields = ["issue_number", "title", "body", "created_at", "updated_at", "closed_at",
              "state", "labels", "issue_type", "source_url"]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in labeled:
            writer.writerow({**record, "labels": json.dumps(record["labels"], ensure_ascii=False)})
    counts = {label: sum(record["issue_type"] == label for record in labeled)
              for label in ("Bug", "Feature Request")}
    audit = {"repository": REPO, "retrieved_at": retrieved_at,
             "source": "GitHub REST issue search via gh api, creation-month queries",
             "start_month": args.start, "end_month": args.end,
             "monthly_counts": monthly_counts, "raw_count": len(all_records),
             "excluded": removed, "labeled_count": len(labeled), "class_counts": counts,
             "label_mapping": {"Bug": ["Bug"], "Feature Request": ["Enhancement", "New Feature"]},
             "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
             "labeled_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
             "note": ("Issue titles and bodies have email addresses redacted. "
                      "Labels are the GitHub snapshot at retrieval, not historical label events. "
                      "Cached monthly files are reused unless --refresh is supplied.")}
    (args.out / "collection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"raw": audit["raw_count"], "excluded": removed,
                      "labeled": audit["labeled_count"], "classes": counts}, indent=2))


if __name__ == "__main__":
    main()
