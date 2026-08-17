"""
prep_retailrocket_for_upload.py

Reshapes RetailRocket's raw events.csv (columns: timestamp, visitorid,
event, itemid, transactionid) into a flat CSV matching whatever schema
your app's CSV importer expects.

BEFORE RUNNING:
1. Download RetailRocket from https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
   and unzip it. You need events.csv (and optionally item_properties*.csv
   if you want real category names instead of hashed IDs).
2. Open the CSV you already successfully uploaded (e.g. amazon_flipkart_orders.csv)
   and copy its exact column headers into TARGET_COLUMNS below, in the
   same order the app expects them.
3. Adjust the mapping in `build_row()` so each RetailRocket field lands in
   the right target column.

USAGE:
    python prep_retailrocket_for_upload.py events.csv --sample 20000
    python prep_retailrocket_for_upload.py events.csv --sample 20000 --out test_upload.csv
"""

import argparse
import csv
import random
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# TARGET_COLUMNS matches CustomerPulse AI standard e-commerce event schema
# ---------------------------------------------------------------------------
TARGET_COLUMNS = [
    "customer_id",
    "order_id",
    "order_date",
    "event_type",
    "product_id",
    "category",
    "amount",
]

EVENT_MAP = {
    "view": "view",
    "addtocart": "addtocart",
    "transaction": "transaction",
}


def build_row(rr_row, item_price_lookup=None):
    """Map one RetailRocket events.csv row -> TARGET_COLUMNS dict."""
    ts_ms = int(rr_row["timestamp"])
    order_date = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
    raw_event = str(rr_row.get("event", "")).lower()
    event = EVENT_MAP.get(raw_event, raw_event)
    item_id = rr_row.get("itemid", "")

    # If transaction event without amount, leave empty or calculate from lookup
    amount = rr_row.get("amount", "")

    return {
        "customer_id": rr_row.get("visitorid", ""),
        "order_id": rr_row.get("transactionid") or "",
        "order_date": order_date,
        "event_type": event,
        "product_id": item_id,
        "category": rr_row.get("category", ""),
        "amount": amount,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events_csv", help="Path to RetailRocket events.csv")
    parser.add_argument("--sample", type=int, default=20000,
                         help="Number of rows to sample (default 20000)")
    parser.add_argument("--out", default="retailrocket_test_upload.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Reading {args.events_csv} ...")
    with open(args.events_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows):,} raw events.")

    if args.sample and args.sample < len(rows):
        from collections import defaultdict
        by_visitor = defaultdict(list)
        for r in rows:
            by_visitor[r["visitorid"]].append(r)

        visitors = list(by_visitor.keys())
        random.shuffle(visitors)

        kept_rows = []
        for v in visitors:
            kept_rows.extend(by_visitor[v])
            if len(kept_rows) >= args.sample:
                break
        rows = kept_rows
        print(f"Sampled down to {len(rows):,} events (full per-visitor sequences kept intact).")

    out_rows = [build_row(r) for r in rows]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows):,} rows to {args.out}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(0)
    main()
