"""
Lane 7 Demand Forecasting — Transaction Base ETL
Phase 1: Order + OrderLine → transaction_base.csv

Assumptions documented inline. Run from the directory containing the xlsx files.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = "."          # Directory with xlsx files — change if needed
OUTPUT_DIR = "./output" # Where c`leaned CSVs land

ORDER_FILES = [
    "Lane7 Order Table 1 of 3.xlsx",
    "Lane7 Order Table 2 of 3.xlsx",
    "Lane7 Order Table 3 of 3.xlsx",
]

ORDERLINE_FILES = [
    "Lane 7 OrderLine 1 of X.xlsx", # 2017–2019
    "Lane 7 OrderLine 2 of X.xlsx", # 2020
    "Lane 7 OrderLine 3 of X.xlsx", # 2021 first 100k
    "Lane 7 OrderLine 4 of X.xlsx", # 2021 remainder
    "Lane 7 OrderLine 5 of X.xlsx", # 2022 first 150k
    "Lane 7 OrderLine 6 of X.xlsx", # 2022 remainder
    "Lane 7 OrderLine 7 of X.xlsx", # 2023 first 150k
    "Lane 7 OrderLine 8 of X.xlsx", # 2023 next 150k
    "Lane 7 OrderLine 9 of X.xlsx", # 2023 remainder
    "Lane 7 OrderLine 10 of X.xlsx", # 2024 first 150k
    "Lane 7 OrderLine 11 of X.xlsx", # 2024 next 150k
    "Lane 7 OrderLine 12 of X.xlsx", # 2024 remainder
    "Lane 7 OrderLine 13 of X.xlsx", # 2025 first ~185k
    "Lane 7 OrderLine 14 of X.xlsx", # 2025 next 150k
    "Lane 7 OrderLine 15 of X.xlsx", # 2025 next 150k
    "Lane 7 OrderLine 16 of X.xlsx", # 2025 remainder
    "Lane 7 OrderLine 17 of X.xlsx", # 2026
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# SECTION 1 — LOAD & UNION ORDER TABLE
# ─────────────────────────────────────────────

def load_order_files(files, data_dir):
    """Load and union all Order split files. Track provenance per row."""
    dfs = []
    row_counts = {}

    for fname in files:
        path = os.path.join(data_dir, fname)
        print(f"  Loading order file: {fname}")
        df = pd.read_excel(path, sheet_name="Sheet1")
        row_counts[fname] = len(df)
        df["_source_file"] = fname
        dfs.append(df)
        print(f"    → {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total order rows after union: {len(combined):,}")
    print(f"  Sum of individual files:      {sum(row_counts.values()):,}")
    assert len(combined) == sum(row_counts.values()), \
        "ROW COUNT MISMATCH after order union!"
    return combined, row_counts


def clean_order_table(df):
    """
    Standardize and clean the unified Order table.

    ASSUMPTIONS:
    - 'invoice-date' is the primary demand timing signal for forecasting
    - 'month_start' is retained as the accounting-period month for reconciliation / finance views
    - if invoice_date is missing, demand_month falls back to month_start
    - 'order-number' is the primary key for the Order table (confirmed: present in all 3 files)
    - 'order-date' is the canonical order creation date (string format: 'Tuesday, July 25, 2023')
    - 'invoice-date' is when the order was invoiced
    - 'order-status' = 'C' likely means Completed/Closed; 'O' open; 'B' backordered (needs confirmation)
    - 'order-value' / 'OrderValue' are redundant — OrderValue appears to be a corrected/final version
    - 'OrderBoNo' = composite of order-number + bo-number; useful as a backorder line key
    - 'company' values are all 'LSA' — single-company dataset
    - 'warehouse-status' = 'Invoiced' is the completed transaction state
    """
    # ── Date parsing ──────────────────────────────────
    # Dates stored as strings like "Tuesday, July 25, 2023"
    for date_col in ["order-date", "invoice-date", "created-date", "plan-ship-dt", "bo-created"]:
        if date_col in df.columns:
            if df[date_col].dtype == object:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            else:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # ── Standardize column names (snake_case) ─────────
    rename_map = {
        "order-number":            "order_number",
        "order-date":              "order_date",
        "invoice-num":             "invoice_num",
        "invoice-date":            "invoice_date",
        "invoice-total":           "invoice_total",
        "order-value":             "order_value_raw",
        "OrderValue":              "order_value",
        "OrderBoNo":               "order_bo_key",
        "order-status":            "order_status",
        "order-type":              "order_type",
        "warehouse-status":        "warehouse_status",
        "cust-number":             "cust_number",
        "cust-ord-num":            "cust_ord_num",
        "bill-to-cust":            "bill_to_cust",
        "acct-year":               "acct_year",
        "acct-period":             "acct_period",
        "acct-yp":                 "acct_yp",
        "order-created":           "order_created",
        "created-date":            "created_date",
        "plan-ship-dt":            "plan_ship_dt",
        "ship-name":               "ship_name",
        "ship-addr":               "ship_addr",
        "ship-city":               "ship_city",
        "ship-prov":               "ship_prov",
        "ship-pcode":              "ship_pcode",
        "ship-country":            "ship_country",
        "salesman":                "salesman",
        "warehouse":               "warehouse",
        "to-warehouse":            "to_warehouse",
        "terms-code":              "terms_code",
        "price-level":             "price_level",
        "price-categ":             "price_categ",
        "entry-source":            "entry_source",
        "order-code":              "order_code",
        "carrier":                 "carrier",
        "freight-chg":             "freight_chg",
        "cogs":                    "cogs",
        "contr-amount":            "contr_amount",
        "invoiced":                "invoiced",
        "credit-hold":             "credit_hold",
        "bo-number":               "bo_number",
        "bo-created":              "bo_created",
        "pick-slip-print-count":   "pick_slip_print_count",
        "po-number":               "po_number",
        "addr-number":             "addr_number",
        "attn":                    "attn",
        "description":             "description",
        "company":                 "company",
        "container_ship":          "container_ship",
        "user_id":                 "user_id",
        "assignee":                "assignee",
        "cust-ref":                "cust_ref",
        "ref-comp-number":         "ref_comp_number",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── Type coercion ──────────────────────────────────
    df["order_number"]  = df["order_number"].astype(str).str.strip()
    df["cust_number"]   = pd.to_numeric(df["cust_number"], errors="coerce")
    df["acct_year"]     = pd.to_numeric(df["acct_year"], errors="coerce").astype("Int64")
    df["acct_period"]   = pd.to_numeric(df["acct_period"], errors="coerce").astype("Int64")
    df["order_value"]   = pd.to_numeric(df["order_value"], errors="coerce")
    df["cogs"]          = pd.to_numeric(df["cogs"], errors="coerce")
    df["contr_amount"]  = pd.to_numeric(df["contr_amount"], errors="coerce")

    # ── Derived columns ───────────────────────────────
    df["order_month_start"] = df["order_date"].dt.to_period("M").dt.to_timestamp()


    return df


# ─────────────────────────────────────────────
# SECTION 2 — LOAD & UNION ORDERLINE TABLE
# ─────────────────────────────────────────────

def load_orderline_files(files, data_dir):
    """Load and union all OrderLine split files. Track provenance per row."""
    dfs = []
    row_counts = {}

    for fname in files:
        path = os.path.join(data_dir, fname)
        print(f"  Loading orderline file: {fname}")
        df = pd.read_excel(path, sheet_name="Sheet1")
        row_counts[fname] = len(df)
        df["_source_file"] = fname
        dfs.append(df)
        print(f"    → {len(df):,} rows | acct_year: {sorted(df['acct-year'].unique())}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total orderline rows after union: {len(combined):,}")
    print(f"  Sum of individual files:          {sum(row_counts.values()):,}")
    assert len(combined) == sum(row_counts.values()), \
        "ROW COUNT MISMATCH after orderline union!"
    return combined, row_counts


def clean_orderline_table(df):
    """
    Standardize and clean the unified OrderLine table.

    ASSUMPTIONS:
    - Composite PK = (order_number, line_number, bo_number)
      • order_number + line_number alone may not be unique due to backorder variants
      • bo_number = 0 is the original line; bo_number > 0 is a backorder split
    - 'item-number' is the SKU — already present directly in OrderLine
    - 'original-item' is the originally-requested item (may differ from item-number
       when a substitution occurred)
    - 'shipped-qty' is the primary UnitsSold metric
      • 'order-qty' = quantity originally ordered (may include cancelled/backordered qty)
      • 'cancel-qty' = units cancelled
      • 'returned-qty' = units returned after shipment
    - Net shipped formula: shipped_qty is already net shipped (not gross)
    - 'MonthStart' column already pre-computed in the data (first day of acct-period month)
    - 'sale-revenue' = revenue for this line (= price × shipped-qty minus discounts)
    - 'amount' appears to duplicate 'sale-revenue' (confirmed same values in samples)
    - 'invoice-date' is the fulfillment/demand signal date (not order-date)
    - acct-year / acct-period define the accounting calendar period
    - 'item-categ' is a product category — mixed case across years (needs normalization)
    """
    
    # ── Date parsing ──────────────────────────────────
    for date_col in ["invoice-date", "order-date"]:
        if date_col in df.columns:
            if df[date_col].dtype == object:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            else:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")


    
    # MonthStart is already a Timestamp in the data — coerce to ensure type
    if "MonthStart" in df.columns:
        df["MonthStart"] = pd.to_datetime(df["MonthStart"], errors="coerce")

    # ── Primary demand timing for forecasting ───────────
    # Use actual invoiced/shipped month for demand modeling
    df["demand_month"] = df["invoice-date"].dt.to_period("M").dt.to_timestamp()

    # Fallback only if invoice-date is missing
    df["demand_month"] = df["demand_month"].fillna(df["MonthStart"])


    # ── Standardize column names ───────────────────────
    rename_map = {
        "order-number":      "order_number",
        "line-number":       "line_number",
        "bo-number":         "bo_number",
        "item-number":       "item_number",
        "original-item":     "original_item",
        "description":       "description",
        "item-categ":        "item_categ",
        "cust-number":       "cust_number",
        "cust-item-name":    "cust_item_name",
        "order-date":        "order_date",
        "invoice-date":      "invoice_date",
        "acct-year":         "acct_year",
        "acct-period":       "acct_period",
        "acct-yp":           "acct_yp",
        "MonthStart":        "month_start",
        "order-qty":         "order_qty",
        "shipped-qty":       "shipped_qty",
        "cancel-qty":        "cancel_qty",
        "returned-qty":      "returned_qty",
        "backordered":       "backordered",
        "action-qty":        "action_qty",
        "action-code":       "action_code",
        "actual-qty":        "actual_qty",
        "parent-qty":        "parent_qty",
        "sale-revenue":      "sale_revenue",
        "amount":            "amount",
        "price":             "price",
        "line-discount":     "line_discount",
        "discount_type":     "discount_type",
        "cogs":              "cogs",
        "unit-cost":         "unit_cost",
        "std-cost":          "std_cost",
        "est-cost":          "est_cost",
        "contract-cost":     "contract_cost",
        "volume":            "volume",
        "warehouse":         "warehouse",
        "to-warehouse":      "to_warehouse",
        "order-type":        "order_type",
        "company":           "company",
        "ref-comp-number":   "ref_comp_number",
        "ref-line-number":   "ref_line_number",
        "ref-bo-number":     "ref_bo_number",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── Type coercion ──────────────────────────────────
    df["order_number"]  = df["order_number"].astype(str).str.strip()
    df["item_number"]   = df["item_number"].astype(str).str.strip()
    df["original_item"] = df["original_item"].astype(str).str.strip()
    df["line_number"]   = pd.to_numeric(df["line_number"], errors="coerce").astype("Int64")
    df["bo_number"]     = pd.to_numeric(df["bo_number"],   errors="coerce").astype("Int64")
    df["cust_number"]   = df["cust_number"].astype(str).str.strip()  # mixed int/str (e.g. 'ALPHAB')
    df["acct_year"]     = pd.to_numeric(df["acct_year"],   errors="coerce").astype("Int64")
    df["acct_period"]   = pd.to_numeric(df["acct_period"], errors="coerce").astype("Int64")

    for num_col in ["order_qty","shipped_qty","cancel_qty","returned_qty","action_qty",
                    "actual_qty","parent_qty","backordered"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)

    for rev_col in ["sale_revenue","amount","price","line_discount","cogs",
                    "unit_cost","std_cost","contract_cost","volume"]:
        df[rev_col] = pd.to_numeric(df[rev_col], errors="coerce")

    # ── Normalize item_categ (mixed case across years) ─
    df["item_categ"] = df["item_categ"].astype(str).str.strip().str.title()

    # ── Composite PK ──────────────────────────────────
    df["ol_pk"] = (
        df["order_number"].astype(str) + "|"
        + df["line_number"].astype(str) + "|"
        + df["bo_number"].astype(str)
    )

    # ── Substitution flag ─────────────────────────────
    # item_number differs from original_item when a substitution happened
    df["is_substitution"] = (df["item_number"] != df["original_item"]).astype(int)

    # ── Return / cancellation flags ────────────────────
    # ASSUMPTION: returned_qty > 0 means a return; cancel_qty > 0 means a cancellation.
    # These rows are NOT filtered out here — downstream logic decides whether to exclude.
    df["has_return"]     = (df["returned_qty"] > 0).astype(int)
    df["has_cancellation"] = (df["cancel_qty"] > 0).astype(int)
    df["has_zero_shipped"] = (df["shipped_qty"] == 0).astype(int)

    return df


# ─────────────────────────────────────────────
# SECTION 3 — JOIN ORDER + ORDERLINE
# ─────────────────────────────────────────────

def build_transaction_base(orders, orderlines):
    """
    Join Order header to OrderLine detail.

    JOIN KEY: order_number (left join — OrderLine drives; we want every orderline
    even if the order header is missing, and we'll flag mismatches).

    ASSUMPTION: order_number is the correct and sufficient join key.
    The Order table may not cover 100% of orderline years (Order table spans
    unknown date range; OrderLine spans 2017–2026). We report unmatched orderlines
    rather than silently dropping them.
    """
    # Columns to bring from Order header (avoid duplicating fields that exist in OrderLine)
    order_cols = [
        "order_number",
        "order_date",           "invoice_date",
        "order_status",         "order_type",
        "warehouse_status",     "invoiced",
        "cust_number",          "bill_to_cust",
        "cust_ord_num",         "po_number",
        "salesman",             "ship_name",
        "ship_city",            "ship_prov",
        "ship_country",
        "warehouse",            "to_warehouse",
        "terms_code",           "price_level",
        "price_categ",          "entry_source",
        "order_code",           "carrier",
        "freight_chg",
        "order_value",          "contr_amount",
        "cogs",                 "credit_hold",
        "bo_number",            "bo_created",
        "order_month_start",
        "_source_file",
    ]
    # Only keep columns that exist
    order_cols = [c for c in order_cols if c in orders.columns]

    # Rename order-level cogs to avoid collision with line-level cogs
    orders_slim = orders[order_cols].copy()
    orders_slim = orders_slim.rename(columns={
        "cogs":         "order_cogs",
        "order_date":   "order_date_hdr",
        "invoice_date": "invoice_date_hdr",
        "warehouse":    "order_warehouse",
        "to_warehouse": "order_to_warehouse",
        "bo_number":    "order_bo_number",
        "_source_file": "_order_source_file",
    })

    # Left join: every orderline row is retained
    merged = orderlines.merge(
        orders_slim,
        on="order_number",
        how="left",
        suffixes=("", "_order"),
    )

    # ── Match diagnostics ─────────────────────────────
    unmatched = merged["order_status"].isna().sum()
    match_rate = 1 - unmatched / len(merged)
    print(f"\n  JOIN RESULTS:")
    print(f"    OrderLine rows:           {len(orderlines):,}")
    print(f"    Matched to Order header:  {len(merged) - unmatched:,}")
    print(f"    Unmatched (no header):    {unmatched:,}  ({unmatched/len(merged):.2%})")
    print(f"    Match rate:               {match_rate:.2%}")

    merged["_header_matched"] = merged["order_status"].notna().astype(int)

    return merged


# ─────────────────────────────────────────────
# SECTION 4 — VALIDATION
# ─────────────────────────────────────────────

def validate_order_union(df, row_counts):
    print("\n" + "="*60)
    print("VALIDATION — ORDER TABLE")
    print("="*60)
    print(f"\nRow counts by source file:")
    for f, n in row_counts.items():
        print(f"  {f}: {n:,}")
    print(f"  TOTAL: {sum(row_counts.values()):,}")
    print(f"  DataFrame rows: {len(df):,}  ✓" if len(df) == sum(row_counts.values()) else "  ❌ MISMATCH")

    # Duplicate order_number check
    dup_count = df["order_number"].duplicated().sum()
    print(f"\nDuplicate order_number values: {dup_count:,}")
    if dup_count > 0:
        print("  ⚠ WARN: order_number is not unique across Order files.")
        print("  This may mean the order table split on row count, overlapping dates.")
        dup_examples = df[df["order_number"].duplicated(keep=False)]["order_number"].unique()[:5]
        print(f"  Example duplicates: {list(dup_examples)}")

    # Null order_number
    null_keys = df["order_number"].isnull().sum()
    print(f"Null order_number: {null_keys:,}")

    # Date range
    print(f"\norder_date range: {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"invoice_date range: {df['invoice_date'].min()} → {df['invoice_date'].max()}")

    # Order value sanity
    print(f"\norder_value stats:")
    print(df["order_value"].describe().to_string())
    neg_val = (df["order_value"] < 0).sum()
    print(f"Negative order_value rows: {neg_val:,}")


def validate_orderline_union(df, row_counts):
    print("\n" + "="*60)
    print("VALIDATION — ORDERLINE TABLE")
    print("="*60)
    print(f"\nRow counts by source file:")
    for f, n in row_counts.items():
        print(f"  {f}: {n:,}")
    print(f"  TOTAL: {sum(row_counts.values()):,}")
    print(f"  DataFrame rows: {len(df):,}  ✓" if len(df) == sum(row_counts.values()) else "  ❌ MISMATCH")

    # Year coverage
    print(f"\nRows by acct_year:")
    yr_counts = df.groupby("acct_year").size().sort_index()
    print(yr_counts.to_string())

    # PK uniqueness
    dup_pk = df["ol_pk"].duplicated().sum()
    print(f"\nDuplicate (order_number|line_number|bo_number) keys: {dup_pk:,}")
    if dup_pk > 0:
        print("  ⚠ WARN: Composite PK is not unique — investigate.")

    # Null item_number
    null_items = (df["item_number"].isin(["nan","","None"]) | df["item_number"].isnull()).sum()
    print(f"Null/blank item_number: {null_items:,}")

    # shipped_qty distribution
    print(f"\nshipped_qty stats:")
    print(df["shipped_qty"].describe().to_string())
    zero_shipped = (df["shipped_qty"] == 0).sum()
    neg_shipped  = (df["shipped_qty"] < 0).sum()
    print(f"Zero shipped_qty rows: {zero_shipped:,}  ({zero_shipped/len(df):.2%})")
    print(f"Negative shipped_qty rows: {neg_shipped:,}")

    # Return / cancel flags
    print(f"\nRows with returned_qty > 0: {df['has_return'].sum():,}")
    print(f"Rows with cancel_qty > 0:   {df['has_cancellation'].sum():,}")
    print(f"Rows with substitution:      {df['is_substitution'].sum():,}")

    # Date coverage
    print(f"\nmonth_start range: {df['month_start'].min()} → {df['month_start'].max()}")
    print(f"order_date range:  {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"invoice_date range:{df['invoice_date'].min()} → {df['invoice_date'].max()}")
    
    # --- DEMAND MONTH VALIDATION ---
    print("\n--- DEMAND MONTH VALIDATION ---")
    print(f"demand_month range: {df['demand_month'].min()} → {df['demand_month'].max()}")

    null_demand_month = df["demand_month"].isna().sum()
    print(f"Null demand_month rows: {null_demand_month:,}")

    # sale_revenue sanity
    print(f"\nsale_revenue stats:")
    print(df["sale_revenue"].describe().to_string())
    neg_rev = (df["sale_revenue"] < 0).sum()
    print(f"Negative sale_revenue rows: {neg_rev:,}")


def validate_join(transaction_base, orderlines):
    print("\n" + "="*60)
    print("VALIDATION — JOIN COMPLETENESS")
    print("="*60)

    total_ol = len(orderlines)
    total_tx = len(transaction_base)
    matched  = transaction_base["_header_matched"].sum()
    unmatched = total_tx - matched

    print(f"\nOrderLine rows (input):         {total_ol:,}")
    print(f"Transaction base rows (output): {total_tx:,}")
    if total_ol != total_tx:
        print("  ❌ Row count changed during join — investigate fan-out!")
    else:
        print("  ✓ No fan-out; row count preserved")

    print(f"\nMatched to Order header:  {matched:,}  ({matched/total_tx:.2%})")
    print(f"Unmatched (no header):    {unmatched:,}  ({unmatched/total_tx:.2%})")

    if unmatched > 0:
        print("\n  Unmatched rows by acct_year:")
        unmatched_df = transaction_base[transaction_base["_header_matched"] == 0]
        print(unmatched_df.groupby("acct_year").size().to_string())


def validate_date_continuity(df):
    """Check for missing months in the timeline."""
    print("\n" + "="*60)
    print("VALIDATION — DATE CONTINUITY")
    print("="*60)
    months_present = df["month_start"].dropna().dt.to_period("M").unique()
    months_present = sorted(months_present)
    if len(months_present) == 0:
        print("  ❌ No valid month_start values found!")
        return
    full_range = pd.period_range(months_present[0], months_present[-1], freq="M")
    missing = set(full_range) - set(months_present)
    print(f"\nFirst month: {months_present[0]}")
    print(f"Last month:  {months_present[-1]}")
    print(f"Expected months: {len(full_range)}")
    print(f"Present months:  {len(months_present)}")
    if missing:
        print(f"Missing months ({len(missing)}): {sorted(missing)}")
    else:
        print("✓ No missing months in timeline")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LANE 7 ETL — TRANSACTION BASE BUILD")
    print("=" * 60)

    # ── 1. ORDERS ─────────────────────────────────────
    print("\n[1/5] Loading Order files...")
    raw_orders, order_row_counts = load_order_files(ORDER_FILES, DATA_DIR)

    print("\n[2/5] Cleaning Order table...")
    cleaned_orders = clean_order_table(raw_orders)

    
    # ── DEBUG: Order header duplication analysis ───────────

    print("\n--- ORDER DUPLICATION ANALYSIS ---")

    dups = cleaned_orders[cleaned_orders["order_number"].duplicated(keep=False)].copy()

    print(f"Total duplicate rows: {len(dups):,}")
    print(f"Unique duplicate order_numbers: {dups['order_number'].nunique():,}")

    # Sample rows
    print("\nSample duplicate rows:")
    print(
        dups[
            [
            "order_number",
            "order_date",
            "invoice_date",
            "order_status",
            "warehouse_status",
            "order_value",
            "_source_file",
        ]
    ]
    .sort_values(["order_number", "invoice_date", "order_date"])
    .head(20)
    .to_string(index=False)
)

    # Profile differences
    order_dup_profile = (
        dups.groupby("order_number")
        .agg(
            rows=("order_number", "size"),
            distinct_order_date=("order_date", "nunique"),
            distinct_invoice_date=("invoice_date", "nunique"),
            distinct_order_status=("order_status", "nunique"),
            distinct_warehouse_status=("warehouse_status", "nunique"),
            distinct_order_value=("order_value", "nunique"),
        )
        .sort_values("rows", ascending=False)
    )

    print("\nDuplicate profile (top 20):")
    print(order_dup_profile.head(20).to_string())

    # Source file overlap
    dup_sources = (
        dups.groupby("order_number")["_source_file"]
        .agg(lambda x: sorted(set(x)))
        .reset_index()
    )

    print("\nSource file overlap (top 20):")
    print(dup_sources.head(20).to_string(index=False))

    validate_order_union(cleaned_orders, order_row_counts)

    # ── 2. ORDERLINES ─────────────────────────────────
    print("\n[3/5] Loading OrderLine files...")
    raw_orderlines, ol_row_counts = load_orderline_files(ORDERLINE_FILES, DATA_DIR)

    print("\n[4/5] Cleaning OrderLine table...")
    cleaned_orderlines = clean_orderline_table(raw_orderlines)

    validate_orderline_union(cleaned_orderlines, ol_row_counts)

    # ── 3. JOIN ───────────────────────────────────────
    print("\n[5/5] Building transaction base (join Order + OrderLine)...")
    transaction_base = build_transaction_base(cleaned_orders, cleaned_orderlines)

    validate_join(transaction_base, cleaned_orderlines)
    validate_date_continuity(transaction_base)

    # ── 4. WRITE OUTPUTS ──────────────────────────────
    print("\n" + "="*60)
    print("WRITING OUTPUT FILES")
    print("="*60)

    order_out = os.path.join(OUTPUT_DIR, "cleaned_order_table.csv")
    ol_out    = os.path.join(OUTPUT_DIR, "cleaned_orderline_table.csv")
    tx_out    = os.path.join(OUTPUT_DIR, "transaction_base.csv")

    cleaned_orders.to_csv(order_out, index=False)
    print(f"  ✓ {order_out}  ({len(cleaned_orders):,} rows)")

    cleaned_orderlines.to_csv(ol_out, index=False)
    print(f"  ✓ {ol_out}  ({len(cleaned_orderlines):,} rows)")

    transaction_base.to_csv(tx_out, index=False)
    print(f"  ✓ {tx_out}  ({len(transaction_base):,} rows)")

    print("\n✅ ETL COMPLETE\n")
    print("Next steps:")
    print("  1. Review unmatched orderlines — determine if Order table coverage is intentionally partial")
    print("  2. Decide filter strategy for zero-shipped, cancelled, and returned rows")
    print("  3. Confirm 'item_number' is the final SKU (no Item/Style lookup needed?)")
    print("  4. Provide Item/Style lookup tables if style-level or color-level aggregation is needed")
    print("  5. Provide Customer lookup table if channel/segment enrichment is needed")

    return cleaned_orders, cleaned_orderlines, transaction_base


if __name__ == "__main__":
    orders, orderlines, tx_base = main()
