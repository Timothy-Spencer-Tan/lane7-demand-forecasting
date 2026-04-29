"""
Lane 7 Demand Forecasting — Transaction Base ETL
Updated to work with 7 OrderLine split files (was 17).
Adds dim_product rebuild from current lookup tables.

Outputs:
  1. cleaned_order_table.csv
  2. cleaned_orderline_table.csv
  3. transaction_base.csv
  4. gold_fact_monthly_demand_v2.csv
  5. dim_product.csv

Run from the directory containing all xlsx files.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR   = "."
OUTPUT_DIR = "./output"

ORDER_FILES = [
    "Lane7 Order Table 1 of 3.xlsx",
    "Lane7 Order Table 2 of 3.xlsx",
    "Lane7 Order Table 3 of 3.xlsx",
]

# Required lookup files for dim_product
ITEM_FILE       = "Lane7 Item Table.xlsx"
SIZE_FILE       = "Lane7 Size Table.xlsx"
STYLE_FILE      = "Lane7 Style Table.xlsx"
STYLECOLOR_FILE = "Lane7 StyleColor.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# SECTION 0 — DISCOVER ORDERLINE FILES
# ─────────────────────────────────────────────

def discover_orderline_files(data_dir):
    """
    Dynamically discover all OrderLine split files in data_dir.
    Matches pattern: "Lane 7 OrderLine*.xlsx" (note space, not underscore).
    Files are sorted numerically by the leading number in their filename so
    the load order is deterministic and transparent.

    ASSUMPTION: All files matching this pattern belong to the same logical
    OrderLine table and should be unioned in filename order.
    """
    pattern = os.path.join(data_dir, "Lane 7 OrderLine*.xlsx")
    found = sorted(glob.glob(pattern))

    if not found:
        # Fallback: also try underscore variant (some systems rename on download)
        pattern2 = os.path.join(data_dir, "Lane_7_OrderLine*.xlsx")
        found = sorted(glob.glob(pattern2))

    if not found:
        raise FileNotFoundError(
            f"No OrderLine files found matching 'Lane 7 OrderLine*.xlsx' "
            f"in '{data_dir}'. Check DATA_DIR and file naming."
        )

    print(f"\n  Discovered {len(found)} OrderLine file(s):")
    for f in found:
        print(f"    {os.path.basename(f)}")

    return found


# ─────────────────────────────────────────────
# SECTION 1 — LOAD & UNION ORDER TABLE
# ─────────────────────────────────────────────

def load_order_files(files, data_dir):
    """Load and union all Order split files. Track provenance per row."""
    dfs = []
    row_counts = {}
    for fname in files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Order file not found: {path}")
        print(f"  Loading: {fname}")
        df = pd.read_excel(path, sheet_name="Sheet1")
        row_counts[fname] = len(df)
        df["_source_file"] = fname
        dfs.append(df)
        print(f"    → {len(df):,} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total order rows after union: {len(combined):,}")
    assert len(combined) == sum(row_counts.values()), "ROW COUNT MISMATCH after order union!"
    return combined, row_counts


def clean_order_table(df):
    """
    Standardize and clean the unified Order table.

    KEY ASSUMPTIONS:
    - 'invoice-date' is the primary demand timing signal
    - 'order-number' is the join key to OrderLine (deduplicated before joining)
    - 'OrderValue' is the corrected/final order value (preferred over 'order-value')
    - 'order-status' = 'C' = Completed; 'O' = Open; 'B' = Backordered
    - 'company' is always 'LSA' — single-company dataset
    - Duplicate order_number rows arise from the Order table being split by row count
      (not by date) and are resolved by keeping the row with the latest invoice_date
    """
    # ── Date parsing ──────────────────────────────────
    date_cols = ["order-date", "invoice-date", "created-date", "plan-ship-dt", "bo-created"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Rename to snake_case ──────────────────────────
    rename_map = {
        "order-number":          "order_number",
        "order-date":            "order_date",
        "invoice-num":           "invoice_num",
        "invoice-date":          "invoice_date",
        "invoice-total":         "invoice_total",
        "order-value":           "order_value_raw",
        "OrderValue":            "order_value",
        "OrderBoNo":             "order_bo_key",
        "order-status":          "order_status",
        "order-type":            "order_type",
        "warehouse-status":      "warehouse_status",
        "cust-number":           "cust_number",
        "cust-ord-num":          "cust_ord_num",
        "bill-to-cust":          "bill_to_cust",
        "acct-year":             "acct_year",
        "acct-period":           "acct_period",
        "acct-yp":               "acct_yp",
        "order-created":         "order_created",
        "created-date":          "created_date",
        "plan-ship-dt":          "plan_ship_dt",
        "ship-name":             "ship_name",
        "ship-addr":             "ship_addr",
        "ship-city":             "ship_city",
        "ship-prov":             "ship_prov",
        "ship-pcode":            "ship_pcode",
        "ship-country":          "ship_country",
        "salesman":              "salesman",
        "warehouse":             "warehouse",
        "to-warehouse":          "to_warehouse",
        "terms-code":            "terms_code",
        "price-level":           "price_level",
        "price-categ":           "price_categ",
        "entry-source":          "entry_source",
        "order-code":            "order_code",
        "carrier":               "carrier",
        "freight-chg":           "freight_chg",
        "cogs":                  "cogs",
        "contr-amount":          "contr_amount",
        "invoiced":              "invoiced",
        "credit-hold":           "credit_hold",
        "bo-number":             "bo_number",
        "bo-created":            "bo_created",
        "pick-slip-print-count": "pick_slip_print_count",
        "po-number":             "po_number",
        "addr-number":           "addr_number",
        "attn":                  "attn",
        "description":           "description",
        "company":               "company",
        "container_ship":        "container_ship",
        "user_id":               "user_id",
        "assignee":              "assignee",
        "cust-ref":              "cust_ref",
        "ref-comp-number":       "ref_comp_number",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── Type coercion ──────────────────────────────────
    df["order_number"] = df["order_number"].astype(str).str.strip()
    df["cust_number"]  = pd.to_numeric(df["cust_number"], errors="coerce")
    df["acct_year"]    = pd.to_numeric(df.get("acct_year"),  errors="coerce").astype("Int64")
    df["acct_period"]  = pd.to_numeric(df.get("acct_period"), errors="coerce").astype("Int64")
    df["order_value"]  = pd.to_numeric(df.get("order_value"), errors="coerce")
    df["cogs"]         = pd.to_numeric(df.get("cogs"),         errors="coerce")
    df["contr_amount"] = pd.to_numeric(df.get("contr_amount"), errors="coerce")

    # ── Derived columns ───────────────────────────────
    df["order_month_start"] = df["order_date"].dt.to_period("M").dt.to_timestamp()

    return df


# ─────────────────────────────────────────────
# SECTION 2 — LOAD & UNION ORDERLINE TABLE
# ─────────────────────────────────────────────

def load_orderline_files(files):
    """Load and union all discovered OrderLine files. Track provenance."""
    dfs = []
    row_counts = {}
    for path in files:
        fname = os.path.basename(path)
        print(f"  Loading: {fname}")
        df = pd.read_excel(path, sheet_name="Sheet1")
        row_counts[fname] = len(df)
        df["_source_file"] = fname

        # Report acct-year coverage per file (key QA signal)
        if "acct-year" in df.columns:
            years = sorted(df["acct-year"].dropna().unique().astype(int))
            print(f"    → {len(df):,} rows | acct_year: {years}")
        else:
            print(f"    → {len(df):,} rows")

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total orderline rows after union: {len(combined):,}")
    print(f"  Sum of individual files:          {sum(row_counts.values()):,}")
    assert len(combined) == sum(row_counts.values()), "ROW COUNT MISMATCH after orderline union!"
    return combined, row_counts


def clean_orderline_table(df):
    """
    Standardize and clean the unified OrderLine table.

    KEY ASSUMPTIONS:
    - Composite PK = (order_number, line_number, bo_number)
    - 'shipped-qty' is the primary units-sold metric
    - 'demand_month' is derived from 'invoice-date' (fiscal demand signal)
    - No MonthStart column exists in the current OrderLine files; we derive
      accounting month from acct-year + acct-period as a fallback for demand_month
      when invoice-date is null
    - 'item-number' is the SKU (primary product key)
    - 'sale-revenue' = line revenue (price × shipped-qty minus discounts)
    - bo_number = 0 is the original line; > 0 is a backorder variant
    """
    # ── Date parsing ──────────────────────────────────
    for col in ["invoice-date", "order-date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── Primary demand timing ─────────────────────────
    # Use invoice-date month as the demand signal (actual fulfillment date)
    df["demand_month"] = df["invoice-date"].dt.to_period("M").dt.to_timestamp()

    # Fallback: derive month from acct-year + acct-period if invoice-date missing
    # ASSUMPTION: acct-period = calendar month number (1–12) within acct-year
    if "acct-year" in df.columns and "acct-period" in df.columns:
        acct_mask = df["demand_month"].isna()
        if acct_mask.any():
            fallback = pd.to_datetime(
                df.loc[acct_mask, "acct-year"].astype(str)
                + "-"
                + df.loc[acct_mask, "acct-period"].astype(str).str.zfill(2)
                + "-01",
                errors="coerce",
            )
            df.loc[acct_mask, "demand_month"] = fallback

    # ── Rename to snake_case ──────────────────────────
    rename_map = {
        "order-number":    "order_number",
        "line-number":     "line_number",
        "bo-number":       "bo_number",
        "item-number":     "item_number",
        "original-item":   "original_item",
        "description":     "description",
        "item-categ":      "item_categ",
        "cust-number":     "cust_number",
        "cust-item-name":  "cust_item_name",
        "order-date":      "order_date",
        "invoice-date":    "invoice_date",
        "acct-year":       "acct_year",
        "acct-period":     "acct_period",
        "acct-yp":         "acct_yp",
        "order-qty":       "order_qty",
        "shipped-qty":     "shipped_qty",
        "cancel-qty":      "cancel_qty",
        "returned-qty":    "returned_qty",
        "backordered":     "backordered",
        "action-qty":      "action_qty",
        "action-code":     "action_code",
        "actual-qty":      "actual_qty",
        "parent-qty":      "parent_qty",
        "sale-revenue":    "sale_revenue",
        "amount":          "amount",
        "price":           "price",
        "line-discount":   "line_discount",
        "discount_type":   "discount_type",
        "cogs":            "cogs",
        "unit-cost":       "unit_cost",
        "std-cost":        "std_cost",
        "est-cost":        "est_cost",
        "contract-cost":   "contract_cost",
        "volume":          "volume",
        "warehouse":       "warehouse",
        "to-warehouse":    "to_warehouse",
        "order-type":      "order_type",
        "company":         "company",
        "ref-comp-number": "ref_comp_number",
        "ref-line-number": "ref_line_number",
        "ref-bo-number":   "ref_bo_number",
        "OrdLineAmount":   "ord_line_amount",
        "ShipQty":         "ship_qty_alt",
        "OrderBoNo":       "order_bo_key",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── Type coercion ──────────────────────────────────
    df["order_number"]  = df["order_number"].astype(str).str.strip()
    df["item_number"]   = df["item_number"].astype(str).str.strip()
    df["original_item"] = df["original_item"].astype(str).str.strip()
    df["cust_number"]   = df["cust_number"].astype(str).str.strip()
    df["line_number"]   = pd.to_numeric(df.get("line_number"),  errors="coerce").astype("Int64")
    df["bo_number"]     = pd.to_numeric(df.get("bo_number"),    errors="coerce").astype("Int64")
    df["acct_year"]     = pd.to_numeric(df.get("acct_year"),    errors="coerce").astype("Int64")
    df["acct_period"]   = pd.to_numeric(df.get("acct_period"),  errors="coerce").astype("Int64")

    qty_cols = ["order_qty", "shipped_qty", "cancel_qty", "returned_qty",
                "action_qty", "actual_qty", "parent_qty", "backordered"]
    for col in qty_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    rev_cols = ["sale_revenue", "amount", "price", "line_discount", "cogs",
                "unit_cost", "std_cost", "contract_cost", "volume"]
    for col in rev_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Normalize item_categ (mixed case across years) ─
    if "item_categ" in df.columns:
        df["item_categ"] = df["item_categ"].astype(str).str.strip().str.title()

    # ── Composite PK ──────────────────────────────────
    df["ol_pk"] = (
        df["order_number"].astype(str)
        + "|" + df["line_number"].astype(str)
        + "|" + df["bo_number"].astype(str)
    )

    # ── Derived flags ─────────────────────────────────
    df["is_substitution"]   = (df["item_number"] != df["original_item"]).astype(int)
    df["has_return"]        = (df["returned_qty"] > 0).astype(int)
    df["has_cancellation"]  = (df["cancel_qty"]   > 0).astype(int)
    df["has_zero_shipped"]  = (df["shipped_qty"]  == 0).astype(int)

    return df


# ─────────────────────────────────────────────
# SECTION 3 — JOIN ORDER + ORDERLINE
# ─────────────────────────────────────────────

def dedupe_order_header(df):
    """
    Deduplicate Order header by order_number before joining.
    The Order files were split by row count (not by date), so the same
    order_number can appear in multiple files. We keep the row with the
    latest invoice_date, breaking ties with the latest order_date.

    This prevents fan-out in the OrderLine JOIN.
    """
    before = len(df)
    deduped = (
        df.sort_values(
            ["order_number", "invoice_date", "order_date"],
            ascending=[True, True, True],
            na_position="first",
        )
        .drop_duplicates(subset=["order_number"], keep="last")
        .copy()
    )
    after = len(deduped)
    print(f"  Raw order rows:    {before:,}")
    print(f"  Deduped rows:      {after:,}")
    print(f"  Rows removed:      {before - after:,}")
    return deduped


def build_transaction_base(orders, orderlines):
    """
    LEFT JOIN OrderLine → Order header on order_number.
    OrderLine drives; every orderline row is preserved.
    Unmatched rows (no header) are flagged via _header_matched.
    """
    order_cols = [
        "order_number", "order_date", "invoice_date",
        "order_status", "order_type", "warehouse_status", "invoiced",
        "cust_number", "bill_to_cust", "cust_ord_num", "po_number",
        "salesman", "ship_name", "ship_city", "ship_prov", "ship_country",
        "warehouse", "to_warehouse", "terms_code", "price_level",
        "price_categ", "entry_source", "order_code", "carrier",
        "freight_chg", "order_value", "contr_amount", "cogs",
        "credit_hold", "bo_number", "bo_created", "order_month_start",
        "_source_file",
    ]
    order_cols = [c for c in order_cols if c in orders.columns]
    orders_slim = orders[order_cols].copy().rename(columns={
        "cogs":         "order_cogs",
        "order_date":   "order_date_hdr",
        "invoice_date": "invoice_date_hdr",
        "warehouse":    "order_warehouse",
        "to_warehouse": "order_to_warehouse",
        "bo_number":    "order_bo_number",
        "_source_file": "_order_source_file",
    })

    merged = orderlines.merge(orders_slim, on="order_number", how="left", suffixes=("", "_order"))

    unmatched = merged["order_status"].isna().sum() if "order_status" in merged.columns else 0
    match_rate = 1 - unmatched / len(merged)
    print(f"\n  JOIN RESULTS:")
    print(f"    OrderLine rows:           {len(orderlines):,}")
    print(f"    Matched to Order header:  {len(merged) - unmatched:,}")
    print(f"    Unmatched (no header):    {unmatched:,}  ({unmatched / len(merged):.2%})")
    print(f"    Match rate:               {match_rate:.2%}")

    merged["_header_matched"] = merged["order_status"].notna().astype(int) if "order_status" in merged.columns else 0
    return merged


# ─────────────────────────────────────────────
# SECTION 4 — DIM_PRODUCT
# ─────────────────────────────────────────────

def build_dim_product(data_dir):
    """
    Build dim_product at SKU (item_number) grain.

    Source files and join logic:
      Item Table    → base; contains item_number, size-code, color-code, style-code,
                       item-name, item-categ, StyleColor
      Size Table    → size description; joined on size-code (string '01' vs int 1 — normalized)
      Style Table   → style description and web fields; joined on style-code
      StyleColor    → color description; joined on StyleColor key

    ASSUMPTION: item-number is unique in the Item Table (confirmed: 3,806 rows, 3,806 unique).
    ASSUMPTION: size-code in Item Table is zero-padded string ('01'); in Size Table it is int.
                We cast both to int for the join.
    """
    item_path  = os.path.join(data_dir, ITEM_FILE)
    size_path  = os.path.join(data_dir, SIZE_FILE)
    style_path = os.path.join(data_dir, STYLE_FILE)
    sc_path    = os.path.join(data_dir, STYLECOLOR_FILE)

    for p, label in [(item_path, "Item"), (size_path, "Size"),
                     (style_path, "Style"), (sc_path, "StyleColor")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} lookup file not found: {p}")

    item     = pd.read_excel(item_path,  sheet_name="Sheet1")
    size     = pd.read_excel(size_path,  sheet_name="Sheet1")
    style    = pd.read_excel(style_path, sheet_name="Sheet1")
    sc       = pd.read_excel(sc_path,    sheet_name="Sheet1")

    print(f"  Item rows:       {len(item):,}")
    print(f"  Size rows:       {len(size):,}")
    print(f"  Style rows:      {len(style):,}")
    print(f"  StyleColor rows: {len(sc):,}")

    # ── Normalize join keys ──────────────────────────
    # Item: size-code is '01', '02' … — cast to int for join with Size table
    item["_size_join"] = pd.to_numeric(item["size-code"], errors="coerce").astype("Int64")
    size["_size_join"] = pd.to_numeric(size["Size-code"], errors="coerce").astype("Int64")

    # ── Join Size ─────────────────────────────────────
    size_slim = size[["_size_join", "Size Desc"]].rename(columns={"Size Desc": "size_desc"})
    dim = item.merge(size_slim, on="_size_join", how="left")

    # ── Join StyleColor → color description ───────────
    sc_slim = sc[["StyleColor", "description", "color-code"]].rename(columns={
        "description": "color_desc",
        "color-code":  "color_code_sc",
    })
    dim = dim.merge(sc_slim, on="StyleColor", how="left")

    # ── Join Style → style description, web fields ────
    style_slim = style[["style-code", "description", "web_description",
                          "web-active", "item-status", "StyleCodeDesc"]].rename(columns={
        "description":    "style_desc",
        "web-active":     "web_active",
        "item-status":    "item_status",
        "StyleCodeDesc":  "style_code_desc",
    })
    dim = dim.merge(style_slim, on="style-code", how="left")

    # ── Rename and select final columns ───────────────
    dim = dim.rename(columns={
        "item-number": "SKU",
        "item-name":   "item_name",
        "item-categ":  "item_categ",
        "size-code":   "size_code",
        "color-code":  "color_code",
        "style-code":  "style_code",
        "StyleColor":  "style_color_key",
    })

    output_cols = [
        "SKU",
        "item_name",
        "item_categ",
        "style_code",
        "style_desc",
        "style_code_desc",
        "color_code",
        "color_desc",
        "size_code",
        "size_desc",
        "style_color_key",
        "web_description",
        "web_active",
        "item_status",
    ]
    output_cols = [c for c in output_cols if c in dim.columns]
    dim = dim[output_cols].drop_duplicates(subset=["SKU"])

    # Validate
    assert dim["SKU"].nunique() == len(dim), "dim_product has duplicate SKU rows — investigate!"
    print(f"  dim_product rows: {len(dim):,}  (unique SKUs: {dim['SKU'].nunique():,})")

    return dim


# ─────────────────────────────────────────────
# SECTION 5 — VALIDATION
# ─────────────────────────────────────────────

def validate_order_union(df, row_counts):
    print("\n" + "=" * 60)
    print("VALIDATION — ORDER TABLE")
    print("=" * 60)
    for f, n in row_counts.items():
        print(f"  {f}: {n:,}")
    total = sum(row_counts.values())
    status = "✓" if len(df) == total else "❌ MISMATCH"
    print(f"  TOTAL: {total:,}  |  DataFrame: {len(df):,}  {status}")

    dup_count = df["order_number"].duplicated().sum()
    print(f"\nDuplicate order_number: {dup_count:,}")
    if dup_count > 0:
        print("  ⚠ WARN: order_number not unique across Order files (expected — split by row count)")
        examples = df[df["order_number"].duplicated(keep=False)]["order_number"].unique()[:5]
        print(f"  Examples: {list(examples)}")

    print(f"Null order_number: {df['order_number'].isnull().sum():,}")
    print(f"order_date range:   {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"invoice_date range: {df['invoice_date'].min()} → {df['invoice_date'].max()}")
    print(f"\norder_value stats:\n{df['order_value'].describe().to_string()}")
    print(f"Negative order_value rows: {(df['order_value'] < 0).sum():,}")


def validate_orderline_union(df, row_counts):
    print("\n" + "=" * 60)
    print("VALIDATION — ORDERLINE TABLE")
    print("=" * 60)
    for f, n in row_counts.items():
        print(f"  {f}: {n:,}")
    total = sum(row_counts.values())
    status = "✓" if len(df) == total else "❌ MISMATCH"
    print(f"  TOTAL: {total:,}  |  DataFrame: {len(df):,}  {status}")

    print(f"\nRows by acct_year:\n{df.groupby('acct_year').size().sort_index().to_string()}")

    dup_pk = df["ol_pk"].duplicated().sum()
    print(f"\nDuplicate composite PK (order|line|bo): {dup_pk:,}")
    if dup_pk > 0:
        print("  ⚠ WARN: Composite PK not unique — investigate cross-file overlap")

    null_items = (df["item_number"].isin(["nan", "", "None"]) | df["item_number"].isnull()).sum()
    print(f"Null/blank item_number: {null_items:,}")

    print(f"\nshipped_qty stats:\n{df['shipped_qty'].describe().to_string()}")
    print(f"Zero shipped_qty rows:     {(df['shipped_qty'] == 0).sum():,}  ({(df['shipped_qty'] == 0).mean():.2%})")
    print(f"Negative shipped_qty rows: {(df['shipped_qty'] < 0).sum():,}")

    print(f"\nRows with returned_qty > 0:  {df['has_return'].sum():,}")
    print(f"Rows with cancel_qty > 0:    {df['has_cancellation'].sum():,}")
    print(f"Rows with substitution:       {df['is_substitution'].sum():,}")

    print(f"\norder_date range:    {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"invoice_date range:  {df['invoice_date'].min()} → {df['invoice_date'].max()}")
    print(f"\n--- DEMAND MONTH VALIDATION ---")
    print(f"demand_month range:  {df['demand_month'].min()} → {df['demand_month'].max()}")
    print(f"Null demand_month rows: {df['demand_month'].isna().sum():,}")

    print(f"\nsale_revenue stats:\n{df['sale_revenue'].describe().to_string()}")
    print(f"Negative sale_revenue rows: {(df['sale_revenue'] < 0).sum():,}")


def validate_join(tx, orderlines):
    print("\n" + "=" * 60)
    print("VALIDATION — JOIN COMPLETENESS")
    print("=" * 60)
    total_ol = len(orderlines)
    total_tx = len(tx)
    if total_ol != total_tx:
        print(f"  ❌ Fan-out detected! OrderLine: {total_ol:,} → TX: {total_tx:,}")
    else:
        print(f"  ✓ No fan-out; row count preserved: {total_tx:,}")

    matched   = tx["_header_matched"].sum()
    unmatched = total_tx - matched
    print(f"\nMatched to Order header:  {matched:,}  ({matched / total_tx:.2%})")
    print(f"Unmatched (no header):    {unmatched:,}  ({unmatched / total_tx:.2%})")

    if unmatched > 0:
        print("\nUnmatched rows by acct_year:")
        print(tx[tx["_header_matched"] == 0].groupby("acct_year").size().to_string())


def validate_date_continuity(df):
    print("\n" + "=" * 60)
    print("VALIDATION — DATE CONTINUITY (demand_month)")
    print("=" * 60)
    months_present = df["demand_month"].dropna().dt.to_period("M").unique()
    months_present = sorted(months_present)
    if not months_present:
        print("  ❌ No valid demand_month values found!")
        return
    full_range = pd.period_range(months_present[0], months_present[-1], freq="M")
    missing = sorted(set(full_range) - set(months_present))
    print(f"First month: {months_present[0]}")
    print(f"Last month:  {months_present[-1]}")
    print(f"Expected months: {len(full_range)} | Present: {len(months_present)}")
    if missing:
        print(f"Missing months ({len(missing)}): {missing}")
    else:
        print("✓ No missing months in timeline")


def validate_gold(gold):
    print("\n" + "=" * 60)
    print("VALIDATION — GOLD DEMAND TABLE")
    print("=" * 60)
    print(f"Rows: {len(gold):,}")
    print(f"Unique SKUs:     {gold['SKU'].nunique():,}")
    print(f"Date range:      {gold['MonthStart'].min()} → {gold['MonthStart'].max()}")
    print(f"Total UnitsSold: {gold['UnitsSold'].sum():,.0f}")
    print(f"Total Revenue:   ${gold['Revenue'].sum():,.2f}")
    dup_grain = gold.duplicated(subset=["MonthStart", "SKU"]).sum()
    print(f"Duplicate (MonthStart, SKU) rows: {dup_grain:,}")
    if dup_grain > 0:
        print("  ❌ Gold grain not unique — investigate aggregation!")
    else:
        print("  ✓ Gold grain is unique (1 row per SKU per month)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LANE 7 ETL — TRANSACTION BASE BUILD")
    print("=" * 60)

    # ── 1. DISCOVER ORDERLINE FILES ─────────────────────
    print("\n[0] Discovering OrderLine files...")
    ol_files = discover_orderline_files(DATA_DIR)

    # ── 2. ORDERS ──────────────────────────────────────
    print("\n[1/6] Loading Order files...")
    raw_orders, order_row_counts = load_order_files(ORDER_FILES, DATA_DIR)

    print("\n[2/6] Cleaning Order table...")
    cleaned_orders = clean_order_table(raw_orders)
    validate_order_union(cleaned_orders, order_row_counts)

    # ── 3. ORDERLINES ──────────────────────────────────
    print("\n[3/6] Loading OrderLine files...")
    raw_orderlines, ol_row_counts = load_orderline_files(ol_files)

    print("\n[4/6] Cleaning OrderLine table...")
    cleaned_orderlines = clean_orderline_table(raw_orderlines)
    validate_orderline_union(cleaned_orderlines, ol_row_counts)

    # ── 4. DEDUPE + JOIN ────────────────────────────────
    print("\n[5/6] Deduplicating Order header...")
    orders_dedup = dedupe_order_header(cleaned_orders)

    print("\n[5/6] Building transaction base (OrderLine LEFT JOIN Order)...")
    transaction_base = build_transaction_base(orders_dedup, cleaned_orderlines)
    validate_join(transaction_base, cleaned_orderlines)
    validate_date_continuity(transaction_base)

    # ── 5. GOLD DEMAND TABLE ────────────────────────────
    print("\n[6/6] Building gold_fact_monthly_demand_v2...")

    EXCLUDE_PATTERNS = "FREIGHT|LTL|SHIPPING|MISC"

    demand_df = transaction_base[
        (transaction_base["shipped_qty"] > 0)
        &
        (
            ~transaction_base["item_number"]
            .astype(str)
            .str.upper()
            .str.contains(EXCLUDE_PATTERNS, na=False)
        )
    ].copy()

    excluded_rows = transaction_base[
        (transaction_base["shipped_qty"] > 0)
        &
        transaction_base["item_number"]
        .astype(str)
        .str.upper()
        .str.contains(EXCLUDE_PATTERNS, na=False)
    ]

    print(f"Excluded non-product rows: {len(excluded_rows):,}")
    print(f"Excluded revenue: ${excluded_rows['sale_revenue'].sum():,.2f}")
    print(f"Excluded unique pseudo-SKUs: {excluded_rows['item_number'].nunique():,}")

    gold = (
        demand_df
        .groupby(["demand_month", "item_number"], as_index=False)
        .agg(UnitsSold=("shipped_qty", "sum"), Revenue=("sale_revenue", "sum"))
        .rename(columns={"item_number": "SKU", "demand_month": "MonthStart"})
    )
    validate_gold(gold)

    # ── 6. DIM_PRODUCT ──────────────────────────────────
    print("\n[6/6] Building dim_product...")
    dim_product = build_dim_product(DATA_DIR)

    # ── 7. WRITE OUTPUTS ────────────────────────────────
    print("\n" + "=" * 60)
    print("WRITING OUTPUT FILES")
    print("=" * 60)

    outputs = [
        (cleaned_orders,       os.path.join(OUTPUT_DIR, "cleaned_order_table.csv")),
        (cleaned_orderlines,   os.path.join(OUTPUT_DIR, "cleaned_orderline_table.csv")),
        (transaction_base,     os.path.join(OUTPUT_DIR, "transaction_base.csv")),
        (gold,                 os.path.join(OUTPUT_DIR, "gold_fact_monthly_demand_v2.csv")),
        (dim_product,          os.path.join(OUTPUT_DIR, "dim_product.csv")),
    ]

    for df_out, path in outputs:
        df_out.to_csv(path, index=False)
        print(f"  ✓ {path}  ({len(df_out):,} rows)")

    print("\n✅ ETL COMPLETE\n")
    print("Post-run checks:")
    print("  1. Review unmatched orderlines — are they from years outside Order table coverage?")
    print("  2. Confirm acct_year range in gold matches expected historical span")
    print("  3. Check for SKUs in gold_fact that are missing from dim_product (orphaned items)")
    print("  4. Verify date continuity — any gaps in demand_month?")
    print("  5. Spot-check UnitsSold vs known totals for a few key months")

    return cleaned_orders, cleaned_orderlines, transaction_base, gold, dim_product


if __name__ == "__main__":
    main()
