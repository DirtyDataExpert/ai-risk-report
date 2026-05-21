#!/usr/bin/env python3
"""
AI Risk Report Generator
Author: Gary Segal / Dirty Data Expert

Purpose:
Reads a CSV file and produces an executive AI Risk Report showing:
- Executive Summary
- Segmentation Risk Indicators
- Dirty Data Exposure Score
- Operational data risks that may distort AI outputs, recommendations, and decisions

Run:
    python ai_risk_report.py
"""

import os
import re
import sys
from datetime import datetime

import pandas as pd


# -----------------------------
# Utility functions
# -----------------------------

def clean_col_name(col):
    return str(col).strip().lower().replace(" ", "_")


def find_likely_columns(df):
    """
    Attempts to identify likely segmentation-related columns.
    This is intentionally flexible because most operational CSV files are messy.
    """
    column_map = {
        "sku": [],
        "product": [],
        "category": [],
        "customer": [],
        "location": [],
        "date": [],
        "quantity": [],
        "sales": [],
        "price": []
    }

    patterns = {
        "sku": ["sku", "item", "barcode", "stock_code", "product_code"],
        "product": ["product", "description", "item_name", "product_name"],
        "category": ["category", "department", "segment", "class", "group", "family"],
        "customer": ["customer", "client", "account"],
        "location": ["store", "branch", "region", "location"],
        "date": ["date", "month", "period", "invoice_date", "transaction_date"],
        "quantity": ["qty", "quantity", "units", "volume"],
        "sales": ["sales", "revenue", "turnover", "amount", "value"],
        "price": ["price", "unit_price", "selling_price"]
    }

    for col in df.columns:
        c = clean_col_name(col)
        for key, words in patterns.items():
            if any(word in c for word in words):
                column_map[key].append(col)

    return column_map


def calculate_dirty_data_metrics(df):
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols if total_rows and total_cols else 1

    missing_cells = int(df.isna().sum().sum())
    missing_pct = round((missing_cells / total_cells) * 100, 2)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows else 0

    blank_or_unknown_count = 0
    text_cols = df.select_dtypes(include=["object"]).columns

    dirty_tokens = [
        "", " ", "na", "n/a", "none", "null", "unknown", "misc",
        "miscellaneous", "other", "tbc", "to be confirmed", "undefined"
    ]

    for col in text_cols:
        values = df[col].astype(str).str.strip().str.lower()
        blank_or_unknown_count += int(values.isin(dirty_tokens).sum())

    blank_or_unknown_pct = round((blank_or_unknown_count / total_cells) * 100, 2)

    inconsistent_text_cols = []
    for col in text_cols:
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue

        leading_trailing_spaces = sample.str.contains(r"^\s+|\s+$", regex=True).sum()
        mixed_case = sample.nunique() != sample.str.lower().nunique()

        if leading_trailing_spaces > 0 or mixed_case:
            inconsistent_text_cols.append(col)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
        "blank_or_unknown_count": blank_or_unknown_count,
        "blank_or_unknown_pct": blank_or_unknown_pct,
        "inconsistent_text_cols": inconsistent_text_cols,
    }


def calculate_segmentation_risk(df, likely_cols):
    risks = []
    score = 0

    category_cols = likely_cols["category"]
    sku_cols = likely_cols["sku"]
    product_cols = likely_cols["product"]
    customer_cols = likely_cols["customer"]
    sales_cols = likely_cols["sales"]
    qty_cols = likely_cols["quantity"]

    if not category_cols:
        risks.append("No clear product/category/department segmentation column detected.")
        score += 20

    if not sku_cols:
        risks.append("No clear SKU/item/product-code column detected.")
        score += 20

    if not product_cols:
        risks.append("No clear product description/name column detected.")
        score += 10

    if not customer_cols:
        risks.append("No customer/client/account segmentation column detected.")
        score += 10

    if not sales_cols and not qty_cols:
        risks.append("No clear demand signal detected, such as sales, revenue, quantity, or units.")
        score += 20

    # Check category concentration and vague categories
    vague_terms = ["misc", "miscellaneous", "other", "general", "unknown", "undefined", "unclassified"]

    for col in category_cols:
        series = df[col].dropna().astype(str).str.strip().str.lower()
        if len(series) == 0:
            risks.append(f"Segmentation column '{col}' is mostly empty.")
            score += 15
            continue

        unique_count = series.nunique()
        vague_count = series.isin(vague_terms).sum()
        vague_pct = (vague_count / len(series)) * 100 if len(series) else 0

        if unique_count <= 3 and len(df) > 100:
            risks.append(f"Column '{col}' has very few categories for the size of the file.")
            score += 15

        if vague_pct > 5:
            risks.append(f"Column '{col}' contains a high share of vague categories such as Other/Misc/Unknown.")
            score += 15

    # SKU uniqueness check
    for col in sku_cols:
        sku_null_pct = df[col].isna().mean() * 100
        if sku_null_pct > 5:
            risks.append(f"SKU column '{col}' has more than 5% missing values.")
            score += 15

    return min(score, 100), risks


def calculate_dirty_data_exposure_score(metrics, segmentation_score):
    score = 0

    score += min(metrics["missing_pct"] * 2.0, 25)
    score += min(metrics["duplicate_pct"] * 2.0, 20)
    score += min(metrics["blank_or_unknown_pct"] * 3.0, 20)
    score += min(len(metrics["inconsistent_text_cols"]) * 3.0, 15)
    score += segmentation_score * 0.20

    return round(min(score, 100), 1)


def risk_band(score):
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MODERATE"
    return "LOW"


def generate_report(df, file_path):
    likely_cols = find_likely_columns(df)
    metrics = calculate_dirty_data_metrics(df)
    segmentation_score, segmentation_risks = calculate_segmentation_risk(df, likely_cols)
    exposure_score = calculate_dirty_data_exposure_score(metrics, segmentation_score)
    band = risk_band(exposure_score)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    report = []
    report.append("=" * 80)
    report.append("AI RISK REPORT")
    report.append("=" * 80)
    report.append("A structured view of the operational data risks that may distort AI outputs,")
    report.append("recommendations, and decisions.")
    report.append("")
    report.append(f"File analysed: {file_path}")
    report.append(f"Date generated: {now}")
    report.append("")

    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 80)
    report.append(
        f"The file shows a Dirty Data Exposure Score of {exposure_score}/100, "
        f"placing it in the {band} AI risk band."
    )

    if band in ["CRITICAL", "HIGH"]:
        report.append(
            "This means AI models built directly on this data may produce distorted forecasts, "
            "misleading recommendations, poor segmentation, and false confidence at executive level."
        )
    elif band == "MODERATE":
        report.append(
            "This means the data may be usable for early analysis, but should be cleaned, segmented, "
            "and governed before being trusted for automated AI recommendations."
        )
    else:
        report.append(
            "This means the file shows relatively low visible data-quality risk, although business-rule "
            "validation and deeper segmentation testing are still recommended."
        )

    report.append("")
    report.append("DIRTY DATA EXPOSURE SCORE")
    report.append("-" * 80)
    report.append(f"Score: {exposure_score}/100")
    report.append(f"Risk Band: {band}")
    report.append("")
    report.append("Score drivers:")
    report.append(f"- Rows analysed: {metrics['total_rows']}")
    report.append(f"- Columns analysed: {metrics['total_cols']}")
    report.append(f"- Missing cells: {metrics['missing_cells']} ({metrics['missing_pct']}%)")
    report.append(f"- Duplicate rows: {metrics['duplicate_rows']} ({metrics['duplicate_pct']}%)")
    report.append(f"- Blank/Unknown/Misc values: {metrics['blank_or_unknown_count']} ({metrics['blank_or_unknown_pct']}%)")
    report.append(f"- Text columns with possible inconsistency: {len(metrics['inconsistent_text_cols'])}")

    if metrics["inconsistent_text_cols"]:
        report.append("")
        report.append("Columns with possible text inconsistency:")
        for col in metrics["inconsistent_text_cols"][:20]:
            report.append(f"- {col}")

    report.append("")
    report.append("SEGMENTATION RISK INDICATORS")
    report.append("-" * 80)

    report.append("Detected segmentation-related fields:")
    for key, cols in likely_cols.items():
        if cols:
            report.append(f"- {key.title()}: {', '.join(cols)}")
        else:
            report.append(f"- {key.title()}: Not detected")

    report.append("")
    report.append(f"Segmentation Risk Score: {segmentation_score}/100")

    if segmentation_risks:
        report.append("Key segmentation risks:")
        for risk in segmentation_risks:
            report.append(f"- {risk}")
    else:
        report.append("No major visible segmentation risks detected from column structure.")

    report.append("")
    report.append("OPERATIONAL AI RISK INTERPRETATION")
    report.append("-" * 80)
    report.append(
        "AI systems do not merely read data. They interpret patterns. If products, SKUs, "
        "customers, categories, demand signals, or transaction values are poorly structured, "
        "the model may learn the wrong business logic."
    )
    report.append("")
    report.append("Potential executive consequences:")
    report.append("- Forecasts may appear mathematically correct while being commercially wrong.")
    report.append("- Pricing recommendations may be based on broken product grouping.")
    report.append("- Inventory decisions may overstock weak items and understock strategic items.")
    report.append("- Customer or SKU behaviour may be misread because segmentation is too broad.")
    report.append("- AI dashboards may create executive confidence without executive intelligence.")

    report.append("")
    report.append("RECOMMENDED NEXT ACTIONS")
    report.append("-" * 80)
    report.append("1. Clean missing, duplicate, vague, and inconsistent records.")
    report.append("2. Create or repair SKU, product, category, customer, and demand segmentation.")
    report.append("3. Separate transactional fields from analytical fields.")
    report.append("4. Build an ABC/XYZ or demand-behaviour segmentation layer before AI modelling.")
    report.append("5. Validate AI outputs externally before embedding decisions inside ERP workflows.")

    return "\n".join(report), exposure_score, band


def main():
    print("=" * 80)
    print("AI RISK REPORT GENERATOR")
    print("=" * 80)
    print("This tool reads a CSV file and produces an executive AI Risk Report.")
    print("")

    file_path = input("Enter full path to CSV file: ").strip().strip('"').strip("'")

    if not file_path:
        print("No file path entered. Exiting.")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Could not read CSV file: {e}")
        sys.exit(1)

    report, exposure_score, band = generate_report(df, file_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"AI_Risk_Report_{timestamp}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print("")
    print(report)
    print("")
    print("=" * 80)
    print(f"Report saved as: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
