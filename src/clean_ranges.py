import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/cleaned/paired_iop_cct_round1_deduplicated.csv"
)

OUTPUT_FILE = Path(
    "data/cleaned/final_cleaned_dataset.csv"
)

INVALID_FILE = Path(
    "data/cleaned/invalid_ranges.csv"
)


# ============================================================
# VALID RANGES
#
# CCT:
#   480–580 µm inclusive
#
# iCare IOP:
#   > 0 and <= 80 mmHg
#
# Applanation/GAT IOP:
#   > 0 and <= 80 mmHg
# ============================================================

CCT_MIN = 480
CCT_MAX = 580

IOP_MIN = 0
IOP_MAX = 80


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("FINAL RANGE CLEANING")
print("=" * 70)

print(f"\nLoading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)

print(f"Rows loaded: {original_rows:,}")
print(f"Columns: {df.columns.tolist()}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "euid",
    "icare_iop",
    "cct",
    "applanation_iop",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 3. ENSURE MEASUREMENTS ARE NUMERIC
# ============================================================

numeric_columns = [
    "icare_iop",
    "cct",
    "applanation_iop",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 4. IDENTIFY INVALID RANGES
# ============================================================

invalid_cct = (
    (df["cct"] < CCT_MIN) |
    (df["cct"] > CCT_MAX)
)

invalid_icare = (
    (df["icare_iop"] <= IOP_MIN) |
    (df["icare_iop"] > IOP_MAX)
)

invalid_applanation = (
    (df["applanation_iop"] <= IOP_MIN) |
    (df["applanation_iop"] > IOP_MAX)
)


# ============================================================
# 5. CREATE OVERALL INVALID MASK
# ============================================================

invalid_rows = (
    invalid_cct
    | invalid_icare
    | invalid_applanation
)


# ============================================================
# 6. CREATE REJECTION REASON
# ============================================================

def get_rejection_reason(row):
    reasons = []

    if row["cct"] < CCT_MIN:
        reasons.append("CCT below 480  ")

    elif row["cct"] > CCT_MAX:
        reasons.append("CCT above 580")

    if row["icare_iop"] <= IOP_MIN:
        reasons.append("iCare IOP <= 0")

    elif row["icare_iop"] > IOP_MAX:
        reasons.append("iCare IOP > 80")

    if row["applanation_iop"] <= IOP_MIN:
        reasons.append("Applanation IOP <= 0")

    elif row["applanation_iop"] > IOP_MAX:
        reasons.append("Applanation IOP > 80")

    return "; ".join(reasons)


invalid_data = df.loc[
    invalid_rows
].copy()

if len(invalid_data) > 0:
    invalid_data["rejection_reason"] = (
        invalid_data.apply(
            get_rejection_reason,
            axis=1
        )
    )


# ============================================================
# 7. REMOVE INVALID ROWS
# ============================================================

cleaned_df = df.loc[
    ~invalid_rows
].copy()


# ============================================================
# 8. REPORT RANGE VIOLATIONS
# ============================================================

print("\n" + "=" * 70)
print("RANGE VALIDATION")
print("=" * 70)

print(
    f"\nInvalid CCT rows: "
    f"{invalid_cct.sum():,}"
)

print(
    f"Invalid iCare IOP rows: "
    f"{invalid_icare.sum():,}"
)

print(
    f"Invalid Applanation IOP rows: "
    f"{invalid_applanation.sum():,}"
)

print(
    f"Total rows containing at least one invalid value: "
    f"{invalid_rows.sum():,}"
)

print(
    f"Valid rows remaining: "
    f"{len(cleaned_df):,}"
)


# ============================================================
# 9. SAVE INVALID OBSERVATIONS
# ============================================================

INVALID_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

invalid_data.to_csv(
    INVALID_FILE,
    index=False
)


# ============================================================
# 10. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

# Check that no invalid ranges remain.

assert not (
    (cleaned_df["cct"] < CCT_MIN)
    | (cleaned_df["cct"] > CCT_MAX)
).any()

assert not (
    (cleaned_df["icare_iop"] <= IOP_MIN)
    | (cleaned_df["icare_iop"] > IOP_MAX)
).any()

assert not (
    (cleaned_df["applanation_iop"] <= IOP_MIN)
    | (cleaned_df["applanation_iop"] > IOP_MAX)
).any()

# Check for NULLs.

print("\nMissing values:")
print(cleaned_df.isna().sum())

assert cleaned_df.isna().sum().sum() == 0

# Check exact duplicates.

duplicate_count = cleaned_df.duplicated().sum()

print(
    f"\nExact duplicate rows: "
    f"{duplicate_count}"
)

assert duplicate_count == 0


# ============================================================
# 11. SAVE FINAL DATASET
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

cleaned_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING COMPLETE")
print("=" * 70)

print(
    f"\nOriginal rows       : {original_rows:,}"
)

print(
    f"Invalid rows removed: {invalid_rows.sum():,}"
)

print(
    f"Final rows          : {len(cleaned_df):,}"
)

print(
    f"Rows retained       : "
    f"{len(cleaned_df) / original_rows * 100:.2f}%"
)

print("\nFinal columns:")
print(cleaned_df.columns.tolist())

print("\nFinal statistics:")
print(
    cleaned_df[
        [
            "icare_iop",
            "cct",
            "applanation_iop",
        ]
    ].describe()
)

print("\nFiles created:")
print(f"  Valid data   : {OUTPUT_FILE}")
print(f"  Rejected data: {INVALID_FILE}")