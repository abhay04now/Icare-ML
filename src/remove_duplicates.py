import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/cleaned/final_unique_euid_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/cleaned/paired_iop_cct_round1_deduplicated.csv"
)


# ============================================================
# 1. LOAD ROUND 1 DATASET
# ============================================================

print("=" * 70)
print("ROUND 1 — DUPLICATE REMOVAL")
print("=" * 70)

print(f"\nLoading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "uid",
    "date",
    "eye",
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
# 3. CREATE EUID
#
# L + UID = left eye
# R + UID = right eye
#
# Examples:
#   uid 23001112 + LE -> L23001112
#   uid 23001112 + RE -> R23001112
# ============================================================

df["eye"] = df["eye"].str.upper().str.strip()

invalid_eye_values = ~df["eye"].isin(["LE", "RE"])

if invalid_eye_values.any():
    invalid_values = df.loc[
        invalid_eye_values,
        "eye"
    ].unique()

    raise ValueError(
        f"Invalid eye values found: {invalid_values}"
    )

df["euid"] = (
    df["eye"].map({
        "LE": "L",
        "RE": "R"
    })
    + df["uid"].astype(str)
)


# ============================================================
# 4. REORDER COLUMNS
# ============================================================

df = df[
    [
        "date",
        "euid",
        "icare_iop",
        "cct",
        "applanation_iop",
    ]
].copy()


# ============================================================
# 5. FIND EXACT DUPLICATES
#
# An exact duplicate means ALL five fields are identical:
#
#   date
#   euid
#   icare_iop
#   cct
#   applanation_iop
# ============================================================

duplicate_columns = [
    "date",
    "euid",
    "icare_iop",
    "cct",
    "applanation_iop",
]

duplicate_mask = df.duplicated(
    subset=duplicate_columns,
    keep=False,
)

duplicates = df.loc[
    duplicate_mask
].sort_values(
    duplicate_columns
)

print("\n" + "=" * 70)
print("EXACT DUPLICATES")
print("=" * 70)

print(
    f"Rows belonging to duplicate groups: "
    f"{len(duplicates):,}"
)

print(
    f"Rows that will actually be removed: "
    f"{duplicate_mask.sum() - duplicates.drop_duplicates(
        subset=duplicate_columns
    ).shape[0]:,}"
)

if len(duplicates) > 0:
    print("\nDuplicate rows:")
    print(
        duplicates.to_string(index=False)
    )
else:
    print("\nNo exact duplicates found.")


# ============================================================
# 6. REMOVE EXACT DUPLICATES
#
# keep="first" means:
# if two rows are completely identical,
# retain one and remove the copies.
# ============================================================

before_deduplication = len(df)

df = df.drop_duplicates(
    subset=duplicate_columns,
    keep="first",
).copy()

after_deduplication = len(df)

removed_count = (
    before_deduplication
    - after_deduplication
)

print(
    f"\nExact duplicate rows removed: "
    f"{removed_count:,}"
)

print(
    f"Rows remaining: "
    f"{after_deduplication:,}"
)


# ============================================================
# 7. VERIFY NO EXACT DUPLICATES REMAIN
# ============================================================

remaining_duplicates = df.duplicated(
    subset=duplicate_columns
).sum()

if remaining_duplicates != 0:
    raise ValueError(
        "Exact duplicates still remain."
    )

print("\nExact duplicate verification: PASSED")


# ============================================================
# 8. FIND NON-IDENTICAL REPEATED MEASUREMENTS
#
# Same date + euid, but different measurements.
#
# These are NOT removed.
# We only report them for investigation.
# ============================================================

group_counts = (
    df.groupby(
        ["date", "euid"]
    )
    .size()
)

repeated_groups = group_counts[
    group_counts > 1
]

print("\n" + "=" * 70)
print("NON-IDENTICAL REPEATED MEASUREMENTS")
print("=" * 70)

print(
    f"Date + EUID groups with multiple observations: "
    f"{len(repeated_groups):,}"
)

if len(repeated_groups) > 0:

    repeated_index = repeated_groups.index

    repeated_measurements = (
        df.set_index(
            ["date", "euid"]
        )
        .loc[repeated_index]
        .reset_index()
    )

    print(
        "\nThese observations are being KEPT "
        "because they are not exact duplicates:"
    )

    print(
        repeated_measurements.to_string(
            index=False
        )
    )

else:
    print(
        "\nNo non-identical repeated measurements "
        "found for the same date + EUID."
    )


# ============================================================
# 9. FINAL BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(f"Final rows: {len(df):,}")

print("\nMissing values:")
print(df.isna().sum())

if df.isna().sum().sum() != 0:
    raise ValueError(
        "NULL values exist in the final dataset."
    )

numeric_columns = [
    "icare_iop",
    "cct",
    "applanation_iop",
]

zero_count = (
    df[numeric_columns] == 0
).sum().sum()

print(
    f"\nZero measurement values: {zero_count}"
)

if zero_count != 0:
    raise ValueError(
        "Zero values exist in the final dataset."
    )


# ============================================================
# 10. SAVE DATASET
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("DEDUPLICATION COMPLETE")
print("=" * 70)

print(f"Output: {OUTPUT_FILE}")

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(
    df.head(10).to_string(index=False)
)